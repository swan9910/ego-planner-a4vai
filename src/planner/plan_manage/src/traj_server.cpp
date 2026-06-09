#include "bspline_opt/uniform_bspline.h"
#include "nav_msgs/msg/odometry.hpp"
#include "traj_utils/msg/bspline.hpp"
#include "quadrotor_msgs/msg/position_command.hpp"
#include "std_msgs/msg/empty.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <mutex>
#include <rclcpp/rclcpp.hpp>

rclcpp::Publisher<quadrotor_msgs::msg::PositionCommand>::SharedPtr pos_cmd_pub;

quadrotor_msgs::msg::PositionCommand cmd;
double pos_gain[3] = {0, 0, 0};
double vel_gain[3] = {0, 0, 0};

using ego_planner::UniformBspline;

bool receive_traj_ = false;
vector<UniformBspline> traj_;
double traj_duration_;
rclcpp::Time start_time_;
int traj_id_;

// yaw control
double last_yaw_, last_yaw_dot_;
double time_forward_;

// ── clearance 기반 속도 스케일링 ──────────────────────────────
pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_(new pcl::PointCloud<pcl::PointXYZ>);
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_;
bool has_cloud_ = false;
std::mutex cloud_mutex_;

double d_safe_;         // 이 거리 이상이면 최대 속도 (최근접 점 기준)
double d_min_;          // 이 거리 이하면 최저 속도
double v_min_factor_;   // 최저 속도 비율 (0~1)
double lookahead_time_; // 전방 예측 시간 (초)
double density_radius_; // 밀도 탐색 반경 (m)
int    density_max_;    // 이 개수 이상이면 최저 속도

// 가상 시간 (속도 스케일에 맞춰 궤적 위를 천천히/빠르게 이동)
double    t_virtual_       = 0.0;
rclcpp::Time t_last_cmd_;
bool      t_last_init_     = false;
// ─────────────────────────────────────────────────────────────

void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstPtr &msg)
{
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::fromROSMsg(*msg, *cloud);
  if (cloud->empty()) return;

  std::lock_guard<std::mutex> lock(cloud_mutex_);
  latest_cloud_ = cloud;
  kdtree_.setInputCloud(latest_cloud_);
  has_cloud_ = true;
}

// 밀도 기반 속도 팩터: 반경 내 장애물 점 개수로 판단
// 점이 많을수록(밀집) → 좁은 공간 → 느리게
double computeSpeedFactor(const Eigen::Vector3d &pos)
{
  std::lock_guard<std::mutex> lock(cloud_mutex_);
  if (!has_cloud_ || latest_cloud_->empty()) return 1.0;

  pcl::PointXYZ query;
  query.x = static_cast<float>(pos(0));
  query.y = static_cast<float>(pos(1));
  query.z = static_cast<float>(pos(2));

  // 1) 최근접 점이 d_safe 밖이면 최대 속도
  std::vector<int>   nn_idx(1);
  std::vector<float> nn_dist(1);
  if (kdtree_.nearestKSearch(query, 1, nn_idx, nn_dist) > 0)
  {
    double nearest = std::sqrt(static_cast<double>(nn_dist[0]));
    if (nearest >= d_safe_) return 1.0;
    if (nearest <= d_min_)  return v_min_factor_;
  }
  else
  {
    return 1.0;
  }

  // 2) 반경 density_radius_ 내 점 개수로 밀도 계산
  std::vector<int>   r_idx;
  std::vector<float> r_dist;
  int count = kdtree_.radiusSearch(query, density_radius_, r_idx, r_dist);

  // 밀도 비율: 0 → 점 없음(넓음), 1 → density_max 이상(좁음)
  double density_ratio = std::min(static_cast<double>(count) / density_max_, 1.0);

  // 밀도가 높을수록 속도 낮게 (선형 보간)
  return 1.0 - (1.0 - v_min_factor_) * density_ratio;
}

void bsplineCallback(traj_utils::msg::Bspline::ConstPtr msg)
{
  Eigen::MatrixXd pos_pts(3, msg->pos_pts.size());
  Eigen::VectorXd knots(msg->knots.size());

  for (size_t i = 0; i < msg->knots.size(); ++i)
    knots(i) = msg->knots[i];

  for (size_t i = 0; i < msg->pos_pts.size(); ++i)
  {
    pos_pts(0, i) = msg->pos_pts[i].x;
    pos_pts(1, i) = msg->pos_pts[i].y;
    pos_pts(2, i) = msg->pos_pts[i].z;
  }

  UniformBspline pos_traj(pos_pts, msg->order, 0.1);
  pos_traj.setKnot(knots);

  start_time_    = msg->start_time;
  traj_id_       = msg->traj_id;

  traj_.clear();
  traj_.push_back(pos_traj);
  traj_.push_back(traj_[0].getDerivative());
  traj_.push_back(traj_[1].getDerivative());

  traj_duration_ = traj_[0].getTimeSum();

  // 새 궤적 수신 시 가상 시간 초기화
  t_virtual_   = 0.0;
  t_last_init_ = false;

  receive_traj_ = true;
}

std::pair<double, double> calculate_yaw(double t_cur, Eigen::Vector3d &pos,
                                        rclcpp::Time &time_now,
                                        rclcpp::Time &time_last)
{
  constexpr double PI = 3.1415926;
  constexpr double YAW_DOT_MAX_PER_SEC = PI;
  std::pair<double, double> yaw_yawdot(0, 0);
  double yaw = 0, yawdot = 0;

  Eigen::Vector3d dir =
      t_cur + time_forward_ <= traj_duration_
          ? traj_[0].evaluateDeBoorT(t_cur + time_forward_) - pos
          : traj_[0].evaluateDeBoorT(traj_duration_) - pos;

  double yaw_temp = dir.norm() > 0.1 ? atan2(dir(1), dir(0)) : last_yaw_;
  double max_yaw_change = YAW_DOT_MAX_PER_SEC * (time_now - time_last).seconds();

  if (yaw_temp - last_yaw_ > PI)
  {
    if (yaw_temp - last_yaw_ - 2 * PI < -max_yaw_change)
    {
      yaw = last_yaw_ - max_yaw_change;
      if (yaw < -PI) yaw += 2 * PI;
      yawdot = -YAW_DOT_MAX_PER_SEC;
    }
    else
    {
      yaw = yaw_temp;
      yawdot = (yaw - last_yaw_ > PI)
                   ? -YAW_DOT_MAX_PER_SEC
                   : (yaw_temp - last_yaw_) / (time_now - time_last).seconds();
    }
  }
  else if (yaw_temp - last_yaw_ < -PI)
  {
    if (yaw_temp - last_yaw_ + 2 * PI > max_yaw_change)
    {
      yaw = last_yaw_ + max_yaw_change;
      if (yaw > PI) yaw -= 2 * PI;
      yawdot = YAW_DOT_MAX_PER_SEC;
    }
    else
    {
      yaw = yaw_temp;
      yawdot = (yaw - last_yaw_ < -PI)
                   ? YAW_DOT_MAX_PER_SEC
                   : (yaw_temp - last_yaw_) / (time_now - time_last).seconds();
    }
  }
  else
  {
    if (yaw_temp - last_yaw_ < -max_yaw_change)
    {
      yaw = last_yaw_ - max_yaw_change;
      if (yaw < -PI) yaw += 2 * PI;
      yawdot = -YAW_DOT_MAX_PER_SEC;
    }
    else if (yaw_temp - last_yaw_ > max_yaw_change)
    {
      yaw = last_yaw_ + max_yaw_change;
      if (yaw > PI) yaw -= 2 * PI;
      yawdot = YAW_DOT_MAX_PER_SEC;
    }
    else
    {
      yaw    = yaw_temp;
      yawdot = (yaw - last_yaw_ > PI)    ? -YAW_DOT_MAX_PER_SEC
             : (yaw - last_yaw_ < -PI)   ?  YAW_DOT_MAX_PER_SEC
             : (yaw_temp - last_yaw_) / (time_now - time_last).seconds();
    }
  }

  if (fabs(yaw - last_yaw_) <= max_yaw_change)
    yaw = 0.5 * last_yaw_ + 0.5 * yaw;
  yawdot     = 0.5 * last_yaw_dot_ + 0.5 * yawdot;
  last_yaw_     = yaw;
  last_yaw_dot_ = yawdot;

  yaw_yawdot.first  = yaw;
  yaw_yawdot.second = yawdot;
  return yaw_yawdot;
}

void cmdCallback()
{
  if (!receive_traj_) return;

  rclcpp::Clock clock(RCL_ROS_TIME);
  rclcpp::Time time_now = clock.now();

  // ── 가상 시간 진행 ────────────────────────────────────────
  if (!t_last_init_)
  {
    t_last_cmd_  = time_now;
    t_last_init_ = true;
  }
  double dt_real = (time_now - t_last_cmd_).seconds();
  t_last_cmd_ = time_now;

  // 전방 lookahead 위치에서 밀도 기반 speed factor 계산
  double t_ahead    = std::min(traj_duration_, t_virtual_ + lookahead_time_);
  Eigen::Vector3d pos_ahead = traj_[0].evaluateDeBoorT(t_ahead);
  double speed_factor = computeSpeedFactor(pos_ahead);

  // 가상 시간을 speed_factor 비율로 진행
  t_virtual_ = std::min(t_virtual_ + dt_real * speed_factor, traj_duration_);
  double t_cur = t_virtual_;
  // ─────────────────────────────────────────────────────────

  Eigen::Vector3d pos(Eigen::Vector3d::Zero()),
                  vel(Eigen::Vector3d::Zero()),
                  acc(Eigen::Vector3d::Zero()),
                  pos_f;
  std::pair<double, double> yaw_yawdot(0, 0);

  static rclcpp::Time time_last = clock.now();

  if (t_cur < traj_duration_ && t_cur >= 0.0)
  {
    pos = traj_[0].evaluateDeBoorT(t_cur);
    vel = traj_[1].evaluateDeBoorT(t_cur) * speed_factor;
    acc = traj_[2].evaluateDeBoorT(t_cur) * speed_factor * speed_factor;

    yaw_yawdot = calculate_yaw(t_cur, pos, time_now, time_last);

    double tf = min(traj_duration_, t_cur + 2.0);
    pos_f = traj_[0].evaluateDeBoorT(tf);
  }
  else if (t_cur >= traj_duration_)
  {
    pos = traj_[0].evaluateDeBoorT(traj_duration_);
    vel.setZero();
    acc.setZero();

    yaw_yawdot.first  = last_yaw_;
    yaw_yawdot.second = 0;
    pos_f = pos;
  }
  else
  {
    cout << "[Traj server]: invalid time." << endl;
  }
  time_last = time_now;

  cmd.header.stamp     = time_now;
  cmd.header.frame_id  = "world";
  cmd.trajectory_flag  = quadrotor_msgs::msg::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id    = traj_id_;

  cmd.position.x = pos(0);
  cmd.position.y = pos(1);
  cmd.position.z = pos(2);

  cmd.velocity.x = vel(0);
  cmd.velocity.y = vel(1);
  cmd.velocity.z = vel(2);

  cmd.acceleration.x = acc(0);
  cmd.acceleration.y = acc(1);
  cmd.acceleration.z = acc(2);

  cmd.yaw     = yaw_yawdot.first;
  cmd.yaw_dot = yaw_yawdot.second;

  last_yaw_ = cmd.yaw;

  pos_cmd_pub->publish(cmd);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("traj_server");

  auto bspline_sub = node->create_subscription<traj_utils::msg::Bspline>(
      "planning/bspline", 10, bsplineCallback);

  // 장애물 inflated 맵 구독 (GridMap이 발행하는 occupancy_inflate)
  auto cloud_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
      "grid_map/occupancy_inflate", 10, cloudCallback);

  pos_cmd_pub = node->create_publisher<quadrotor_msgs::msg::PositionCommand>(
      "/position_cmd", 50);

  auto cmd_timer = node->create_wall_timer(
      std::chrono::milliseconds(10), cmdCallback);

  cmd.kx[0] = pos_gain[0];
  cmd.kx[1] = pos_gain[1];
  cmd.kx[2] = pos_gain[2];

  cmd.kv[0] = vel_gain[0];
  cmd.kv[1] = vel_gain[1];
  cmd.kv[2] = vel_gain[2];

  // 기존 파라미터
  node->declare_parameter("traj_server/time_forward", -1.0);
  node->get_parameter("traj_server/time_forward", time_forward_);

  // 속도 스케일링 파라미터
  node->declare_parameter("traj_server/d_safe",          2.0);
  node->declare_parameter("traj_server/d_min",           0.5);
  node->declare_parameter("traj_server/v_min_factor",    0.3);
  node->declare_parameter("traj_server/lookahead_time",  1.0);
  node->declare_parameter("traj_server/density_radius",  3.0);
  node->declare_parameter("traj_server/density_max",     200);

  node->get_parameter("traj_server/d_safe",          d_safe_);
  node->get_parameter("traj_server/d_min",           d_min_);
  node->get_parameter("traj_server/v_min_factor",    v_min_factor_);
  node->get_parameter("traj_server/lookahead_time",  lookahead_time_);
  node->get_parameter("traj_server/density_radius",  density_radius_);
  node->get_parameter("traj_server/density_max",     density_max_);

  RCLCPP_INFO(node->get_logger(),
              "[Traj server] Speed scaling: d_safe=%.1f d_min=%.1f v_min=%.1f "
              "lookahead=%.1f density_r=%.1f density_max=%d",
              d_safe_, d_min_, v_min_factor_, lookahead_time_,
              density_radius_, density_max_);

  last_yaw_     = 0.0;
  last_yaw_dot_ = 0.0;

  rclcpp::sleep_for(std::chrono::seconds(1));
  RCLCPP_WARN(node->get_logger(), "[Traj server]: ready.");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
