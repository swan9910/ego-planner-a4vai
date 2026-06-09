#!/usr/bin/env python3
import rclpy
import rclpy.parameter
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry, VehicleStatus, VehicleLocalPosition, FusionWeight
from traj_utils.srv import SetGoal
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import Int32
from custom_msgs.msg import LocalWaypointSetpoint
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

class EgoPlannerPX4Controller(Node):

    def __init__(self) -> None:
        super().__init__('ego_planner_px4_controller')

        # 파라미터 선언
        self.declare_parameter('yaw_offset_deg', 90.0)  # Gazebo Baylands용 기본값: 90도
        self.declare_parameter('takeoff_alt', 5.0)      # ENU 양수 (m) - 이륙 고도
        # 통합 goal (PF→CA 전환 시 ego service replan 용)
        self.declare_parameter('goal_x', 70.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_z', 4.0)
        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.goal_z = float(self.get_parameter('goal_z').value)
        self.yaw_offset_rad = math.radians(self.get_parameter('yaw_offset_deg').value)
        self._takeoff_alt_enu = float(self.get_parameter('takeoff_alt').value)
        self.get_logger().info(f'Yaw offset: {self.get_parameter("yaw_offset_deg").value} degrees')
        self.get_logger().info(f'Takeoff altitude (ENU): {self._takeoff_alt_enu} m')

        # PX4 퍼블리셔 (qos_profile_sensor_data = BEST_EFFORT + VOLATILE, RealGazebo 호환)
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/vehicle1/fmu/in/offboard_control_mode', qos_profile_sensor_data)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/vehicle1/fmu/in/trajectory_setpoint', qos_profile_sensor_data)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/vehicle1/fmu/in/vehicle_command', qos_profile_sensor_data)

        # PX4 서브스크라이버
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/vehicle1/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile_sensor_data)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/vehicle1/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile_sensor_data)
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/vehicle1/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile_sensor_data)

        # EGO-Planner 서브스크라이버
        self.ego_planner_sub = self.create_subscription(
            PositionCommand,
            '/planning/pos_cmd',
            self.ego_planner_callback,
            10
        )

        # === 통합 switching: fusion_weight 구독 ===
        self.PF_FUSION_THRESH = 0.9
        self.fusion_weight_current = 0.0
        self.fusion_weight_prev_pf = False   # 이전 tick PF mode 여부
        self.ned_x_current = 0.0
        self.ned_y_current = 0.0
        self.ned_z_current = 0.0
        self.have_local_pos = False
        self.STARTUP_GRACE_SEC = 5.0
        self.t_node_start = self.get_clock().now()
        # PF→CA 전환 후 새 ego trajectory 도달까지 점진 감속 (급정거 방지)
        self.PF_TO_CA_HOLD_SEC = 2.0    # PF→CA crossfade 시간 (진입속도 → ego속도 blend)
        self.t_pf_to_ca = None    # 전환 시각
        self.ca_entry_vel_ned = [0.0, 0.0, 0.0]  # PF→CA 진입 순간 NED 속도 (감속 ramp 시작값)
        self.t_ca_hover_start = None         # PF→CA 전환 시 hover 시작 시각
        self.CA_HOVER_DURATION = 1.5         # 전환 시 hover 시간 (s) — ego replan 대기
        self.ca_hover_pos = None             # hover 유지 위치 (NED)
        self.fusion_weight_sub = self.create_subscription(
            FusionWeight, '/vehicle1/fmu/in/fusion_weight',
            self._fusion_weight_callback, qos_profile_sensor_data)
        # ego replan 서비스 클라이언트
        self.ego_setgoal_cli = self.create_client(SetGoal, '/ego_planner/set_goal')

        # === PF wp 동기화: heading_idx 받아서 ego goal 을 wp[idx] 로 set ===
        # wp.csv 는 PF NED (x=north, y=east, z=alt positive up)
        # ego service goal 은 ENU (x=east, y=north, z=alt positive up)
        # 변환: ego_goal = (wp.y, wp.x, wp.z)
        self.pf_wps = []                  # list of (x_N, y_E, alt) PF NED
        self.heading_wp_idx = -1
        self.last_ego_goal_idx = -1       # 마지막으로 ego 에 보낸 wp idx (중복 service call 방지)
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        wp_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.wp_qos = wp_qos
        self.create_subscription(LocalWaypointSetpoint,
                                 '/local_waypoint_setpoint_to_PF',
                                 self._pf_wp_callback, wp_qos)
        self.create_subscription(Int32, '/heading_waypoint_index',
                                 self._heading_idx_callback, 10)
        # wp skip 시 PF 에게 새 wp 리스트 발행할 publisher
        self.wp_skip_pub = self.create_publisher(
            LocalWaypointSetpoint, '/local_waypoint_setpoint_to_PF', wp_qos)

        # === Stuck-timeout wp skip (drone 움직임 기반) ===
        # drone 의 최근 N초 위치 분산이 작으면 stuck
        self.STUCK_TIMEOUT_SEC   = 10.0   # 이 시간 동안 거의 안 움직이면 stuck (20→10: false-stuck hover 시간 단축)
        self.STUCK_MOVE_MIN      = 5.0    # 이 거리 미만 이동 = stuck
        self.STUCK_INIT_GRACE    = 5.0    # idx 바뀐 직후엔 잠시 skip 안 함
        self.t_idx_change = None
        self.pos_history = []             # [(t, ned_x, ned_y, ned_z), ...] 최근 N초만 유지
        self.create_timer(1.0, self._stuck_check)

        # EGO-Planner용 퍼블리셔 (nav_msgs/Odometry) - PX4 odom을 ENU로 변환해서 퍼블리시
        self.odom_pub = self.create_publisher(Odometry, '/ego_odom', 10)
        self.odom_grid_pub = self.create_publisher(Odometry, '/ego_odom_grid', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/ego_pose', 10)

        # 상태 변수
        self.offboard_setpoint_counter = 0
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_status = VehicleStatus()

        # 드론 상태 관리
        self.STATE_INIT = 0
        self.STATE_ARMING = 1
        self.STATE_TAKEOFF = 2
        self.STATE_PLANNER_READY = 3

        self.current_state = self.STATE_INIT
        self.takeoff_height = -self._takeoff_alt_enu   # NED z (음수)
        self.tolerance = 0.5                            # 50m 비행에는 좀 더 넉넉하게

        # 제어 파라미터
        self.target_altitude = -self._takeoff_alt_enu
        self.max_velocity = 4.0

        # 현재 드론 상태 (ENU 좌표계)
        self.current_position = [0.0, 0.0, 0.0]
        self.current_velocity = [0.0, 0.0, 0.0]
        self.current_attitude = [0.0, 0.0, 0.0]

        # EGO-Planner 명령 타임아웃
        self.last_planner_cmd_time = self.get_clock().now()
        self.planner_cmd_timeout = 5.0

        # 데이터 유효성 플래그
        self.odometry_data_valid = False

        # Reset counter monitoring
        self.prev_xy_reset_counter = 0
        self.prev_z_reset_counter = 0
        self.reset_monitor_initialized = False

        # TF Broadcaster for dynamic transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Static TF: base_link → lidar sensor frame
        # SDF: <pose>0 0 -0.35 0 3.14159 0</pose> (pitch 180° 뒤집힘)
        # quaternion (qx=0, qy=1, qz=0, qw=0) = pitch π
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        lidar_tf = TransformStamped()
        lidar_tf.header.stamp = self.get_clock().now().to_msg()
        lidar_tf.header.frame_id = 'x500_0/base_link'
        lidar_tf.child_frame_id = 'x500_0/lidar_3d_link/lidar_sensor'
        lidar_tf.transform.translation.x = 0.0
        lidar_tf.transform.translation.y = 0.0
        lidar_tf.transform.translation.z = -0.35
        lidar_tf.transform.rotation.x = 0.0
        lidar_tf.transform.rotation.y = 1.0
        lidar_tf.transform.rotation.z = 0.0
        lidar_tf.transform.rotation.w = 0.0
        self.static_tf_broadcaster.sendTransform(lidar_tf)

        # 타이머 설정 (20Hz 제어)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info('EGO-Planner + PX4 Controller initialized (RealGazebo, qos_profile_sensor_data, With TF Broadcast)')

    def vehicle_odometry_callback(self, vehicle_odometry):
        """PX4 odometry를 받아서 ENU로 변환 후 EGO-Planner에 퍼블리시"""
        self.vehicle_odometry = vehicle_odometry

        # PX4 NED -> ENU 변환
        pos_x_enu = float(vehicle_odometry.position[1])  # East = NED Y
        pos_y_enu = float(vehicle_odometry.position[0])  # North = NED X
        pos_z_enu = float(-vehicle_odometry.position[2]) # Up = -NED Z

        vel_x_enu = float(vehicle_odometry.velocity[1])
        vel_y_enu = float(vehicle_odometry.velocity[0])
        vel_z_enu = float(-vehicle_odometry.velocity[2])

        # 현재 위치/속도 저장 (ENU)
        self.current_position = [pos_x_enu, pos_y_enu, pos_z_enu]
        self.current_velocity = [vel_x_enu, vel_y_enu, vel_z_enu]

        # Quaternion 변환 (PX4 NED body→world  →  REP-103 ENU body→world)
        # 정확한 변환: q_enu = R_n2e_world * q_b2n * R_b_ned2enu
        #   - R_n2e_world: NED world → ENU world (180° around (1,1,0)/√2 axis)
        #   - R_b_ned2enu: NED body (z=down, y=right) → ENU body (z=up, y=left)
        #     = 180° around x-axis (forward)
        q_ned = vehicle_odometry.q  # PX4 [w, x, y, z]
        # scipy 는 [x, y, z, w] 순서
        r_b2n = R.from_quat([q_ned[1], q_ned[2], q_ned[3], q_ned[0]])
        r_n2e_world = R.from_quat([0.7071067811865475, 0.7071067811865475, 0.0, 0.0])
        r_b_ned2enu = R.from_quat([1.0, 0.0, 0.0, 0.0])  # 180° around x
        r_b_enu_to_w_enu = r_n2e_world * r_b2n * r_b_ned2enu

        # yaw_offset 보정 (있으면 ENU world Z축 추가 회전)
        if self.yaw_offset_rad != 0.0:
            r_yaw_offset = R.from_euler('z', self.yaw_offset_rad)
            r_b_enu_to_w_enu = r_yaw_offset * r_b_enu_to_w_enu

        q_xyzw = r_b_enu_to_w_enu.as_quat()
        q_enu_x, q_enu_y, q_enu_z, q_enu_w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])

        # Attitude 추출 (RViz 표시용)
        roll, pitch, yaw_corrected = r_b_enu_to_w_enu.as_euler('xyz', degrees=False)
        self.current_attitude = [math.degrees(roll), math.degrees(pitch), math.degrees(yaw_corrected)]

        self.odometry_data_valid = True

        # EGO-Planner용 Odometry 메시지 생성 (ENU)
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = float(pos_x_enu)
        odom_msg.pose.pose.position.y = float(pos_y_enu)
        odom_msg.pose.pose.position.z = float(pos_z_enu)

        odom_msg.pose.pose.orientation.x = float(q_enu_x)
        odom_msg.pose.pose.orientation.y = float(q_enu_y)
        odom_msg.pose.pose.orientation.z = float(q_enu_z)
        odom_msg.pose.pose.orientation.w = float(q_enu_w)

        odom_msg.twist.twist.linear.x = float(vel_x_enu)
        odom_msg.twist.twist.linear.y = float(vel_y_enu)
        odom_msg.twist.twist.linear.z = float(vel_z_enu)

        odom_msg.twist.twist.angular.x = float(vehicle_odometry.angular_velocity[1])
        odom_msg.twist.twist.angular.y = float(vehicle_odometry.angular_velocity[0])
        odom_msg.twist.twist.angular.z = float(-vehicle_odometry.angular_velocity[2])

        self.odom_pub.publish(odom_msg)
        self.odom_grid_pub.publish(odom_msg)

        pose_msg = PoseStamped()
        pose_msg.header = odom_msg.header
        pose_msg.pose = odom_msg.pose.pose
        self.pose_pub.publish(pose_msg)

        # Publish TF: world -> x500_0/base_link
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'x500_0/base_link'

        t.transform.translation.x = float(pos_x_enu)
        t.transform.translation.y = float(pos_y_enu)
        t.transform.translation.z = float(pos_z_enu)

        t.transform.rotation.x = float(q_enu_x)
        t.transform.rotation.y = float(q_enu_y)
        t.transform.rotation.z = float(q_enu_z)
        t.transform.rotation.w = float(q_enu_w)

        self.tf_broadcaster.sendTransform(t)

    def vehicle_status_callback(self, vehicle_status):
        self.vehicle_status = vehicle_status

    def vehicle_local_position_callback(self, msg):
        """Reset counter 모니터링 + 현재 NED 위치 저장 (hover 명령용)"""
        # 현재 NED 위치 저장
        self.ned_x_current = msg.x
        self.ned_y_current = msg.y
        self.ned_z_current = msg.z
        self.have_local_pos = True
        if not self.reset_monitor_initialized:
            self.prev_xy_reset_counter = msg.xy_reset_counter
            self.prev_z_reset_counter = msg.z_reset_counter
            self.reset_monitor_initialized = True
            return

        if msg.xy_reset_counter != self.prev_xy_reset_counter:
            self.get_logger().warn(f'[RESET] xy: {self.prev_xy_reset_counter} -> {msg.xy_reset_counter}, delta: [{msg.delta_xy[0]:.4f}, {msg.delta_xy[1]:.4f}]')
            self.prev_xy_reset_counter = msg.xy_reset_counter

        if msg.z_reset_counter != self.prev_z_reset_counter:
            self.get_logger().warn(f'[RESET] z: {self.prev_z_reset_counter} -> {msg.z_reset_counter}, delta: {msg.delta_z:.4f}')
            self.prev_z_reset_counter = msg.z_reset_counter

    def _pf_wp_callback(self, msg):
        # wp.csv 받아서 캐싱 (PF NED 좌표계)
        n = len(msg.waypoint_x)
        new_wps = [(float(msg.waypoint_x[i]),
                    float(msg.waypoint_y[i]),
                    float(msg.waypoint_z[i])) for i in range(n)]
        if new_wps != self.pf_wps:
            self.pf_wps = new_wps
            self.get_logger().info(f'☆ wp 캐싱: {n}개 wp = {self.pf_wps}')

    def _heading_idx_callback(self, msg):
        new_idx = int(msg.data)
        if new_idx == self.heading_wp_idx:
            return
        self.get_logger().info(f'☆ heading_wp_idx: {self.heading_wp_idx} → {new_idx}')
        self.heading_wp_idx = new_idx
        # stuck detector 리셋
        self.t_idx_change = self.get_clock().now()
        self.pos_history = []
        # CA 중 idx advance → ego goal 즉시 update (fusion callback 안 와도 동기화)
        is_pf_now = (self.fusion_weight_current >= self.PF_FUSION_THRESH)
        elapsed = (self.get_clock().now() - self.t_node_start).nanoseconds / 1e9
        if (not is_pf_now) and (elapsed >= self.STARTUP_GRACE_SEC) \
           and (new_idx != self.last_ego_goal_idx) and (new_idx >= 0):
            gx, gy, gz, idx = self._current_goal_enu()
            self.get_logger().info(
                f'♢ CA 중 wp advance: ego goal update → wp[{idx}]=ENU({gx:.1f},{gy:.1f},{gz:.1f})')
            req = SetGoal.Request()
            req.goal.x = gx; req.goal.y = gy; req.goal.z = gz
            self.ego_setgoal_cli.call_async(req)
            self.last_ego_goal_idx = idx

    def _stuck_check(self):
        """주기적으로 호출: drone 이 거의 안 움직이면 (hover/stuck) wp skip"""
        if (self.current_state != self.STATE_PLANNER_READY or
                self.heading_wp_idx < 0 or
                self.heading_wp_idx >= len(self.pf_wps) or
                self.t_idx_change is None or
                not self.have_local_pos):
            return
        now = self.get_clock().now()
        elapsed = (now - self.t_idx_change).nanoseconds / 1e9
        if elapsed < self.STUCK_INIT_GRACE:
            return
        # 마지막 wp (goal) 는 절대 skip 안 함
        is_last = (self.heading_wp_idx == len(self.pf_wps) - 1)
        if is_last:
            return
        t_now = now.nanoseconds / 1e9
        self.pos_history.append((t_now, self.ned_x_current,
                                  self.ned_y_current, self.ned_z_current))
        # 최근 STUCK_TIMEOUT_SEC 만 유지
        cutoff = t_now - self.STUCK_TIMEOUT_SEC
        self.pos_history = [p for p in self.pos_history if p[0] >= cutoff]
        # 시간창이 충분히 길지 않으면 판정 보류
        if len(self.pos_history) < 2 or (self.pos_history[-1][0] - self.pos_history[0][0]) < self.STUCK_TIMEOUT_SEC * 0.9:
            return
        # 시간창 안 drone 위치 spread (bbox 대각선)
        xs = [p[1] for p in self.pos_history]
        ys = [p[2] for p in self.pos_history]
        zs = [p[3] for p in self.pos_history]
        spread = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2 + (max(zs)-min(zs))**2)
        if spread < self.STUCK_MOVE_MIN:
            wp = self.pf_wps[self.heading_wp_idx]
            self.get_logger().warn(
                f'★ STUCK at wp[{self.heading_wp_idx}] wp=NED({wp[0]:.1f},{wp[1]:.1f},alt {wp[2]:.1f})  '
                f'drone 움직임 {spread:.2f}m < {self.STUCK_MOVE_MIN}m (over {self.STUCK_TIMEOUT_SEC}s)')
            self._skip_current_wp(spread)

    def _skip_current_wp(self, current_dist):
        cur_idx = self.heading_wp_idx
        if cur_idx < 0 or cur_idx >= len(self.pf_wps) - 1:
            return  # 마지막 wp 또는 invalid
        # 새 wp 리스트: [drone 현재 위치] + 그 다음 wp 들 (cur_idx 빼고)
        drone_alt = -self.ned_z_current
        new_wps = [(float(self.ned_x_current), float(self.ned_y_current), float(drone_alt))]
        for i in range(cur_idx + 1, len(self.pf_wps)):
            new_wps.append(self.pf_wps[i])
        self.get_logger().warn(
            f'★ STUCK at wp[{cur_idx}] (spread {current_dist:.2f}m < {self.STUCK_MOVE_MIN}m '
            f'over {self.STUCK_TIMEOUT_SEC}s) → skip, new wp list ({len(new_wps)} wps)')
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = False  # PF reWP_flag 트리거 → idx=1 reset
        msg.waypoint_x = [w[0] for w in new_wps]
        msg.waypoint_y = [w[1] for w in new_wps]
        msg.waypoint_z = [w[2] for w in new_wps]
        self.wp_skip_pub.publish(msg)
        # local cache 도 update (다음 heading_idx 받기 전까지)
        self.pf_wps = new_wps
        self.heading_wp_idx = -1  # PF 가 새 idx 발행 기다림
        self.t_idx_change = None

    def _rebase_pf_path(self):
        """CA→PF 전환 시 PF 경로를 drone 현재 위치 기준으로 rebase.
        새 wp[0] = drone 현재 위치 → cross-track ≈ 0 → PF 가 경로로 수직 yank 안 함.
        heading wp 는 유지 (skip 과 달리 현재 목표 wp 안 버림)."""
        cur_idx = self.heading_wp_idx
        if cur_idx < 0 or cur_idx >= len(self.pf_wps) or not self.have_local_pos:
            return
        drone_alt = -self.ned_z_current
        new_wps = [(float(self.ned_x_current), float(self.ned_y_current), float(drone_alt))]
        for i in range(cur_idx, len(self.pf_wps)):   # heading wp 포함 (skip 과 차이)
            new_wps.append(self.pf_wps[i])
        self.get_logger().info(
            f'☆ CA→PF: PF 경로 rebase (wp[0]=drone현재, heading wp[{cur_idx}] 유지, {len(new_wps)} wps)')
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = False   # reWP_flag → PF idx=1 reset, cross-track 0
        msg.waypoint_x = [w[0] for w in new_wps]
        msg.waypoint_y = [w[1] for w in new_wps]
        msg.waypoint_z = [w[2] for w in new_wps]
        self.wp_skip_pub.publish(msg)
        self.pf_wps = new_wps
        self.heading_wp_idx = -1
        self.t_idx_change = None

    def _current_goal_enu(self):
        """현재 ego goal 결정: heading_idx 가 valid 면 wp[idx], 아니면 fallback param"""
        if 0 <= self.heading_wp_idx < len(self.pf_wps):
            wp_n, wp_e, wp_alt = self.pf_wps[self.heading_wp_idx]
            # PF NED (N, E, alt) → ENU (east, north, alt)
            return float(wp_e), float(wp_n), float(wp_alt), self.heading_wp_idx
        return float(self.goal_x), float(self.goal_y), float(self.goal_z), -1

    def _fusion_weight_callback(self, msg):
        new_w = float(msg.fusion_weight)
        is_pf_now = (new_w >= self.PF_FUSION_THRESH)
        # startup grace 기간 동안 transition 무시 (drone takeoff + ego init 안정화 보호)
        elapsed_since_start = (self.get_clock().now() - self.t_node_start).nanoseconds / 1e9
        if elapsed_since_start < self.STARTUP_GRACE_SEC:
            self.fusion_weight_prev_pf = is_pf_now   # state 만 sync, 동작 없음
            self.fusion_weight_current = new_w
            return
        # CA→PF 전환: ego goal clear + PF 경로 rebase (cross-track yank 방지)
        if (not self.fusion_weight_prev_pf) and is_pf_now:
            ready = self.ego_setgoal_cli.service_is_ready()
            self.get_logger().info(f'☆ CA→PF: ego goal clear (service_ready={ready})')
            req = SetGoal.Request()
            req.goal.x = 0.0; req.goal.y = 0.0; req.goal.z = -999.0  # sentinel = clear
            self.ego_setgoal_cli.call_async(req)
            self.last_ego_goal_idx = -1
            self._rebase_pf_path()   # CA→PF cross-track yank 방지 (A/B 검증: roll -42%, stuck 4→0)
        # PF→CA 전환: 즉시 ego replan + 짧은 hold (new trajectory 도달 대기)
        if self.fusion_weight_prev_pf and not is_pf_now:
            ready = self.ego_setgoal_cli.service_is_ready()
            gx, gy, gz, idx = self._current_goal_enu()
            self.get_logger().info(
                f'★ PF→CA: ego replan to wp[{idx}]=ENU({gx:.1f},{gy:.1f},{gz:.1f}) '
                f'+ {self.PF_TO_CA_HOLD_SEC}s hold (service_ready={ready})')
            req = SetGoal.Request()
            req.goal.x = gx; req.goal.y = gy; req.goal.z = gz
            self.ego_setgoal_cli.call_async(req)
            self.last_ego_goal_idx = idx
            self.t_pf_to_ca = self.get_clock().now()
            # 진입 순간 NED 속도 캡처 (ENU current_velocity → NED): 감속 ramp 시작값
            self.ca_entry_vel_ned = [float(self.current_velocity[1]),   # north = ENU y
                                     float(self.current_velocity[0]),   # east  = ENU x
                                     float(-self.current_velocity[2])]  # down  = -ENU z
        # CA 중 heading_idx 가 advance 했으면 ego goal 도 update
        elif (not is_pf_now) and (self.heading_wp_idx != self.last_ego_goal_idx) and (self.heading_wp_idx >= 0):
            gx, gy, gz, idx = self._current_goal_enu()
            self.get_logger().info(
                f'♢ CA 중 wp advance: ego goal update → wp[{idx}]=ENU({gx:.1f},{gy:.1f},{gz:.1f})')
            req = SetGoal.Request()
            req.goal.x = gx; req.goal.y = gy; req.goal.z = gz
            self.ego_setgoal_cli.call_async(req)
            self.last_ego_goal_idx = idx
        self.fusion_weight_prev_pf = is_pf_now
        self.fusion_weight_current = new_w

    def ego_planner_callback(self, cmd_msg):
        if self.current_state != self.STATE_PLANNER_READY:
            return

        self.last_planner_cmd_time = self.get_clock().now()

        # PF→CA crossfade window: 멈추지 않고 진입속도 → ego 명령으로 부드럽게 blend
        blend_frac = None   # None=blend 안 함, 0~1 = ego 명령 비중
        if self.t_pf_to_ca is not None:
            elapsed = (self.get_clock().now() - self.t_pf_to_ca).nanoseconds / 1e9
            if elapsed < self.PF_TO_CA_HOLD_SEC:
                blend_frac = elapsed / self.PF_TO_CA_HOLD_SEC   # 0 → 1
            else:
                self.t_pf_to_ca = None
                self.get_logger().info('☆ crossfade 종료, ego trajectory 추종 시작')

        # PF mode 일 때: drone 현재 위치 hover position 으로 trajectory_setpoint 발행
        # → pf 단위테스트 와 동일 패턴 (offboard.py 가 takeoff pos 발행하던 효과)
        # → PX4 의 trajectory_setpoint stream 유지하면서 fusion=1.0 으로 pf attitude+thrust 가 실제 제어
        if self.fusion_weight_current >= self.PF_FUSION_THRESH:
            hmsg = TrajectorySetpoint()
            if self.have_local_pos:
                hmsg.position = [float(self.ned_x_current),
                                 float(self.ned_y_current),
                                 float(self.ned_z_current)]
            else:
                hmsg.position = [float('nan')]*3
            hmsg.velocity = [float('nan')]*3
            hmsg.acceleration = [float('nan')]*3
            hmsg.yaw = float('nan')
            hmsg.yawspeed = float('nan')
            hmsg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(hmsg)
            return

        # EGO-Planner의 명령 (ENU)
        pos_x_enu = cmd_msg.position.x
        pos_y_enu = cmd_msg.position.y
        pos_z_enu = cmd_msg.position.z

        vel_x_enu = cmd_msg.velocity.x
        vel_y_enu = cmd_msg.velocity.y
        vel_z_enu = cmd_msg.velocity.z

        acc_x_enu = cmd_msg.acceleration.x
        acc_y_enu = cmd_msg.acceleration.y
        acc_z_enu = cmd_msg.acceleration.z

        # 속도 제한
        vel_x_enu = np.clip(vel_x_enu, -self.max_velocity, self.max_velocity)
        vel_y_enu = np.clip(vel_y_enu, -self.max_velocity, self.max_velocity)
        vel_z_enu = np.clip(vel_z_enu, -self.max_velocity, self.max_velocity)

        # ENU -> PX4 NED 변환
        msg = TrajectorySetpoint()
        msg.position = [float(pos_y_enu), float(pos_x_enu), float(-pos_z_enu)]
        msg.velocity = [float(vel_y_enu), float(vel_x_enu), float(-vel_z_enu)]
        msg.acceleration = [float(acc_y_enu), float(acc_x_enu), float(-acc_z_enu)]

        # Yaw 변환: ENU -> NED
        # ENU: yaw는 East(+X)에서 반시계방향
        # NED: yaw는 North(+X)에서 시계방향
        # 변환: yaw_ned = pi/2 - yaw_enu
        yaw_enu = cmd_msg.yaw
        yaw_ned = math.pi / 2.0 - yaw_enu
        # -pi ~ pi 범위로 정규화
        yaw_ned = math.atan2(math.sin(yaw_ned), math.cos(yaw_ned))
        msg.yaw = float(yaw_ned)

        # Yaw rate 변환 (방향 반전)
        msg.yawspeed = float(-cmd_msg.yaw_dot)

        # PF→CA crossfade: 진입속도(감쇠) ↔ ego 속도 선형 blend → 안 멈추고 부드럽게 전환
        if blend_frac is not None:
            ev = self.ca_entry_vel_ned
            w_ego = blend_frac
            w_ent = 1.0 - blend_frac
            msg.velocity = [w_ent * ev[0] + w_ego * msg.velocity[0],
                            w_ent * ev[1] + w_ego * msg.velocity[1],
                            w_ent * ev[2] + w_ego * msg.velocity[2]]
            msg.position = [float('nan')]*3      # blend 중엔 velocity-primary (위치 snap 방지)
            msg.acceleration = [float('nan')]*3

        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.trajectory_setpoint_publisher.publish(msg)

    def check_planner_timeout(self):
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_planner_cmd_time).nanoseconds / 1e9

        if time_diff > self.planner_cmd_timeout:
            msg = TrajectorySetpoint()
            msg.velocity = [0.0, 0.0, 0.0]
            msg.position = [float('nan'), float('nan'), float('nan')]
            msg.acceleration = [float('nan'), float('nan'), float('nan')]
            msg.yaw = float('nan')
            msg.yawspeed = float('nan')
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Offboard mode")

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        # takeoff 단계는 position 명령 필요 → fusion 무시
        # PLANNER_READY 진입 후 PF mode 일 때만 velocity 로 전환
        in_ready = (self.current_state == self.STATE_PLANNER_READY)
        is_pf = in_ready and (self.fusion_weight_current >= self.PF_FUSION_THRESH)
        msg.position = not is_pf
        msg.velocity = is_pf
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float('nan')  # Yaw 비활성화
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        self.publish_offboard_control_heartbeat_signal()
        self.offboard_setpoint_counter += 1

        if self.current_state == self.STATE_PLANNER_READY:
            self.check_planner_timeout()

        if self.current_state == self.STATE_INIT and self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.current_state = self.STATE_ARMING

        elif self.current_state == self.STATE_ARMING and self.offboard_setpoint_counter == 20:
            self.arm()
            self.current_state = self.STATE_TAKEOFF

        elif self.current_state == self.STATE_TAKEOFF:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

            if self.odometry_data_valid and abs(self.current_position[2] - (-self.takeoff_height)) < self.tolerance:
                self.current_state = self.STATE_PLANNER_READY
                self.get_logger().info(f"Ready! Pos(ENU): {self.current_position}")

def main(args=None) -> None:
    print('Starting EGO-Planner + PX4 controller (Using PX4 Odometry, With Yaw Control)...')
    rclpy.init(args=args)
    drone_controller = EgoPlannerPX4Controller()
    rclpy.spin(drone_controller)
    drone_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
