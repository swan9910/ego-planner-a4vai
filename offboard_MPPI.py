#!/usr/bin/env python3
"""
offboard_MPPI.py
- offboard.py 에서 ego-planner 의존성을 제거한 버전
- 드론 arming + 이륙 + 호버까지만 수행
- 호버 후에는 path following + MPPI 테스트가 실행될 수 있도록 위치 setpoint 유지
- ENU/NED 변환과 odometry 퍼블리시 기능은 유지 (다른 노드들이 사용 가능)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleOdometry,
    VehicleStatus,
    VehicleLocalPosition,
)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import math


class OffboardMPPIController(Node):
    """드론을 이륙시키고 지정 고도에서 호버 유지 (ego-planner 의존성 없음)"""

    def __init__(self) -> None:
        super().__init__('offboard_mppi_controller')

        # ── 파라미터 ────────────────────────────────────────────
        self.declare_parameter('takeoff_height', 5.0)   # ENU +Z (m)
        self.declare_parameter('hover_x', 0.0)
        self.declare_parameter('hover_y', 0.0)
        self.declare_parameter('yaw_offset_deg', 90.0)

        self.takeoff_height = float(self.get_parameter('takeoff_height').value)
        self.hover_x        = float(self.get_parameter('hover_x').value)
        self.hover_y        = float(self.get_parameter('hover_y').value)
        self.yaw_offset_rad = math.radians(self.get_parameter('yaw_offset_deg').value)
        self.tolerance      = 0.20

        # ── PX4 퍼블리셔 ────────────────────────────────────────
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/vehicle1/fmu/in/offboard_control_mode', qos_profile_sensor_data)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/vehicle1/fmu/in/trajectory_setpoint', qos_profile_sensor_data)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/vehicle1/fmu/in/vehicle_command', qos_profile_sensor_data)

        # ── PX4 서브스크라이버 ──────────────────────────────────
        self.create_subscription(
            VehicleOdometry, '/vehicle1/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback, qos_profile_sensor_data)
        self.create_subscription(
            VehicleStatus, '/vehicle1/fmu/out/vehicle_status',
            self.vehicle_status_callback, qos_profile_sensor_data)
        self.create_subscription(
            VehicleLocalPosition, '/vehicle1/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback, qos_profile_sensor_data)

        # ── ENU odometry 퍼블리셔 (다른 노드용) ──────────────────
        self.odom_pub = self.create_publisher(Odometry, '/ego_odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/ego_pose', 10)

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        lidar_tf = TransformStamped()
        lidar_tf.header.stamp = self.get_clock().now().to_msg()
        lidar_tf.header.frame_id = 'x500_0/base_link'
        lidar_tf.child_frame_id  = 'x500_0/lidar_3d_link/lidar_sensor'
        lidar_tf.transform.translation.z = -0.35
        lidar_tf.transform.rotation.w    = 1.0
        self.static_tf_broadcaster.sendTransform(lidar_tf)

        # ── 상태 머신 ──────────────────────────────────────────
        self.STATE_INIT     = 0
        self.STATE_ARMING   = 1
        self.STATE_TAKEOFF  = 2
        self.STATE_HOVER    = 3   # path following + MPPI 테스트용 hover state
        self.current_state  = self.STATE_INIT

        # ── 상태 변수 ──────────────────────────────────────────
        self.offboard_setpoint_counter = 0
        self.current_position = [0.0, 0.0, 0.0]
        self.odometry_data_valid = False
        self.vehicle_status = VehicleStatus()

        # ── 타이머 (20Hz) ──────────────────────────────────────
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info(
            f'offboard_MPPI initialized: takeoff_height={self.takeoff_height}m, '
            f'hover=({self.hover_x},{self.hover_y})')

    # ──────────────────────────────────────────────────────────
    # 콜백
    # ──────────────────────────────────────────────────────────
    def vehicle_odometry_callback(self, msg):
        # PX4 NED -> ENU
        pos_x_enu = float(msg.position[1])
        pos_y_enu = float(msg.position[0])
        pos_z_enu = float(-msg.position[2])

        vel_x_enu = float(msg.velocity[1])
        vel_y_enu = float(msg.velocity[0])
        vel_z_enu = float(-msg.velocity[2])

        self.current_position = [pos_x_enu, pos_y_enu, pos_z_enu]
        self.odometry_data_valid = True

        # quaternion NED -> ENU (yaw 보정 포함)
        q_ned = msg.q
        qw, qx, qy, qz = q_ned[0], q_ned[2], q_ned[1], -q_ned[3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1, min(1, sinp)))
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp) + self.yaw_offset_rad

        cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
        cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
        cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
        qw2 = cr*cp*cy + sr*sp*sy
        qx2 = sr*cp*cy - cr*sp*sy
        qy2 = cr*sp*cy + sr*cp*sy
        qz2 = cr*cp*sy - sr*sp*cy

        # ENU odometry 발행
        odom = Odometry()
        odom.header.stamp    = self.get_clock().now().to_msg()
        odom.header.frame_id = "world"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x = pos_x_enu
        odom.pose.pose.position.y = pos_y_enu
        odom.pose.pose.position.z = pos_z_enu
        odom.pose.pose.orientation.x = qx2
        odom.pose.pose.orientation.y = qy2
        odom.pose.pose.orientation.z = qz2
        odom.pose.pose.orientation.w = qw2
        odom.twist.twist.linear.x = vel_x_enu
        odom.twist.twist.linear.y = vel_y_enu
        odom.twist.twist.linear.z = vel_z_enu
        self.odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose   = odom.pose.pose
        self.pose_pub.publish(pose)

        # TF
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id  = 'x500_0/base_link'
        t.transform.translation.x = pos_x_enu
        t.transform.translation.y = pos_y_enu
        t.transform.translation.z = pos_z_enu
        t.transform.rotation.x = qx2
        t.transform.rotation.y = qy2
        t.transform.rotation.z = qz2
        t.transform.rotation.w = qw2
        self.tf_broadcaster.sendTransform(t)

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    def vehicle_local_position_callback(self, msg):
        # 필요시 reset counter 모니터링 가능
        pass

    # ──────────────────────────────────────────────────────────
    # PX4 명령 헬퍼
    # ──────────────────────────────────────────────────────────
    def publish_offboard_control_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint_ned(self, x_ned, y_ned, z_ned):
        msg = TrajectorySetpoint()
        msg.position     = [x_ned, y_ned, z_ned]
        msg.velocity     = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        msg.yaw          = float('nan')
        msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = params.get("param1", 0.0)
        msg.param2           = params.get("param2", 0.0)
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Offboard mode")

    # ──────────────────────────────────────────────────────────
    # 메인 루프
    # ──────────────────────────────────────────────────────────
    def timer_callback(self):
        # offboard heartbeat 매번 발행
        self.publish_offboard_control_heartbeat()
        self.offboard_setpoint_counter += 1

        # ENU hover_target → NED로 변환해서 setpoint 발행
        # ENU(x,y,z) -> NED(y,x,-z)
        target_ned = (self.hover_y, self.hover_x, -self.takeoff_height)

        if self.current_state == self.STATE_INIT and self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.current_state = self.STATE_ARMING

        elif self.current_state == self.STATE_ARMING and self.offboard_setpoint_counter == 20:
            self.arm()
            self.current_state = self.STATE_TAKEOFF

        elif self.current_state == self.STATE_TAKEOFF:
            self.publish_position_setpoint_ned(*target_ned)

            if self.odometry_data_valid:
                z_err = abs(self.current_position[2] - self.takeoff_height)
                xy_err = math.hypot(
                    self.current_position[0] - self.hover_x,
                    self.current_position[1] - self.hover_y,
                )
                if z_err < self.tolerance and xy_err < self.tolerance * 3:
                    self.current_state = self.STATE_HOVER
                    self.get_logger().info(
                        f'★ HOVER reached at ENU({self.current_position[0]:.2f}, '
                        f'{self.current_position[1]:.2f}, {self.current_position[2]:.2f})')
                    self.get_logger().info(
                        '  → Now safe to start: ros2 run pathfollowing node_att_ctrl')

        elif self.current_state == self.STATE_HOVER:
            # 호버 위치 setpoint 계속 발행 (path following이 자세를 가져가더라도 PX4가 hold)
            self.publish_position_setpoint_ned(*target_ned)


def main(args=None):
    print('Starting offboard_MPPI controller (no ego-planner) ...')
    rclpy.init(args=args)
    node = OffboardMPPIController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
