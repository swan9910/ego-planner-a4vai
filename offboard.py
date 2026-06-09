#!/usr/bin/env python3
import rclpy
import rclpy.parameter
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry, VehicleStatus, VehicleLocalPosition
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from quadrotor_msgs.msg import PositionCommand
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
        """Reset counter 모니터링"""
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

    def ego_planner_callback(self, cmd_msg):
        if self.current_state != self.STATE_PLANNER_READY:
            return

        self.last_planner_cmd_time = self.get_clock().now()

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
        msg.position = True
        msg.velocity = False
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
