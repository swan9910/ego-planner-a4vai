#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

# x, y, z: local relative to drone start, 좌표 그대로 PX4에 전달
# ======================================================
# 웨이포인트 설정 (x, y, z) — local relative to start
# 여기만 수정하면 됩니다
# ======================================================
WAYPOINTS = [
    (  0.0000,   0.0000,   0.0000),
    (  0.0000,   0.0000,  -2.8746),
    ( 13.5349,  -8.8186,  -2.8746),
    ( 27.0165, -17.6920,  -2.8746),
    ( 40.5535, -26.5084,  -2.5552),
    ( 54.1097, -35.3050,  -1.9164),
    ( 67.0369, -44.7501,  -0.6388),
    ( 83.8271, -50.2121,   8.3044),
    (103.2726, -52.9361,  24.2744),
    (109.7669, -69.0143,  11.1790),
    (113.5200, -87.9190,   4.7910),
    (113.5200, -87.9190,   7.9850),
]
# ======================================================

TOLERANCE = 3.0  # m


class WaypointOffboard(Node):

    def __init__(self):
        super().__init__('waypoint_offboard')

        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/vehicle1/fmu/in/offboard_control_mode', qos_profile_sensor_data)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/vehicle1/fmu/in/trajectory_setpoint', qos_profile_sensor_data)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/vehicle1/fmu/in/vehicle_command', qos_profile_sensor_data)

        self.create_subscription(
            VehicleLocalPosition, '/vehicle1/fmu/out/vehicle_local_position',
            self.local_position_cb, qos_profile_sensor_data)
        self.create_subscription(
            VehicleStatus, '/vehicle1/fmu/out/vehicle_status_v1',
            self.vehicle_status_cb, qos_profile_sensor_data)

        self.pos = [0.0, 0.0, 0.0]
        self.pos_valid = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

        self.counter = 0
        self.wp_index = 0

        self.STATE_INIT = 0
        self.STATE_ARM  = 1
        self.STATE_FLY  = 2
        self.STATE_DONE = 3
        self.state = self.STATE_INIT

        self.create_timer(0.05, self.timer_cb)  # 20Hz
        self.get_logger().info('Waypoint offboard node started')

    def local_position_cb(self, msg):
        self.pos = [msg.x, msg.y, msg.z]
        self.pos_valid = True

    def vehicle_status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def dist_to_wp(self, x, y, z):
        return math.sqrt((self.pos[0]-x)**2 + (self.pos[1]-y)**2 + (self.pos[2]-z)**2)

    def publish_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def z_offset(self, idx):
        return 5.0 if idx < 4 else 20.0

    def publish_setpoint(self, x, y, z, offset):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(-z - offset)]
        msg.velocity = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        msg.yaw = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def send_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def timer_cb(self):
        self.publish_heartbeat()
        self.counter += 1

        if self.state == self.STATE_INIT and self.counter == 10:
            self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.get_logger().info('Offboard mode requested')
            self.state = self.STATE_ARM

        elif self.state == self.STATE_ARM and self.counter == 20:
            self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            self.get_logger().info(f'Arming... Flying to WP[0]: {WAYPOINTS[0]}')
            self.state = self.STATE_FLY

        elif self.state == self.STATE_FLY:
            x, y, z = WAYPOINTS[self.wp_index]
            offset = self.z_offset(self.wp_index)
            self.publish_setpoint(x, y, z, offset)
            dist = self.dist_to_wp(x, y, -z - offset)
            self.get_logger().info(
                f'WP[{self.wp_index}] | pos={[f"{v:.2f}" for v in self.pos]} target=({x:.2f},{y:.2f},{z-offset:.2f}) dist={dist:.2f}',
                throttle_duration_sec=2.0)

            is_last = (self.wp_index == len(WAYPOINTS) - 1)
            tol = 1.0 if is_last else TOLERANCE
            if self.pos_valid and dist < tol:
                self.get_logger().info(f'WP[{self.wp_index}] reached')
                self.wp_index += 1

                if self.wp_index >= len(WAYPOINTS):
                    self.get_logger().info('All waypoints complete. Hovering.')
                    self.state = self.STATE_DONE
                else:
                    self.get_logger().info(f'Flying to WP[{self.wp_index}]: {WAYPOINTS[self.wp_index]}')

        elif self.state == self.STATE_DONE:
            x, y, z = WAYPOINTS[-1]
            self.publish_setpoint(x, y, z, self.z_offset(len(WAYPOINTS) - 1))


def main(args=None):
    rclpy.init(args=args)
    node = WaypointOffboard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
