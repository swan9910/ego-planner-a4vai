#!/usr/bin/env python3
"""
RealGazebo EGO-Planner Unified Logger
- /planning/pos_cmd        : EGO-Planner 목표 (ENU)
- /vehicle1/fmu/out/vehicle_odometry : PX4 실제값 (NED → ENU 변환)
- /vehicle1/fmu/out/vehicle_local_position : reset counter 모니터링

CSV 저장: logs/ego_flight_YYYYMMDD_HHMMSS.csv
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from quadrotor_msgs.msg import PositionCommand
from px4_msgs.msg import VehicleOdometry, VehicleLocalPosition

import csv
import os
import math
import time
from datetime import datetime


class EgoFlightLogger(Node):
    def __init__(self):
        super().__init__('ego_flight_logger')

        # CSV 파일 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(log_dir, f'ego_flight_{timestamp}.csv')

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'time_sec',
            # EGO-Planner 목표 (ENU)
            'ego_pos_x', 'ego_pos_y', 'ego_pos_z',
            'ego_vel_x', 'ego_vel_y', 'ego_vel_z',
            'ego_acc_x', 'ego_acc_y', 'ego_acc_z',
            'ego_yaw', 'ego_yaw_dot', 'trajectory_id',
            # PX4 실제값 (ENU 변환)
            'actual_pos_x', 'actual_pos_y', 'actual_pos_z',
            'actual_vel_x', 'actual_vel_y', 'actual_vel_z',
            'actual_yaw',
            # 오차
            'error_pos_x', 'error_pos_y', 'error_pos_z',
            'error_pos_norm',
            # 이벤트
            'replan',
            'xy_reset_counter', 'z_reset_counter',
        ])

        self.start_time = time.time()

        # EGO-Planner 목표 (ENU)
        self.ego_pos = [0.0, 0.0, 0.0]
        self.ego_vel = [0.0, 0.0, 0.0]
        self.ego_acc = [0.0, 0.0, 0.0]
        self.ego_yaw = 0.0
        self.ego_yaw_dot = 0.0
        self.trajectory_id = 0
        self.prev_trajectory_id = 0
        self.replan_count = 0
        self.ego_received = False

        # PX4 실제값 (ENU 변환)
        self.actual_pos = [0.0, 0.0, 0.0]
        self.actual_vel = [0.0, 0.0, 0.0]
        self.actual_yaw = 0.0
        self.actual_received = False

        # Reset counters
        self.xy_reset = 0
        self.z_reset = 0
        self.prev_xy_reset = None
        self.prev_z_reset = None

        # Subscribers
        self.create_subscription(
            PositionCommand, '/planning/pos_cmd',
            self.ego_cmd_cb, 10)

        self.create_subscription(
            VehicleOdometry, '/vehicle1/fmu/out/vehicle_odometry',
            self.odom_cb, qos_profile_sensor_data)

        self.create_subscription(
            VehicleLocalPosition, '/vehicle1/fmu/out/vehicle_local_position',
            self.local_pos_cb, qos_profile_sensor_data)

        # 50Hz 로깅
        self.create_timer(0.02, self.log_cb)

        self.get_logger().info(f'EGO Flight Logger started → {self.csv_path}')

    def ego_cmd_cb(self, msg):
        self.ego_pos = [msg.position.x, msg.position.y, msg.position.z]
        self.ego_vel = [msg.velocity.x, msg.velocity.y, msg.velocity.z]
        self.ego_acc = [msg.acceleration.x, msg.acceleration.y, msg.acceleration.z]
        self.ego_yaw = msg.yaw
        self.ego_yaw_dot = msg.yaw_dot
        self.trajectory_id = msg.trajectory_id
        self.ego_received = True

    def odom_cb(self, msg):
        # PX4 NED → ENU
        self.actual_pos = [
            float(msg.position[1]),   # ENU x = NED y
            float(msg.position[0]),   # ENU y = NED x
            float(-msg.position[2]),  # ENU z = -NED z
        ]
        self.actual_vel = [
            float(msg.velocity[1]),
            float(msg.velocity[0]),
            float(-msg.velocity[2]),
        ]
        # Yaw from quaternion (NED→ENU)
        q = msg.q
        q_w, q_x, q_y, q_z = q[0], q[2], q[1], -q[3]
        siny = 2.0 * (q_w * q_z + q_x * q_y)
        cosy = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
        self.actual_yaw = math.atan2(siny, cosy)
        self.actual_received = True

    def local_pos_cb(self, msg):
        self.xy_reset = msg.xy_reset_counter
        self.z_reset = msg.z_reset_counter

        if self.prev_xy_reset is not None and self.xy_reset != self.prev_xy_reset:
            self.get_logger().warn(
                f'[XY RESET] {self.prev_xy_reset} → {self.xy_reset}, '
                f'delta: ({msg.delta_xy[0]:.3f}, {msg.delta_xy[1]:.3f})')
        if self.prev_z_reset is not None and self.z_reset != self.prev_z_reset:
            self.get_logger().warn(
                f'[Z RESET] {self.prev_z_reset} → {self.z_reset}, '
                f'delta: {msg.delta_z:.3f}')

        self.prev_xy_reset = self.xy_reset
        self.prev_z_reset = self.z_reset

    def log_cb(self):
        if not self.actual_received:
            return

        t = time.time() - self.start_time

        # Replan 감지
        replan = 0
        if self.trajectory_id != self.prev_trajectory_id:
            replan = 1
            self.replan_count += 1
            self.get_logger().info(
                f'Replan #{self.replan_count}: traj {self.prev_trajectory_id} → {self.trajectory_id}')
            self.prev_trajectory_id = self.trajectory_id

        # 위치 오차 (ENU)
        ex = self.ego_pos[0] - self.actual_pos[0]
        ey = self.ego_pos[1] - self.actual_pos[1]
        ez = self.ego_pos[2] - self.actual_pos[2]
        e_norm = math.sqrt(ex * ex + ey * ey + ez * ez)

        self.writer.writerow([
            f'{t:.4f}',
            # ego target
            f'{self.ego_pos[0]:.4f}', f'{self.ego_pos[1]:.4f}', f'{self.ego_pos[2]:.4f}',
            f'{self.ego_vel[0]:.4f}', f'{self.ego_vel[1]:.4f}', f'{self.ego_vel[2]:.4f}',
            f'{self.ego_acc[0]:.4f}', f'{self.ego_acc[1]:.4f}', f'{self.ego_acc[2]:.4f}',
            f'{self.ego_yaw:.4f}', f'{self.ego_yaw_dot:.4f}', self.trajectory_id,
            # actual
            f'{self.actual_pos[0]:.4f}', f'{self.actual_pos[1]:.4f}', f'{self.actual_pos[2]:.4f}',
            f'{self.actual_vel[0]:.4f}', f'{self.actual_vel[1]:.4f}', f'{self.actual_vel[2]:.4f}',
            f'{self.actual_yaw:.4f}',
            # error
            f'{ex:.4f}', f'{ey:.4f}', f'{ez:.4f}', f'{e_norm:.4f}',
            # events
            replan,
            self.xy_reset, self.z_reset,
        ])
        self.csv_file.flush()

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f'Log saved: {self.csv_path} ({self.replan_count} replans)')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EgoFlightLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
