#!/usr/bin/env python3
"""
RealGazebo Lidar to World Frame Converter
- /ego_odom (offboard.py가 퍼블리시하는 ENU odom) 구독
- /lidar/points (Gz→ROS2 bridge PointCloud2) 구독
- /camera/depth/points (EGO-Planner용 월드 좌표계 PointCloud2) 퍼블리시
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation


class LidarWorldTransformer(Node):
    def __init__(self):
        super().__init__('lidar_world_transformer')

        self.odom_received = False

        # 최신 odom (ENU)
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        self.drone_quat = np.array([0.0, 0.0, 0.0, 1.0])

        # 필터링 파라미터
        self.min_distance = 1.0
        self.z_threshold = 0.1

        self.get_logger().info('=== Lidar World Transformer (RealGazebo) ===')

        # offboard.py가 퍼블리시하는 ENU odom 구독
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_odom',
            self.odom_callback,
            50
        )

        # RealGazebo LiDAR 구독 (Gz→ROS2 bridge)
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.lidar_callback,
            10
        )

        # EGO-Planner용 월드 좌표계 포인트클라우드 퍼블리셔
        self.world_pub = self.create_publisher(
            PointCloud2,
            '/camera/depth/points',
            10
        )

        self.count = 0

    def odom_callback(self, msg):
        """최신 Odom 저장 (ENU)"""
        self.drone_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        self.drone_quat = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info('Odom received')

    def lidar_callback(self, msg):
        self.count += 1

        if not self.odom_received:
            if self.count % 100 == 0:
                self.get_logger().warn('Waiting for odom...')
            return

        # 포인트클라우드 파싱
        points = self.parse_pointcloud2(msg)
        if points is None or len(points) == 0:
            return

        # 필터링
        xy_distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        height_diff = np.abs(points[:, 2])
        mask = (xy_distances >= self.min_distance) | ((xy_distances >= 0.5) & (height_diff > self.z_threshold))
        filtered_points = points[mask]

        if len(filtered_points) == 0:
            return

        # Body frame → World frame 변환
        current_rot = Rotation.from_quat(self.drone_quat)
        world_points = current_rot.apply(filtered_points) + self.drone_pos

        # 디버그 출력
        if self.count % 100 == 0:
            euler = current_rot.as_euler('XYZ', degrees=True)
            z_mean = world_points[:, 2].mean()
            z_std = world_points[:, 2].std()
            self.get_logger().info(
                f'[{self.count}] Pos: ({self.drone_pos[0]:.1f}, {self.drone_pos[1]:.1f}, {self.drone_pos[2]:.1f}) | '
                f'RPY: ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg | '
                f'Pts: {len(world_points)} | Z: {z_mean:.1f}±{z_std:.1f}'
            )

        # 발행
        world_msg = self.create_pointcloud2(world_points)
        world_msg.header.stamp = msg.header.stamp
        world_msg.header.frame_id = "world"
        self.world_pub.publish(world_msg)

    def parse_pointcloud2(self, cloud_msg):
        """point_step 기반으로 x,y,z 추출 (RealGazebo bridge: point_step=32)"""
        point_step = cloud_msg.point_step
        data = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        n_points = cloud_msg.width * cloud_msg.height

        if len(data) < n_points * point_step:
            return None

        data = data[:n_points * point_step].reshape(n_points, point_step)
        x = np.frombuffer(data[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(data[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(data[:, 8:12].tobytes(), dtype=np.float32)

        points = np.column_stack((x, y, z))
        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
        return points[valid_mask] if np.any(valid_mask) else None

    def create_pointcloud2(self, points):
        msg = PointCloud2()
        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.fields = [
            self.create_field('x', 0, PointField.FLOAT32, 1),
            self.create_field('y', 4, PointField.FLOAT32, 1),
            self.create_field('z', 8, PointField.FLOAT32, 1)
        ]
        msg.data = points.astype(np.float32).tobytes()
        return msg

    def create_field(self, name, offset, datatype, count):
        field = PointField()
        field.name = name
        field.offset = offset
        field.datatype = datatype
        field.count = count
        return field


def main(args=None):
    rclpy.init(args=args)
    node = LidarWorldTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
