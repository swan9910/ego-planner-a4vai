#!/usr/bin/env python3
"""
표준 TF 기반 PointCloud transformer (lidar 180° 마운트 회전 정확히 처리)
- /lidar/points → /lidar/points_world (world frame)
- TF: world → x500_0/base_link → lidar_sensor (static TF가 회전 정보 가지고 있어야 함)
"""

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformListener
import numpy as np
from scipy.spatial.transform import Rotation


class CorrectTransformer(Node):
    def __init__(self):
        super().__init__('correct_pc_transformer')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointCloud2, '/lidar/points', self._cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(
            PointCloud2, '/lidar/points_world', 10)

        self.get_logger().info('Correct TF-based PointCloud transformer started')
        self.cnt = 0

    def _cb(self, msg):
        self.cnt += 1
        try:
            t = self.tf_buffer.lookup_transform(
                'world', msg.header.frame_id, rclpy.time.Time(),
                Duration(seconds=0.1))
        except Exception as e:
            if self.cnt % 50 == 1:
                self.get_logger().warn(f'TF fail: {e}')
            return

        q = t.transform.rotation
        trans = np.array([t.transform.translation.x,
                          t.transform.translation.y,
                          t.transform.translation.z])
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])

        # parse xyz from raw
        offsets = {f.name: f.offset for f in msg.fields if f.name in ('x','y','z')}
        if len(offsets) < 3:
            return
        dt = np.dtype({'names': ['x','y','z'], 'formats': ['<f4','<f4','<f4'],
                       'offsets': [offsets['x'], offsets['y'], offsets['z']],
                       'itemsize': msg.point_step})
        s = np.frombuffer(msg.data, dtype=dt)
        pts = np.stack([s['x'], s['y'], s['z']], axis=1).astype(np.float32)

        # filter
        valid = np.isfinite(pts).all(axis=1)
        d = np.linalg.norm(pts, axis=1)
        valid &= (d > 0.75)
        pts = pts[valid]
        if len(pts) == 0:
            return

        # standard transform: world_pt = R * sensor_pt + t
        world_pts = (rot.apply(pts) + trans).astype(np.float32)

        # Build out msg
        out = PointCloud2()
        out.header = msg.header
        out.header.frame_id = 'world'
        out.height = 1
        out.width = len(world_pts)
        out.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        out.is_bigendian = False
        out.point_step = 12
        out.row_step = 12 * len(world_pts)
        out.data = world_pts.tobytes()
        out.is_dense = True
        self.pub.publish(out)

        if self.cnt % 100 == 1:
            self.get_logger().info(
                f'[{self.cnt}] pts={len(world_pts)} '
                f'rot_quat=({q.x:.2f},{q.y:.2f},{q.z:.2f},{q.w:.2f}) '
                f'trans=({trans[0]:.1f},{trans[1]:.1f},{trans[2]:.1f}) '
                f'z range: [{world_pts[:,2].min():.1f}, {world_pts[:,2].max():.1f}]')


def main():
    rclpy.init()
    n = CorrectTransformer()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
