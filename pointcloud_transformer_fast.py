#!/usr/bin/env python3
import rclpy
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
import numpy as np

np.seterr(invalid='ignore')

class FastPointCloudTransformer(Node):
    def __init__(self):
        super().__init__('fast_pointcloud_transformer')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pc_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.pointcloud_callback, 10)
        self.pc_pub = self.create_publisher(
            PointCloud2, '/lidar/points_world', 10)

        self.get_logger().info('Fast PointCloud Transformer started (pitch+roll correction)')

    def pointcloud_callback(self, msg):
        try:
            # Use latest available TF (Time(0)) to avoid "extrapolation into future" errors
            transform = self.tf_buffer.lookup_transform(
                'world', msg.header.frame_id, rclpy.time.Time(),
                Duration(seconds=0.1))

            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z
            q = transform.transform.rotation

            # Gazebo gpu_lidar: points are in sensor local frame.
            # Correction order: Ry(-rpy[0]) pitch, Rx(+rpy[1]) roll, Rz(yaw)
            qx, qy, qz, qw = q.x, q.y, q.z, q.w

            # Extract rpy[0] (euler roll = drone pitch)
            sinr = 2.0 * (qw * qx + qy * qz)
            cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
            rpy0 = np.arctan2(sinr, cosr)

            # Extract rpy[1] (euler pitch = drone roll)
            sinp = 2.0 * (qw * qy - qz * qx)
            sinp = np.clip(sinp, -1.0, 1.0)
            rpy1 = np.arcsin(sinp)

            # Extract yaw
            siny = 2.0 * (qw * qz + qx * qy)
            cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny, cosy)

            # Ry(-rpy0): correct pitch
            cp = np.cos(-rpy0)
            sp = np.sin(-rpy0)
            # Rx(+rpy1): correct roll
            cr = np.cos(rpy1)
            sr = np.sin(rpy1)
            # Rz(yaw): apply yaw to world frame
            cy = np.cos(yaw)
            sy = np.sin(yaw)

            # Parse xyz directly from raw bytes using numpy structured array
            offsets = {}
            for field in msg.fields:
                if field.name in ('x', 'y', 'z'):
                    offsets[field.name] = field.offset

            if len(offsets) < 3:
                return

            step = msg.point_step
            n_pts = msg.width * msg.height
            if n_pts == 0:
                return

            dt = np.dtype({'names': ['x','y','z'], 'formats': ['<f4','<f4','<f4'],
                           'offsets': [offsets['x'], offsets['y'], offsets['z']],
                           'itemsize': step})
            structured = np.frombuffer(msg.data, dtype=dt)
            x = structured['x'].copy()
            y = structured['y'].copy()
            z = structured['z'].copy()

            # Filter NaN and nearby points (drone body, < 0.5m)
            dist = np.sqrt(x * x + y * y + z * z)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (dist > 0.75)
            x, y, z = x[valid], y[valid], z[valid]

            if len(x) == 0:
                return

            # Step 1: Ry(-rpy0) for pitch correction
            x1 = cp * x + sp * z
            z1 = -sp * x + cp * z

            # Step 2: Rx(+rpy1) for roll correction
            x2 = x1
            y2 = cr * y - sr * z1
            z2 = sr * y + cr * z1

            # Step 3: Rz(yaw) to rotate into world frame
            wx = cy * x2 - sy * y2 + tx
            wy = sy * x2 + cy * y2 + ty
            wz = z2 + tz

            # # Add virtual ceiling/floor planes (disabled)
            # grid = np.arange(-10.0, 10.1, 1.5, dtype=np.float32)
            # gx, gy = np.meshgrid(grid, grid)
            # gx = gx.ravel() + tx
            # gy = gy.ravel() + ty
            # n_plane = len(gx)
            # ceil_z = np.full(n_plane, tz + 2.0, dtype=np.float32)
            # floor_z = np.full(n_plane, tz - 2.0, dtype=np.float32)
            # wx = np.concatenate([wx, gx, gx])
            # wy = np.concatenate([wy, gy, gy])
            # wz = np.concatenate([wz, ceil_z, floor_z])

            # Build output PointCloud2 directly
            n_out = len(wx)
            out_points = np.empty((n_out, 3), dtype=np.float32)
            out_points[:, 0] = wx
            out_points[:, 1] = wy
            out_points[:, 2] = wz

            out_msg = PointCloud2()
            out_msg.header = msg.header
            out_msg.header.frame_id = 'world'
            out_msg.height = 1
            out_msg.width = n_out
            out_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            out_msg.is_bigendian = False
            out_msg.point_step = 12
            out_msg.row_step = 12 * n_out
            out_msg.data = out_points.tobytes()
            out_msg.is_dense = True

            self.pc_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().warn(f'Transform failed: {e}', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = FastPointCloudTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
