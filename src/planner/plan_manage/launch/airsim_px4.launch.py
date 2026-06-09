
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    drone_id = LaunchConfiguration('drone_id')
    map_size_x = LaunchConfiguration('map_size_x')
    map_size_y = LaunchConfiguration('map_size_y')
    map_size_z = LaunchConfiguration('map_size_z')
    max_vel = LaunchConfiguration('max_vel')
    max_acc = LaunchConfiguration('max_acc')
    max_vel_z = LaunchConfiguration('max_vel_z')
    max_acc_z = LaunchConfiguration('max_acc_z')
    # EGO-Planner 노드
    ego_planner_node = Node(
        package='ego_planner',
        executable='ego_planner_node',
        name='ego_planner_node',
        output='screen',
        remappings=[
            ('odom_world', '/ego_odom'),
            ('grid_map/odom', '/ego_odom_grid'),
            ('grid_map/pose', '/ego_pose'),
            ('grid_map/cloud', '/lidar/points_world'),
            ('planning/bspline', '/planning/bspline'),
            ('planning/data_display', '/planning/data_display'),
            ('planning/broadcast_bspline_from_planner', '/broadcast_bspline'),
            ('planning/broadcast_bspline_to_planner', '/broadcast_bspline'),
        ],
        parameters=[
            # FSM 설정
            {'fsm/flight_type': 1},
            {'fsm/thresh_replan_time': 0.2},
            {'fsm/thresh_no_replan_meter': 0.3},
            {'fsm/planning_horizon': 25.0},
            {'fsm/planning_horizen_time': 6.0},
            {'fsm/emergency_time': 0.3},
            {'fsm/realworld_experiment': True},
            {'fsm/fail_safe': True},

            # Grid Map 설정 - ⭐ 큰 맵용 ⭐
            {'grid_map/resolution': 0.4},
            {'grid_map/map_size_x': map_size_x},
            {'grid_map/map_size_y': map_size_y},
            {'grid_map/map_size_z': map_size_z},
            {'grid_map/local_update_range_x': 25.0},
            {'grid_map/local_update_range_y': 25.0},
            {'grid_map/local_update_range_z': 20.0},
            {'grid_map/obstacles_inflation': 0.4},
            {'grid_map/local_map_margin': 20},  # 15 → 20
            {'grid_map/ground_height': -30.0},

            {'grid_map/virtual_ceil_height': 30.0},
            {'grid_map/visualization_truncate_height': 30.0},
            {'grid_map/show_occ_time': False},
            {'grid_map/pose_type': 2},
            {'grid_map/frame_id': 'world'},

            # Planner Manager (S2000용)
            {'manager/max_vel': max_vel},
            {'manager/max_acc': max_acc},
            {'manager/max_vel_z': max_vel_z},
            {'manager/max_acc_z': max_acc_z},
            {'manager/max_jerk': 2.5},
            {'manager/control_points_distance': 0.4},
            {'manager/feasibility_tolerance': 0.1},
            {'manager/planning_horizon': 25.0},
            {'manager/use_distinctive_trajs': False},
            {'manager/drone_id': 0},

            {'optimization/lambda_smooth': 9.943194837820485},
            {'optimization/lambda_collision': 3.404123317222346},
            {'optimization/lambda_feasibility': 0.47142979454800915},
            {'optimization/lambda_fitness': 0.43586141639667325},
            {'optimization/dist0': 3.0},
            {'optimization/swarm_clearance': 0.5},
            {'optimization/max_vel': max_vel},
            {'optimization/max_acc': max_acc},
            {'optimization/max_vel_z': max_vel_z},
            {'optimization/max_acc_z': max_acc_z},


            # B-Spline
            {'bspline/limit_vel': max_vel},
            {'bspline/limit_acc': max_acc},
            {'bspline/limit_ratio': 1.1},
        ]
    )

    # Trajectory Server 노드
    traj_server_node = Node(
        package='ego_planner',
        executable='traj_server',
        name='traj_server',
        output='screen',
        remappings=[
            ('position_cmd', '/planning/pos_cmd'),
            ('planning/bspline', '/planning/bspline'),
            ('grid_map/occupancy_inflate', '/grid_map/occupancy_inflate'),
        ],
        parameters=[
            {'traj_server/time_forward': 1.0},
            # 밀도 기반 속도 스케일링
            {'traj_server/d_safe': 2.0},           # 최근접 장애물 이 거리 이상 → 최대속도
            {'traj_server/d_min': 0.5},            # 이 거리 이하 → 최저속도
            {'traj_server/v_min_factor': 0.3},     # 최저 속도 비율 (30%)
            {'traj_server/lookahead_time': 1.0},   # 전방 예측 시간 (초)
            {'traj_server/density_radius': 3.0},   # 밀도 탐색 반경 (m)
            {'traj_server/density_max': 200},      # 이 개수 이상이면 최저속도
        ]
    )

    # LaunchDescription 생성
    ld = LaunchDescription()

    # Declare arguments
    ld.add_action(DeclareLaunchArgument('drone_id', default_value='0'))
    ld.add_action(DeclareLaunchArgument('map_size_x', default_value='400.0'))
    ld.add_action(DeclareLaunchArgument('map_size_y', default_value='400.0'))
    ld.add_action(DeclareLaunchArgument('map_size_z', default_value='30.0'))
    ld.add_action(DeclareLaunchArgument('max_vel', default_value='3.0'))
    ld.add_action(DeclareLaunchArgument('max_acc', default_value='1.5'))
    ld.add_action(DeclareLaunchArgument('max_vel_z', default_value='1.5'))
    ld.add_action(DeclareLaunchArgument('max_acc_z', default_value='1.5'))

    # Add nodes
    ld.add_action(ego_planner_node)
    ld.add_action(traj_server_node)

    return ld
