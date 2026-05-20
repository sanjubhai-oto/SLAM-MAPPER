"""Bring up gz<->ros bridge, fisheye stitcher, uXRCE-DDS agent, stella_vslam.

Run PX4 + gz sim separately:
    cd ~/PX4-Autopilot
    make px4_sitl gz_rover_360cam
"""
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration('world')
    model = LaunchConfiguration('model')

    # Default world + model used by PX4 SITL target `gz_rover_360cam_slam_obstacles`.
    # Override at runtime: `ros2 launch ... world:=... model:=...` (re-launch needed —
    # the bridge args list is built at launch-description time).
    DEFAULT_WORLD = 'slam_obstacles'
    DEFAULT_MODEL = 'rover_360cam'
    imu_topic = (f'/world/{DEFAULT_WORLD}/model/{DEFAULT_MODEL}'
                 f'/link/base_link/sensor/imu_sensor/imu')

    bridge_topics = [
        # camera images (gz Image -> ROS sensor_msgs/Image)
        '/fisheye_front/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/fisheye_rear/image@sensor_msgs/msg/Image[gz.msgs.Image',
        # clock
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        # IMU
        f'{imu_topic}@sensor_msgs/msg/Imu[gz.msgs.IMU',
    ]

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='slam_obstacles'),
        DeclareLaunchArgument('model', default_value='rover_360cam'),

        # uXRCE-DDS agent for PX4 <-> ROS2
        ExecuteProcess(
            cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
            output='screen'),

        # gz<->ros bridge (topic remaps below)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            arguments=bridge_topics,
            output='screen'),

        # 360 stitcher
        Node(
            package='rover_bringup',
            executable='fisheye_to_equirect',
            name='fisheye_to_equirect',
            output='screen',
            parameters=[{
                'front_topic': '/fisheye_front/image',
                'rear_topic': '/fisheye_rear/image',
                'out_topic': '/rover_360cam/equirect',
                'eq_width': 1600,
                'eq_height': 800,
                'fisheye_size': 800,
                'fov_deg': 180.0,
            }]),

        # stella_vslam ROS2 node (built separately, see README)
        Node(
            package='stella_vslam_ros',
            executable='run_slam',
            name='stella_vslam',
            output='screen',
            arguments=[
                '-v', PathJoinSubstitution(
                    [FindPackageShare('rover_bringup'), 'config', 'orb_vocab.fbow']),
                '-c', PathJoinSubstitution(
                    [FindPackageShare('rover_bringup'), 'config', 'equirect_rover.yaml']),
                '--map-db-out', '/tmp/rover_map.msg',
                '--viewer', 'iridescence',
            ],
            remappings=[('camera/image_raw', '/rover_360cam/equirect')]),
    ])
