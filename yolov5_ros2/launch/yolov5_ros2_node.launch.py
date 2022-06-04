from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    sub_arg = DeclareLaunchArgument(
        "sub_topic", default_value=TextSubstitution(text="/image")
    )
    pub_arg = DeclareLaunchArgument(
        "pub_topic", default_value=TextSubstitution(text="/yolov5_ros2/image")
    )
    weight_arg = DeclareLaunchArgument(
        "weights", default_value=TextSubstitution(text="yolov5l.pt")
    )
    device_arg = DeclareLaunchArgument(
        "device", default_value=TextSubstitution(text="")
    )
    
    yolov5_node_with_params = Node(
            package='yolov5_ros2',
            executable='obj_detect',
            parameters=[{
                "subscribed_topic": LaunchConfiguration('sub_topic'),
                "published_topic": LaunchConfiguration('pub_topic'),
                "weights": LaunchConfiguration('weights'),
                "device": LaunchConfiguration('device'),
                }]
        )

    return LaunchDescription([
        sub_arg,
        pub_arg,
        weight_arg,
        device_arg,
        yolov5_node_with_params,
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
        )
    ])

