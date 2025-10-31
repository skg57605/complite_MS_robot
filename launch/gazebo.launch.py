import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_name = 'complete_robot'
    
    # URDF 파일 경로 설정
    urdf_file_name = 'ms_robot.urdf.xacro'
    urdf_path = os.path.join(get_package_share_directory(pkg_name), 'urdf', urdf_file_name)
    robot_description_content = xacro.process_file(urdf_path).toxml()

    # (1) robot_state_publisher: URDF와 /joint_states를 기반으로 로봇의 TF를 발행
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )

    # (2) 직접 작성한 통합 제어 노드
    teleop_ik_node = Node(
        package=pkg_name,
        executable='teleop_ik_node',
        name='teleop_ik_node',
        output='screen'
    )

    # 실행할 노드 리스트
    return LaunchDescription([
        robot_state_publisher_node,
        teleop_ik_node,
    ])