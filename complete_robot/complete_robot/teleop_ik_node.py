import rclpy
import math
import tf2_ros

from rclpy.node import Node
from sensor_msgs.msg import JointState
from dynamixel_sdk import *
from math import pi
from tf2_ros import TransformException

# 통신 설정
PROTOCOL_VERSION = 2.0
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'

# 다이나믹셀 ID
DXL_IDS = {'m_joint_1': 10, 'm_joint_2': 11, 'm_joint_3': 12, 'm_joint_4': 13,
           's_joint_z': 0, 's_joint_1': 1, 's_joint_2': 2, 's_end_joint_z': 3, 's_end_joint_y': 4}

# 레지스터 주소
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_GOAL_POSITION = 116
#ADDR_GOAL_VELOCITY = 104
ADDR_GOAL_VELOCITY = 112

# 단위 변환 상수
VELOCITY_UNIT_TO_RAD_PER_SEC = 0.229 * (2 * pi) / 60


class FullRobotControllerNode(Node):
    """
    마스터 로봇의 상태를 읽어 TF를 생성하고, 이를 추종하여 슬레이브 로봇을 직접 제어하는 통합 노드.
    """
    def __init__(self):
        super().__init__('full_robot_controller_node')
        self.get_logger().info('Full Robot Controller Node has been started.')

        # --- 상태 발행기 초기화 ---
        self.joint_state_publisher = self.create_publisher(JointState, '/joint_states', 10)

        # --- 다이나믹셀 핸들러 초기화 ---
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        if not self.portHandler.openPort():
            self.get_logger().error(f"Failed to open port {DEVICENAME}")
            rclpy.shutdown()
        if not self.portHandler.setBaudRate(BAUDRATE):
            self.get_logger().error(f"Failed to set baudrate to {BAUDRATE}")
            rclpy.shutdown()
        
        # 토크 설정: 마스터는 끄고, 슬레이브는 켬
        for joint_name, dxl_id in DXL_IDS.items():
            if joint_name.startswith('m_'):
                self.disable_torque(dxl_id)
                self.get_logger().info(f"Torque OFF for master joint: {joint_name} (ID: {dxl_id})")
            elif joint_name.startswith('s_'):
                self.enable_torque(dxl_id)
                self.get_logger().info(f"Torque ON for slave joint: {joint_name} (ID: {dxl_id})")

        # --- TF 리스너 초기화 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- 역기구학 변수 초기화 ---
        self.l1, self.l2 = 0.2, 0.24  # 슬레이브 로봇의 링크 길이 (미터 단위)
        self.base_link_offset_x = 0.1
        self.base_link_offset_z = 0.08
        self.z_axis_pitch_m = 0.040  #리드 스크류 리드

        # 현재 슬레이브 관절 각도를 저장할 변수 추가
        self.current_s_joint_1 = 0.0
        self.current_s_joint_2 = 0.0
        self.current_s_z_pos_m = 0.0

        self.URDF_Z_HOME_POS_M = 0.1
        self.z_axis_offset = None

        self.joint_limits = {
            'm_joint_1': {'min': -1.7453, 'max': 1.7453},
            'm_joint_2': {'min': -1.7453, 'max': 1.7453},
            'm_joint_3': {'min': -1.7453, 'max': 1.7453},
            'm_joint_4': {'min': -1.7453, 'max': 1.7453},

            's_joint_1': {'min': -2.3562, 'max': 2.3562},
            's_joint_2': {'min': -2.3562, 'max': 2.3562},
            's_joint_z': {'min': 0.05,    'max': 0.45},
            's_end_joint_z': {'min': -2.3562, 'max': 2.3562},
            's_end_joint_y': {'min': -2.3562, 'max': 2.3562}
        }

        # --- 타이머 초기화 ---
        self.state_publishing_timer = self.create_timer(0.02, self.state_publishing_callback)
        self.ik_control_timer = self.create_timer(0.02, self.ik_control_callback)

    # --- 다이나믹셀 유틸리티 메서드 ---
    def enable_torque(self, dxl_id):
        self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)

    def disable_torque(self, dxl_id):
        self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)

    def convert_value_to_radian(self, value):
        return (value - 2048) * (2 * pi) / 4096

    def convert_radian_to_value(self, radian):
        return int(radian * 4096 / (2 * pi) + 2048)

    def convert_value_to_radian_per_second(self, value):
        if value > 2147483647: value -= 4294967296
        return value * VELOCITY_UNIT_TO_RAD_PER_SEC

    def write_goal_position(self, dxl_id, radian):
        dxl_goal_position = self.convert_radian_to_value(radian)
        self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_GOAL_POSITION, dxl_goal_position)

    def _distance_to_radian(self, distance_m):
        revolution = distance_m / self.z_axis_pitch_m
        return revolution * 2 * math.pi

    def _radian_to_distance(self, radian):
        revolution = radian / (2 * math.pi)
        return revolution * self.z_axis_pitch_m

    def _multiturn_value_to_radian(self, value):
        if value > 2147483647: value -= 4294967296
        return value * (2 * pi) / 4096

    def _radian_to_multiturn_value(self, radian):
        return int(radian * 4096 / (2 * pi))

    #def write_goal_velocity(self, dxl_id, rad_per_sec):
    #    """라디안/초 단위의 속도를 다이나믹셀의 목표 속도로 전송합니다."""
    #    dxl_goal_velocity = self.convert_radian_per_second_to_value(rad_per_sec)
    #    self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_GOAL_VELOCITY, dxl_goal_velocity)

    def write_goal_position(self, dxl_id, radian):
        # ✨ 1. 목표 속도를 설정합니다 (예: 115 = 약 13.17 RPM). 이 값을 조절하여 부드러움 조절.
        # 0으로 설정하면 모터의 최대 속도로 움직입니다 (현재의 "급격한" 상태).
        self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_GOAL_VELOCITY, 115) 

        # 2. 목표 위치를 설정합니다
        dxl_goal_position = self.convert_radian_to_value(radian)
        self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_GOAL_POSITION, dxl_goal_position)

    def convert_radian_per_second_to_value(self, rad_per_sec):
        """라디안/초 단위를 다이나믹셀 속도 단위로 변환합니다."""
        return int(rad_per_sec / VELOCITY_UNIT_TO_RAD_PER_SEC)


    # --- 주기적으로 실행되는 콜백 함수들 ---
    def state_publishing_callback(self):
        """모든 다이나믹셀의 상태를 읽어 /joint_states 토픽으로 발행합니다."""
        try:
            state_msg = JointState()
            state_msg.header.stamp = self.get_clock().now().to_msg()
            
            positions = []
            velocities = []
            
            joint_names_ordered = ['m_joint_1', 'm_joint_2', 'm_joint_3', 'm_joint_4','s_joint_z', 's_joint_1', 's_joint_2', 's_end_joint_z', 's_end_joint_y']
            state_msg.name = joint_names_ordered

            for joint_name in joint_names_ordered:
                dxl_id = DXL_IDS[joint_name]
                pos_val, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, ADDR_PRESENT_POSITION)

                # 슬레이브의 현재 관절 각도 값을 클래스 변수에 저장
                if joint_name == 's_joint_z':
                    if self.z_axis_offset is None:
                        # 실제 모터 값과 이론적인 홈 위치 값의 차이를 계산하여 Offset으로 저장
                        actual_pos_val = pos_val
                        home_pos_rad = self._distance_to_radian(self.URDF_Z_HOME_POS_M)
                        target_home_pos_val = self._radian_to_multiturn_value(home_pos_rad)
                        self.z_axis_offset = actual_pos_val - target_home_pos_val
                        self.get_logger().info(f'Z-axis calibrated by URDF home position! Offset: {self.z_axis_offset}')
                    
                    # 캡처된 원점 기준으로 현재 위치 보정
                    pos_val -= self.z_axis_offset
                    
                    # 보정된 값을 기반으로 현재 높이(m) 계산
                    pos_rad = self._multiturn_value_to_radian(pos_val)
                    pos_m = self._radian_to_distance(pos_rad)
                    positions.append(pos_m)
                    self.current_s_z_pos_m = pos_m
                else:
                    pos_rad = self.convert_value_to_radian(pos_val)
                    positions.append(pos_rad)
                    if joint_name == 's_joint_1':
                        self.current_s_joint_1 = pos_rad
                    elif joint_name == 's_joint_2':
                        self.current_s_joint_2 = pos_rad
                    elif joint_name == 'm_joint_3':
                        self.current_m_joint_3 = pos_rad
                    elif joint_name == 'm_joint_4':
                        self.current_m_joint_4 = pos_rad

                # 속도값 읽기 (필요한 조인트만)
                if joint_name in ['s_joint_1', 's_joint_2']:
                    vel, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, ADDR_PRESENT_VELOCITY)
                    velocities.append(self.convert_value_to_radian_per_second(vel))
                else:
                    velocities.append(0.0)

            state_msg.position = positions
            state_msg.velocity = velocities
            self.joint_state_publisher.publish(state_msg)

        except Exception as e:
            self.get_logger().warn(f"Failed to read dynamixel state: {e}", throttle_duration_sec=1.0)

    def ik_control_callback(self):
        """마스터 로봇의 end_effector를 추적하여 슬레이브 로봇을 제어합니다."""
        target_frame = 's_base_link'
        source_frame = 'm_link_3'

        if self.z_axis_offset is None:
            self.get_logger().warn("Z-axis is not calibrated yet. Waiting...", throttle_duration_sec=1.0)
            return

        if not self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)):
            self.get_logger().warn(
                f"Transform from {source_frame} to {target_frame} is not yet available.",
                throttle_duration_sec=1.0)
            return

        try:
            t = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {source_frame} to {target_frame}: {ex}',
                throttle_duration_sec=1.0)
            return

        target_x = t.transform.translation.x
        target_y = t.transform.translation.y
        target_z = t.transform.translation.z

        try:
            # s_joint_1, s_joint_2 제어
            ik_target_x = target_x - self.base_link_offset_x

            s1_limits = self.joint_limits['s_joint_1']
            s2_limits = self.joint_limits['s_joint_2']

            theta1, theta2 = self.solve_ik(ik_target_x, target_y, self.current_s_joint_1, self.current_s_joint_2)

            safe_theta1 = max(s1_limits['min'], min(theta1, s1_limits['max']))
            safe_theta2 = max(s2_limits['min'], min(theta2, s2_limits['max']))
            
            self.write_goal_position(DXL_IDS['s_joint_1'], safe_theta1)
            self.write_goal_position(DXL_IDS['s_joint_2'], safe_theta2)


        except ValueError as e:
            self.get_logger().warn(f'IK Error: {e}', throttle_duration_sec=1.0)
            return

        try:
            # 1. 제어 게인(Kp) 및 최대 속도 설정
            kp, max_velocity_rad_s = 3.0, 6.0
            # URDF 홈 위치를 0으로 간주하고, 그에 맞는 상대적인 이동 범위 설정
            z_limits = self.joint_limits['s_joint_z']
            
            # 목표치(target_z)도 홈 위치 기준으로 변환하여 오차 계산에 사용
            relative_target_z = target_z - self.URDF_Z_HOME_POS_M
            relative_z_min = z_limits['min'] - self.URDF_Z_HOME_POS_M
            relative_z_max = z_limits['max'] - self.URDF_Z_HOME_POS_M
            safe_target_z = max(relative_z_min, min(relative_target_z, relative_z_max))
            target_z_rad = self._distance_to_radian(safe_target_z)
            current_z_rad = self._distance_to_radian(self.current_s_z_pos_m)

            error_rad = target_z_rad - current_z_rad
            goal_velocity_rad_s = kp * error_rad
            
            # 속도 제한 적용
            goal_velocity_rad_s = max(-max_velocity_rad_s, min(goal_velocity_rad_s, max_velocity_rad_s))

            # 계산된 목표 속도를 다이나믹셀에 명령 (한 번만 실행)
            self.write_goal_velocity(DXL_IDS['s_joint_z'], goal_velocity_rad_s)

        except Exception as e:
            self.get_logger().warn(f'Z-axis control error: {e}', throttle_duration_sec=1.0)

        try:
            # m_joint_3 -> s_end_joint_y (y축 회전)
            s_y_limits = self.joint_limits['s_end_joint_y']
            # 마스터의 각도를 슬레이브의 안전 범위 내로 제한
            safe_y_angle = max(s_y_limits['min'], min(self.current_m_joint_3, s_y_limits['max']))
            self.write_goal_position(DXL_IDS['s_end_joint_y'], safe_y_angle)

            # m_joint_4 -> s_end_joint_z (z축 회전)
            s_z_limits = self.joint_limits['s_end_joint_z']
            # 마스터의 각도를 슬레이브의 안전 범위 내로 제한
            safe_z_angle = max(s_z_limits['min'], min(self.current_m_joint_4, s_z_limits['max']))
            self.write_goal_position(DXL_IDS['s_end_joint_z'], safe_z_angle)

        except Exception as e:
            self.get_logger().warn(f'End-effector orientation control error: {e}', throttle_duration_sec=1.0)

    def solve_ik(self, x, y, current_q1, current_q2):
        l1, l2 = self.l1, self.l2
        d_sq = x**2 + y**2

        if not ((l1 - l2)**2 <= d_sq <= (l1 + l2)**2):
            raise ValueError(f"Target (x:{x:.2f}, y:{y:.2f}) is outside workspace.")
            
        cos_theta2 = min(1.0, max(-1.0, (d_sq - l1**2 - l2**2) / (2 * l1 * l2)))
        sin_theta2_abs = math.sqrt(1 - cos_theta2**2)

        theta2_down = math.atan2(-sin_theta2_abs, cos_theta2)
        k1_down = l1 + l2 * math.cos(theta2_down)
        k2_down = l2 * math.sin(theta2_down)
        theta1_down = math.atan2(y, x) - math.atan2(k2_down, k1_down)

        theta2_up = math.atan2(sin_theta2_abs, cos_theta2)
        k1_up = l1 + l2 * math.cos(theta2_up)
        k2_up = l2 * math.sin(theta2_up)
        theta1_up = math.atan2(y, x) - math.atan2(k2_up, k1_up)

        dist_sq_down = (theta1_down - current_q1)**2 + (theta2_down - current_q2)**2
        dist_sq_up = (theta1_up - current_q1)**2 + (theta2_up - current_q2)**2

        if dist_sq_down < dist_sq_up:
            return theta1_down, theta2_down
        else:
            return theta1_up, theta2_up

    # --- 노드 종료 시 자원 해제 ---
    def destroy_node(self):
        self.get_logger().info('Shutting down. Disabling torque and closing port.')
        self.write_goal_velocity(DXL_IDS['s_joint_z'], 0)
        for dxl_id in DXL_IDS.values():
            self.disable_torque(dxl_id)
        self.portHandler.closePort()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FullRobotControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()