import rclpy
import tf2_ros
import cv2
import numpy as np
import pandas as pd
import os
import time

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import SetBool # '녹화 시작/정지' 제어를 위한 표준 서비스
from tf2_ros import TransformException

# --- 우리가 만들 CSV 파일 설정 ---
CSV_PATH = 'master_dataset.csv'
CSV_COLUMNS = [
    'demonstration_id', 'timestamp',
    'goal_obj_0', 'goal_obj_1', 'goal_obj_2', # (x, y, 2D_angle) - 예시
    'current_arm_0', 'current_arm_1', 'current_arm_2', 'current_arm_3', 'current_arm_4', 'current_arm_5', 'current_arm_6', # (x,y,z,qx,qy,qz,qw)
    'target_arm_0', 'target_arm_1', 'target_arm_2', 'target_arm_3', 'target_arm_4', 'target_arm_5', 'target_arm_6'  # (x,y,z,qx,qy,qz,qw)
]
# 참고: goal_obj는 7D가 되어야 하지만, 여기서는 OpenCV 2D 결과를 임시로 3D(x,y,angle)로 저장합니다.
# 이 부분을 7D로 변환하는 것은 '보정(Calibration)' 작업이 필요합니다.

class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        self.get_logger().info("데이터 수집 노드 시작...")

        # --- 1. ROS 2 통신 초기화 ---
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 카메라 이미지 구독
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw', # (실제 카메라 토픽명으로 변경 필요)
            self.image_callback,
            10)
        self.latest_image = None # 최신 이미지를 저장할 변수

        # 녹화 제어 서비스
        self.record_service = self.create_service(
            SetBool,
            'toggle_recording',
            self.toggle_recording_callback)
        
        # 메인 데이터 로깅 루프 (예: 30Hz)
        self.logging_timer = self.create_timer(1.0 / 30.0, self.logging_callback)

        # --- 2. 데이터 로깅 상태 변수 ---
        self.is_recording = False
        self.demonstration_id = 0
        self.goal_obj_pose = None # [obj_cx, obj_cy, obj_angle]
        
        # CSV 파일 헤더 확인
        self.check_csv_header()
        
        self.get_logger().info("준비 완료. '/toggle_recording' 서비스를 호출하여 녹화를 시작하세요.")

    def check_csv_header(self):
        """CSV 파일이 없으면 헤더를 포함하여 새로 생성합니다."""
        if not os.path.exists(CSV_PATH):
            df = pd.DataFrame(columns=CSV_COLUMNS)
            df.to_csv(CSV_PATH, index=False)
            self.get_logger().info(f"'{CSV_PATH}' 파일 생성됨.")

    def image_callback(self, msg):
        """카메라 이미지를 받아서 최신 프레임으로 저장"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge 오류: {e}")

    # --- 3. OpenCV Contour 기능 ---
    def detect_object_2d(self, image):
        """(이전 논의) 2D Contour로 객체 위치/자세 파악"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # (임계값은 환경에 맞게 튜닝 필요)
        ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        
        # (노이즈 제거 - 선택 사항)
        # kernel = np.ones((5, 5), np.uint8)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            self.get_logger().warn("OpenCV: 이미지에서 물체를 찾지 못했습니다.")
            return None

        # 가장 큰 Contour를 객체로 간주
        main_contour = max(contours, key=cv2.contourArea)

        # 2D 위치 (중심점)
        M = cv2.moments(main_contour)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 2D 자세 (회전 각도)
        rect = cv2.minAreaRect(main_contour)
        angle = rect[2]
        
        # (중요!) 여기서 (cx, cy, angle)을 캘리브레이션 정보를 이용해
        # 로봇 베이스 좌표계의 7D 자세로 변환해야 하지만,
        # 여기서는 간단히 3D 벡터 [cx, cy, angle]로 저장합니다.
        # 이 부분은 추후에 반드시 7D로 변환하는 로직으로 대체되어야 합니다.
        return [cx, cy, angle]

    # --- 4. 녹화 제어 서비스 콜백 ---
    def toggle_recording_callback(self, request, response):
        """'ros2 service call /toggle_recording std_srvs/SetBool "data: true"' 명령으로 호출됨"""
        
        # --- 녹화 시작 ---
        if request.data and not self.is_recording:
            if self.latest_image is None:
                response.success = False
                response.message = "오류: 카메라 이미지를 수신하지 못했습니다."
                self.get_logger().error(response.message)
                return response

            # 1. 'Look': 현재 이미지로 목표 객체 자세 1회 감지
            self.goal_obj_pose = self.detect_object_2d(self.latest_image)
            
            if self.goal_obj_pose is None:
                response.success = False
                response.message = "오류: 객체 감지에 실패하여 녹화를 시작할 수 없습니다."
                self.get_logger().error(response.message)
                return response
            
            # 2. 녹화 상태 변경
            self.is_recording = True
            self.demonstration_id += 1 # 새 시연 ID 부여
            
            response.success = True
            response.message = f"녹화 시작 (Demonstration ID: {self.demonstration_id})"
            self.get_logger().info(response.message)
        
        # --- 녹화 중지 ---
        elif not request.data and self.is_recording:
            self.is_recording = False
            self.goal_obj_pose = None # 목표 자세 초기화
            
            response.success = True
            response.message = f"녹화 중지 (Demonstration ID: {self.demonstration_id})"
            self.get_logger().info(response.message)
        
        else:
            response.success = False
            response.message = "요청 오류 (이미 녹화 중이거나 중지 상태임)"
        
        return response

    # --- 5. 메인 데이터 로깅 루프 ---
    def logging_callback(self):
        """30Hz로 실행되며, 녹화 중일 때 데이터를 CSV에 저장합니다."""
        
        # 녹화 중이 아니거나, 목표가 설정되지 않았으면 아무것도 안 함
        if not self.is_recording or self.goal_obj_pose is None:
            return

        try:
            now = rclpy.time.Time()
            
            # --- 입력 1: 현재 로봇 자세 (Current Arm Pose) ---
            # (슬레이브 로봇의 엔드 이펙터 TF)
            t_current = self.tf_buffer.lookup_transform('world', 's_end_joint_y_link', now, timeout=rclpy.duration.Duration(seconds=0.1))
            current_arm_pose = [
                t_current.transform.translation.x, t_current.transform.translation.y, t_current.transform.translation.z,
                t_current.transform.rotation.x, t_current.transform.rotation.y, t_current.transform.rotation.z, t_current.transform.rotation.w
            ]
            
            # --- 입력 2: 목표 행동 (Target Arm Action) ---
            # (마스터 로봇의 엔드 이펙터 TF)
            t_target = self.tf_buffer.lookup_transform('world', 'm_link_3', now, timeout=rclpy.duration.Duration(seconds=0.1))
            target_arm_action = [
                t_target.transform.translation.x, t_target.transform.translation.y, t_target.transform.translation.z,
                t_target.transform.rotation.x, t_target.transform.rotation.y, t_target.transform.rotation.z, t_target.transform.rotation.w
            ]
            
            # --- 데이터 행(Row) 조합 ---
            timestamp = self.get_clock().now().nanoseconds
            
            # [demo_id, timestamp] + [goal 3개] + [current 7개] + [target 7개]
            row_data = [self.demonstration_id, timestamp] + \
                       self.goal_obj_pose + \
                       current_arm_pose + \
                       target_arm_action
                       
            # --- CSV 파일에 추가 (Append) ---
            df_row = pd.DataFrame([row_data], columns=CSV_COLUMNS)
            df_row.to_csv(CSV_PATH, mode='a', header=False, index=False)

        except TransformException as ex:
            self.get_logger().warn(f"TF 조회 실패, 데이터 로깅 건너뜀: {ex}", throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f"로깅 콜백 오류: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()