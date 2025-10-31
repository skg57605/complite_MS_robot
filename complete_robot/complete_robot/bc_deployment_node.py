# bc_deployment_node.py (의사 코드 포함)

import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import joblib
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
# (FK/IK 라이브러리 import, 예: KDL, tf2_ros)
# (다이나믹셀에 명령을 보낼 메시지 타입 import)

# --- 1. 모델과 스케일러 로드 ---
# train.py에서 MLP 클래스 정의를 그대로 복사해옴
class MLP(nn.Module):
    # (train.py와 동일한 모델 구조... )
    pass

class DeploymentNode(Node):
    def __init__(self):
        super().__init__('bc_deployment_node')
        
        # --- 2. 모델, 스케일러, FK/IK 솔버 초기화 ---
        self.INPUT_DIM = 14
        self.OUTPUT_DIM = 7
        self.model = MLP(self.INPUT_DIM, self.OUTPUT_DIM)
        self.model.load_state_dict(torch.load('bc_model.pth'))
        self.model.eval() # 👈 필수: 추론 모드로 설정
        
        self.scaler = joblib.load('data_scaler.pkl')
        
        # FK/IK 솔버 초기화 (예: URDF 로딩)
        # self.fk_solver = ...
        # self.ik_solver = ...
        
        self.goal_obj_pose = None # 초기 목표 자세를 저장할 변수
        self.current_arm_pose = None # 실시간 로봇 자세를 저장할 변수

        # --- 3. ROS 2 퍼블리셔/서브스크라이버/서비스 ---
        
        # 다이나믹셀 SDK가 읽을 수 있는 토픽으로 발행
        # (예시: goal_joint_states 토픽에 JointState 메시지를 발행)
        self.action_publisher = self.create_publisher(JointState, 'goal_joint_states', 10)
        
        # 다이나믹셀 SDK로부터 현재 모터 각도를 구독
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states', # (SDK가 발행하는 실제 토픽명으로 변경)
            self.joint_state_callback,
            10)
            
        # (perception_node에 목표 자세를 요청할 서비스 클라이언트...)
        # self.pose_client = ...

        # --- 4. 메인 제어 루프 ---
        self.timer = self.create_timer(0.05, self.control_loop) # 20Hz로 제어
        
        self.get_logger().info("배포 노드 초기화 완료. 목표 자세 대기 중...")
        self.get_initial_goal_pose() # (구현 필요) 작업 시작 시 1회 호출


    def joint_state_callback(self, msg):
        # 다이나믹셀의 현재 모터 각도(msg.position)를 받음
        # 👉 [브릿지 1: FK]
        # FK 솔버를 사용해 5개의 각도를 7D 자세(current_arm_pose)로 변환
        # self.current_arm_pose = self.fk_solver.solve(msg.position)
        pass

    def control_loop(self):
        # 1. 두 입력값이 모두 준비되었는지 확인
        if self.goal_obj_pose is None or self.current_arm_pose is None:
            return

        # 2. 모델 입력 준비 (14D)
        input_data = np.concatenate([self.goal_obj_pose, self.current_arm_pose])
        
        # 3. ⚠️ 스케일러로 변환
        input_data_scaled = self.scaler.transform([input_data])
        input_tensor = torch.Tensor(input_data_scaled)
        
        # 4. 추론 (모델이 다음 행동 예측)
        with torch.no_grad():
            target_arm_pose_tensor = self.model(input_tensor)
        
        target_arm_pose = target_arm_pose_tensor.squeeze().numpy() # 7D 목표 자세

        # 5. 👉 [브릿지 2: IK]
        # IK 솔버를 사용해 7D 목표 자세를 5개의 다이나믹셀 각도로 변환
        # target_joint_angles = self.ik_solver.solve(target_arm_pose) # [rad, rad, rad, rad, rad]

        # 6. 다이나믹셀 SDK로 명령 발행
        joint_cmd_msg = JointState()
        # (target_joint_angles를 SDK가 이해하는 메시지 형식으로 변환)
        # joint_cmd_msg.position = target_joint_angles 
        self.action_publisher.publish(joint_cmd_msg)

# (rclpy.init, spin 등 메인 함수...)