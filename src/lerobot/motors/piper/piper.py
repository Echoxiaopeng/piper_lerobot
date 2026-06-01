import time
from dataclasses import dataclass
from typing import Dict

from piper_sdk import C_PiperInterface_V2


@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]

class PiperMotorsBus:
    """
        对Piper SDK的二次封装
    """
    def __init__(self, 
                 config: PiperMotorsBusConfig):
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        self.motors = config.motors
        # 录制数据集时改成0
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [6 joints + 1 gripper] * 0.0
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pose_factor = 1000 # 单位 0.001mm
        self.joint_factor = 57324.840764 # 1000*180/3.14， rad -> 度（单位0.001度）

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]


    def connect(self, enable:bool) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if(enable):
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0,1000,0x01, 0)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0,1000,0x02, 0)
            print(f"使能状态: {enable_flag}")
            print("--------------------")
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else: 
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        print(f"Returning response: {resp}")
        # init_read = self.read()
        # list_init_read = list(init_read.values())
        # import math
        # list_init_read_degrees = [math.degrees(i) for i in list_init_read]
        # self.write(list_init_read)
        return resp

    def set_calibration(self):
        return
    
    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
            移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)

    def apply_calibration_master(self):
        """
            master移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)
        

    def write(self, target_joint:list):
        '''
        |joint_name|     limit(rad)       |    limit(angle)    |
        |----------|     ----------       |     ----------     |
        |joint1    |   [-2.6179, 2.6179]  |    [-150.0, 150.0] |
        |joint2    |   [0, 3.14]          |    [0, 180.0]      |
        |joint3    |   [-2.967, 0]        |    [-170, 0]       |
        |joint4    |   [-1.745, 1.745]    |    [-100.0, 100.0] |
        |joint5    |   [-1.22, 1.22]      |    [-70.0, 70.0]   |
        |joint6    |   [-2.09439, 2.09439]|    [-120.0, 120.0] |
        '''
        # input 是正常的弧度（rad）
        # 将弧度 转成 度 * 1000倍
        # round(i / 0.0174533 * 1e3)
        joint_0 = round(target_joint[0] / 0.0174533 * 1e3)
        joint_1 = round(target_joint[1] / 0.0174533 * 1e3)
        joint_2 = round(target_joint[2] / 0.0174533 * 1e3)
        joint_3 = round(target_joint[3] / 0.0174533 * 1e3)
        joint_4 = round(target_joint[4] / 0.0174533 * 1e3)
        joint_5 = round(target_joint[5] / 0.0174533 * 1e3)
        gripper_range = round(target_joint[6]*1000*1000)
        
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        # self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0) # 单位 0.001°

    def read(self) -> Dict:
        """
            - 机械臂关节消息,单位0.001度
            - 机械臂夹爪消息
        """
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        # 将读取到的 度*1000 转换到 弧度（rad）
        joint_state = tuple(getattr(joint_state, f"joint_{i+1}") / 1e3 * 0.0174533 for i in range(6))
        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state

        return {
            "joint_1": joint_state[0],
            "joint_2": joint_state[1],
            "joint_3": joint_state[2],
            "joint_4": joint_state[3],
            "joint_5": joint_state[4],
            "joint_6": joint_state[5],
            "gripper": gripper_state.grippers_angle
        }


    def safe_disconnect(self):
        """ 
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)

    def safe_disconnect_master(self):
        """ 
            Move to safe disconnect position
        """
        self.write_master(target_joint=self.safe_disable_position)