import mujoco
import numpy as np

# 1. 加载模型与数据
model = mujoco.MjModel.from_xml_path('/home/echo/lerobot/mujoco/mujoco_model/piper_description.xml')
data = mujoco.MjData(model)

class piper_control:
    # 是6个关节角 + gripper
    def __init__(self, model:mujoco.MjModel, data:mujoco.MjData):
        
        self.data = data
        self.model = model
        self.actuator_name = ["jont1","jont2","jont3","jont4","jont5","jont6","jont7","jont8"]
        self.actuator_id = [mujoco.mj_name2id(name) for name in self.actuator_name]
        
    def jont_control(self, joint):
        self.mj_control_joint = joint

        gripper = self.target_joint[6]
        gripper_width = gripper/2
        gripper_left = gripper_width
        gripper_right = -gripper_width

        for i in range(self.model.nu):
            actuator_id = self.actuator_id[i]

            self.mj_control_joint.append(gripper_left)
            self.mj_control_joint.append(gripper_right)
            ctrl_range = self.model.actuator_ctrlrange[actuator_id]

            self.mj_control_joint = np.clip(self.mj_control_joint, ctrl_range[0], ctrl_range[1])

            self.data.ctrl[actuator_id] = self.mj_control_joint[i]

    

