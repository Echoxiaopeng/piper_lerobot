import mujoco
import numpy as np
import mujoco.viewer
import time

class piper_control:
    # 6个关节角 + 1个gripper总宽度
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.data = data
        self.model = model
        self.actuator_name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
        
        # 💡 提示：新版推荐使用 mqT_ACTUATOR，如果报错可以换回旧版常量
        try:
            self.actuator_id = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mqT_ACTUATOR, name) for name in self.actuator_name]
        except AttributeError:
            self.actuator_id = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_name]
        
    def jont_control(self, joint):
        # 💡 修正 1：只取前 6 个关节，扔掉第 7 个总宽度位置，防止数组变成 9 维
        self.mj_control_joint = list(joint[:6])

        gripper = joint[6]
        gripper_width = gripper / 2.0
        gripper_left = gripper_width
        gripper_right = -gripper_width
        
        # 现在的数组完美组装成了 8 维
        self.mj_control_joint.append(gripper_left)
        self.mj_control_joint.append(gripper_right)
        
        for i in range(self.model.nu):
            actuator_id = self.actuator_id[i]
            ctrl_range = self.model.actuator_ctrlrange[actuator_id]

            # 💡 修正 2：只对当前这单个通道的数值进行 clip 限制
            safe_val = np.clip(self.mj_control_joint[i], ctrl_range[0], ctrl_range[1])
            self.data.ctrl[actuator_id] = safe_val


if __name__ == "__main__":
    # 1. 加载模型与数据
    model = mujoco.MjModel.from_xml_path('/home/echo/lerobot/mujoco/mujoco_model/piper_description.xml')
    data = mujoco.MjData(model)

    piper = piper_control(model, data)
    
    # 使用 with 上下文管理器安全管理窗口
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        
        # 定义测试动作序列 (6维角度 + 1维夹爪总宽度)
        # 0.0475 * 2 = 0.095 (全开)
        poses = [
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.095], # step 1
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],   # step 2 (闭合)
            [0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.095]  # step 3
        ]
        
        pose_idx = 0
        
        while viewer.is_running():
            print(f"正在执行当前姿态序列: Step {pose_idx + 1}")
            
            # 下发当前的目标姿态
            piper.jont_control(poses[pose_idx])
            
            # 修正 3：不能只 step 一下。给执行器 250 步物理步长（约0.5秒），让机械臂有时间动过去
            for _ in range(250):
                step_start = time.time()
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 维持正常的仿真频率
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            
            # 切换到下一个姿态
            pose_idx = (pose_idx + 1) % len(poses)