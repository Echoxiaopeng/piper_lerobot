import mujoco
import mujoco.viewer
import time
import numpy as np
# 加载逆解函数
from src.lerobot.robots.piper_follower.piper_IK.forward_inverse_kinematics import Arm_FK, Arm_IK


# 1. 加载模型与数据
model = mujoco.MjModel.from_xml_path('/home/echo/lerobot/mujoco/mujoco_model/piper_description.xml')
data = mujoco.MjData(model)

# 2. 启动被动可视化窗口 (launch_passive 不会阻塞 Python 主线程)
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    target_pose_1 = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]        # 姿态 1
    target_pose_2 = [1.0, 1.0, -1.2, 0.5, 0.5, 1.0, 0.03, -0.03]     # 姿态 2 (夹爪张开/闭合)
    
    current_target = target_pose_1
    last_toggle_time = time.time()

    # 3. 仿真主循环
    while viewer.is_running():
        step_start = time.time()
        
        # 每隔 3 秒在两个姿态之间切换，观察机械臂运动
        if time.time() - last_toggle_time > 3.0:
            if current_target == target_pose_1:
                current_target = target_pose_2
                print("切换到目标姿态 2")
            else:
                current_target = target_pose_1
                print("切换到目标姿态 1")
            last_toggle_time = time.time()
        
        # ─── 核心控制代码 ───
        # 将目标位置写入执行器的控制向量 data.ctrl 中
        # data.ctrl 的顺序与 XML 中 <actuator> 定义的顺序完全一致 (0 到 7)
        for i in range(model.nu):
            data.ctrl[i] = current_target[i]
        
        # 或者是直接整路赋值: data.ctrl[:] = current_target
        # ────────────────────

        # 物理引擎向前推进一个时间步长
        mujoco.mj_step(model, data)
        
        # 将最新的物理状态同步到可视化窗口
        viewer.sync()
        
        # 严格控制仿真速度，使其匹配真实时间（MuJoCo 默认步长 model.opt.timestep 通常是 0.002 秒）
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)