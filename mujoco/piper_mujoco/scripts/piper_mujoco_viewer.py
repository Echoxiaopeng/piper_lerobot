import mujoco
import mujoco.viewer
import time

# 1. 加载模型
model = mujoco.MjModel.from_xml_path('/home/echo/lerobot/mujoco/mujoco_model/piper_description.xml')
data = mujoco.MjData(model)

# 2. 启动可视化 Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 保持窗口运行
    while viewer.is_running():
        # 推进物理仿真步长
        mujoco.mj_step(model, data)
        
        # 刷新渲染画面
        viewer.sync()
        
        # 控制仿真速度大约在 60Hz
        time.sleep(0.016)