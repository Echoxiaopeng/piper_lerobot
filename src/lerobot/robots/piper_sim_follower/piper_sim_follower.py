#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_piper_sim_follower import PIPERSimFollowerConfig

import numpy as np

from mujoco_sim.piper_mujoco.scripts import piper_control
import mujoco
import mujoco.viewer
logger = logging.getLogger(__name__)

def get_motor_names(arm: dict[str, Any]) -> list[str]:
    return [motor for arm_key, bus in arm.items() for motor in bus.motors]

class PIPERSimFollower(Robot):
    config_class = PIPERSimFollowerConfig
    name = "piper_sim_follower"

    def __init__(self, config: PIPERSimFollowerConfig):
        super().__init__(config)
        self.config = config
        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)
        self.xml_path = "/home/echo/lerobot/src/mujoco_sim/mujoco_model/piper_description.xml"
        
        print(f"[LeRobot Sim] 正在加载 MuJoCo 机器人模型: {self.xml_path}")
        
        # 2.直接在内部一口气把物理引擎的核心骨架全部初始化
        
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = None
        # 3.把刚生成的 model 和 data 喂给你写好的控制器
        self.piper = piper_control(self.mj_model, self.mj_data)
 

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,   
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        state_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        """用于 record/replay 的电机动作描述"""
        motor_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

        return {f"{name}.pos": float for name in motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """用于 record/replay 的相机图像描述"""
        return {
            f"observation.images.{cam_key}": (cam.height, cam.width, 3)
            for cam_key, cam in self.cameras.items()
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def configure(self, **kwargs):
        # 不做任何事,过抽象类用
        pass

    @property
    def is_connected(self) -> bool:
        """机器人和所有相机是否都已连接"""

        return all(cam.is_connected for cam in self.cameras.values()) and self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """机器人是否已完成标定"""

        return True

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)
    

    @check_if_already_connected
    def connect(self) -> None:
        """Connect piper and cameras"""
        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            self._is_connected = self._is_connected and self.cameras[name].is_connected
            print(f"camera {name} connected")
   
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            print("[LeRobot Sim] MuJoCo 仿真可视化窗口已成功拉起。")

        print("All connected")
        self._is_connected = True

        self.calibrate()

    def disconnect(self) -> None:
        """move to home position, disenable piper and cameras"""
        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self._is_connected = False

    def calibrate(self):
        """move piper to the home position"""
        if not self._is_connected:
            raise ConnectionError()
        
        self._is_calibrated = True  # 标记为已标定

    @check_if_not_connected
    def get_observation(self) -> dict:
        """Capture current joint positions and camera images"""
        # if not self._is_connected:
        #     raise DeviceNotConnectedError("Piper is not connected. Run `robot.connect()` first.")
    

        # 读取关节状态
        state = self.piper.read_joint()
        obs_dict = {f"{joint}.pos": float(val) for joint, val in state.items()}
        # print(f"obs_dict:{obs_dict}")

        # 读取图像
        for name, cam in self.cameras.items():
            obs_dict[f"observation.images.{name}"] = cam.async_read()

        return obs_dict


    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Receive action dict from teleop/record and send to motor"""

        motor_order = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper"
        ]


        # 当前状态
        current_state = self.piper.read_joint()


        # 范围是 2（关） - 97（开）
        sense_gripper = action["pika.gripper"]

        # 若 action 中缺少某关节，则保持当前位置
        target_joints = [
            float(action.get(f"{motor}.pos", current_state[motor]))
            for motor in motor_order
        ]
        sim_gripper = (sense_gripper - 2) * 0.001

        target_joints[6] = sim_gripper

        # print("Target joints:", target_joints)

        self.piper.jont_control(target_joints)


        # 2. 重点：让物理引擎往前踩几步，直到把这段时间间隙填满
        # 假设 LeRobot 发控制率是 50Hz (20ms一次)，而 MuJoCo 的 timestep 是 2ms (0.002s)
        # 那么每一次 send_action 内部需要让物理引擎走 20ms / 2ms = 10 步
        control_period = 0.02  # 50Hz 对应 0.02 秒
        sim_timestep = self.mj_model.opt.timestep
        steps_to_run = int(control_period / sim_timestep) if sim_timestep > 0 else 10

        for _ in range(steps_to_run):
            mujoco.mj_step(self.mj_model, self.mj_data)
        # mujoco.mj_step(self.mj_model, self.mj_data)
        # 3. 刷新可视化窗口
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()


        return {f"{motor}.pos": pos for motor, pos in zip(motor_order, target_joints)}