#!/usr/bin/env python

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents

from .configuration_pika import PikaTeleopConfig

from pika.tracker import ViveTracker
from pika.sense import Sense
import numpy as np



class PikaTeleop(Teleoperator):
    """
    Teleop class to use pika inputs for control.
    """

    config_class = PikaTeleopConfig
    name = "pika"

    def __init__(self, config: PikaTeleopConfig):
        # require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        # self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

        # 用来记录pika是否已经连接
        self.is_connected_pika = False

        self.pika = ViveTracker()
        self.sense_gripper = Sense()

    @property
    def action_features(self) -> dict:
        """
        如实描述 Pika (Vive Tracker) 设备输出的数据特征
        """
        return {
            "pika.pos": np.ndarray,      # 空间 3 维位置 [x, y, z]
            "pika.rot": np.ndarray,      # 空间 4 维四元数 [qx, qy, qz, qw]
            "pika.timestamp": float,     # Tracker 的时间戳
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.is_connected_pika 

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        logging.info("正在连接 Pika (Vive Tracker)...")
        self.is_connected_pika = self.pika.connect()
        if self.is_connected_pika:
            logging.info("Pika 连接成功，后台追踪线程已启动。")
        else:
            logging.error("Pika 连接失败，请检查硬件或驱动。")

        is_connected_sense = self.sense_gripper.connect()
        if is_connected_sense:
            logging.info("Sense 连接成功，后台追踪线程已启动。")
        else:
            logging.error("Sense 连接失败，请检查硬件或驱动。")




    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        pass

    def _on_release(self, key):
        pass

    def _drain_pressed_keys(self):
        pass

    def configure(self):
        pass
    
    @check_if_not_connected
    def get_action(self) -> RobotAction:
            pose_data = self.pika.get_pose("T20")
            gripper = self.sense_gripper.get_gripper_distance()
            if pose_data is None:
                return {"pika.pos": np.zeros(3), "pika.rot": np.array([0,0,0,1]), "pika.timestamp": time.time(),"pika.gripper": 0}
            
            return {
                "pika.pos": np.array(pose_data.position, dtype=np.float32),
                "pika.rot": np.array(pose_data.rotation, dtype=np.float32),
                "pika.timestamp": float(pose_data.timestamp),
                "pika.gripper": float(gripper)
            }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
