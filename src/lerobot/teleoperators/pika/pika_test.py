from pika.tracker import ViveTracker
import time

def main():
    # 1. 创建 tracker（可传 config）
    tracker = ViveTracker()

    # 2. 连接设备
    if not tracker.connect():
        print("连接失败")
        return

    print("已连接，开始读取数据...")

    try:
        while True:
            # 3. 获取所有设备位姿
            poses = tracker.get_pose()

            # 4. 打印
            for name, pose in poses.items():
                print(pose)

            time.sleep(0.02)  # 50 Hz读取

    except KeyboardInterrupt:
        print("停止")

    finally:
        # 5. 断开
        tracker.disconnect()


if __name__ == "__main__":
    main()