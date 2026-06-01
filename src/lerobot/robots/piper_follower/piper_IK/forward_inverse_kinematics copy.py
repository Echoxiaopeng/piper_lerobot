#!/usr/bin/env python3
import casadi
import math
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from transformations import quaternion_from_matrix
import os
import sys
import rospkg

# 模拟参数配置
class RobotArgs:
    def __init__(self):
        self.lift = False
        # 假设末端执行器（Gripper）在 joint6 基础上的局部偏移 [x, y, z, roll, pitch, yaw]
        # 0.19m 可能是你注释里提到的 Piper 夹爪长度
        self.gripper_xyzrpy = [0.19, 0.0, 0.0, 0.0, 0.0, 0.0]

def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]

def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    return transformation_matrix

class Arm_FK:
    def __init__(self, args):
        self.args = args
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('piper_description') 
        urdf_path = os.path.join(package_path, 'urdf', 'piper_description' + ('-lift.urdf' if args.lift else '.urdf'))

        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_path)
        self.mixed_jointsToLockIDs = ["joint7", "joint8"]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        self.first_matrix = create_transformation_matrix(0, 0, 0, 0, -1.57, 0)
        self.second_matrix = create_transformation_matrix(self.args.gripper_xyzrpy[0], self.args.gripper_xyzrpy[1], self.args.gripper_xyzrpy[2],
                                                          self.args.gripper_xyzrpy[3], self.args.gripper_xyzrpy[4], self.args.gripper_xyzrpy[5])
        self.last_matrix = np.dot(self.first_matrix, self.second_matrix)

        q = quaternion_from_matrix(self.last_matrix)
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([self.last_matrix[0, 3], self.last_matrix[1, 3], self.last_matrix[2, 3]]),
                      ),
                      pin.FrameType.OP_FRAME)
        )

    def get_pose(self, q):
        index = 6 + (1 if self.args.lift else 0)
        # index = self.reduced_robot.model.getFrameId("ee")
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, np.array(q))
        
        end_pose = create_transformation_matrix(self.reduced_robot.data.oMi[index].translation[0], self.reduced_robot.data.oMi[index].translation[1], self.reduced_robot.data.oMi[index].translation[2],
                                                math.atan2(self.reduced_robot.data.oMi[index].rotation[2, 1], self.reduced_robot.data.oMi[index].rotation[2, 2]),
                                                math.asin(-self.reduced_robot.data.oMi[index].rotation[2, 0]),
                                                math.atan2(self.reduced_robot.data.oMi[index].rotation[1, 0], self.reduced_robot.data.oMi[index].rotation[0, 0]))
        
        end_pose = np.dot(end_pose, self.last_matrix)
        return matrix_to_xyzrpy(end_pose)


class Arm_IK:
    def __init__(self, args):
        self.args = args
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('piper_description') 
        urdf_path = os.path.join(package_path, 'urdf', 'piper_description' + ('-lift.urdf' if args.lift else '.urdf'))

        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_path)
        self.mixed_jointsToLockIDs = ["joint7", "joint8"]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        self.first_matrix = create_transformation_matrix(0, 0, 0, 0, -1.57, 0)
        self.second_matrix = create_transformation_matrix(self.args.gripper_xyzrpy[0], self.args.gripper_xyzrpy[1], self.args.gripper_xyzrpy[2],
                                                          self.args.gripper_xyzrpy[3], self.args.gripper_xyzrpy[4], self.args.gripper_xyzrpy[5])
        self.last_matrix = np.dot(self.first_matrix, self.second_matrix)
        q = quaternion_from_matrix(self.last_matrix)
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([self.last_matrix[0, 3], self.last_matrix[1, 3], self.last_matrix[2, 3]]),
                      ),
                      pin.FrameType.OP_FRAME)
        )

        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        for i in range(4, 10):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # Casadi 模型初始化
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)).vector,
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)

        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]  
        ori_error = error_vec[3:]  
        weight_position = 1.0  
        weight_orientation = 0.1  
        
        self.totalcost = casadi.sumsqr(weight_position * pos_error) + casadi.sumsqr(weight_orientation * ori_error)
        self.regularization = casadi.sumsqr(self.var_q)

        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper_arr = np.array([gripper/2.0, -gripper/2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        # ❌ 已修复：移除了未定义的 self.vis 调用
        self.opti.set_value(self.param_tf, target_pose)

        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                self.init_data = sol_q
                if max_diff > 30.0/180.0*3.1415:
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q

            self.history_data = sol_q

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v,
                              np.zeros(self.reduced_robot.model.nv))

            is_collision = self.check_self_collision(sol_q, gripper_arr)
            return sol_q, tau_ff, not is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info. {e}")
            return None, '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        return collision

    def get_pose(self, q):
        index = 6 + (1 if self.args.lift else 0)
        # index = self.reduced_robot.model.getFrameId("ee")
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, np.array(q))
        end_pose = create_transformation_matrix(self.reduced_robot.data.oMi[index].translation[0], self.reduced_robot.data.oMi[index].translation[1], self.reduced_robot.data.oMi[index].translation[2],
                                                math.atan2(self.reduced_robot.data.oMi[index].rotation[2, 1], self.reduced_robot.data.oMi[index].rotation[2, 2]),
                                                math.asin(-self.reduced_robot.data.oMi[index].rotation[2, 0]),
                                                math.atan2(self.reduced_robot.data.oMi[index].rotation[1, 0], self.reduced_robot.data.oMi[index].rotation[0, 0]))
        end_pose = np.dot(end_pose, self.last_matrix)
        return matrix_to_xyzrpy(end_pose)

def main():
    print("=================== 开始测试 Piper IK 逆运动学 ===================")

    args = RobotArgs()
    fk_solver = Arm_FK(args)
    ik_solver = Arm_IK(args)

    # 给一组测试关节角（假设 6 个自由度）
    target_joint = np.array([0.1, -0.2, 0.3, 0.1, 0.5, -0.1]) 
    
    # 1. 获取目标位姿
    xyzrpy = fk_solver.get_pose(target_joint)
    print(f"目标位姿 (xyzrpy): {xyzrpy}")
    
    # 2. 根据目标位姿生成 4x4 目标矩阵
    target_pose_matrix = create_transformation_matrix(xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5])

    # 3. 求解逆运动学
    sol_q, tau_ff, success = ik_solver.ik_fun(
        target_pose=target_pose_matrix, 
        gripper=0.0
    )
    
    print(f"逆运动学求解成功: {success}")
    print(f"求解出的关节角 (sol_q): {sol_q}")
    
    if sol_q is not None:
        # 4. 通过解出来的关节角，验证其正运动学位姿
        verified_pose = ik_solver.get_pose(sol_q)
        
        # 5. 精准计算矩阵误差
        T_target = pin.SE3(target_pose_matrix)
        
        # 重新构建验证矩阵
        verified_pose_matrix = create_transformation_matrix(
            verified_pose[0], verified_pose[1], verified_pose[2], 
            verified_pose[3], verified_pose[4], verified_pose[5]
        )
        T_verified = pin.SE3(verified_pose_matrix)
        
        # 计算两个空间位姿之间的相对变换误差
        # 用 log6 得到的 6维向量：前3维是空间平移误差(m)，后3维是空间旋转误差(rad)
        error_se3 = pin.log6(T_verified.inverse() * T_target).vector
        
        print("\n============= 误差分析 =============")
        print(f"绝对位置误差 (X, Y, Z) 单位:米:\n {error_se3[:3]}")
        print(f"绝对姿态轴角误差 (Rx, Ry, Rz) 单位:弧度:\n {error_se3[3:]}")
        
        # 如果位置误差小于 1mm 且角度误差极小，说明解完全正确
        if np.allclose(error_se3[:3], 0, atol=1e-3) and np.allclose(error_se3[3:], 0, atol=1e-2):
            print("🎉 IK 解算完全正确！")
        else:
            print("❌ 解算收敛，但位姿未对齐，请检查 cpin.log6 内部的目标 Frame 是否一致。")


if __name__ == '__main__':
    main()