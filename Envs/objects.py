import numpy as np
from numpy import pi
from numpy.linalg import norm
from Utils.utils import *

import pybullet as p

class UR5_Agent:
    def __init__(self, ur5_start_pos, ur5_start_quat, ur5_reset_joint_pos=[0,0,0,0,0,0],
                 ur5_reset_joint_vel=[0,0,0,0,0,0], action_dim=6, ur5_control_mode='velocity',
                 ur5_force_limit=[150,150,150,28,28,28], urdf_path=None):
        '''
        :param ur5_start_pos:       [0, -0.4, 0.607]
        :param ur5_start_quat:      [0., 0., 0., 1.]
        :param ur5_reset_joint_pos: [pi/2, -pi/2, pi/2, -pi/2, 0, 0, 0]
        :param ur5_reset_joint_vel: [0, 0, 0, 0, 0, 0]
        :param action_dim:          6
        :param ur5_control_mode:    'velocity'
        :param ur5_force_limit:     [150, 150, 150, 28, 28, 28]
        :param urdf_path:           'urdf/ur5.urdf'
        '''
        self.ur5_start_pos = ur5_start_pos
        self.ur5_start_quat = ur5_start_quat
        self.ur5_reset_joint_pos = ur5_reset_joint_pos
        self.ur5_reset_joint_vel = ur5_reset_joint_vel
        self.action_dim = action_dim
        self.ur5_reset_linear_vel = (0., 0., 0.)
        self.ur5_reset_angular_vel = (0., 0., 0.)
        if ur5_control_mode == 'velocity':
            self.ur5_control_mode = p.VELOCITY_CONTROL
        elif ur5_control_mode == 'position':
            self.ur5_control_mode = p.POSITION_CONTROL
        self.ur5_force_limit = ur5_force_limit
        self.urdf_path = urdf_path

        self.id = p.loadURDF(self.urdf_path, self.ur5_start_pos, self.ur5_start_quat, flags=p.URDF_USE_SELF_COLLISION)

        self.limit_lower, self.limit_upper = [], []
        for i in range(self.action_dim):
            self.limit_lower.append(p.getJointInfo(self.id, i)[8])
            self.limit_upper.append(p.getJointInfo(self.id, i)[9])

    def reset(self):
        p.resetBasePositionAndOrientation(self.id, self.ur5_start_pos, self.ur5_start_quat)
        p.resetBaseVelocity(self.id, self.ur5_reset_linear_vel, self.ur5_reset_angular_vel)
        for i in range(self.action_dim):
            p.resetJointState(self.id, i, targetValue=self.ur5_reset_joint_pos[i], targetVelocity=self.ur5_reset_joint_vel[i])
            p.setJointMotorControl2(self.id, i, controlMode=self.ur5_control_mode, targetVelocity=self.ur5_reset_joint_vel[i], force=self.ur5_force_limit[i])

    def check_init(self):
        joint_pos = self.get_joint_pos()
        joint_vel = self.get_joint_vel()
        agent_pos = self.get_pos()
        agent_quat = self.get_quat()
        agent_linear_vel = self.get_linear_vel()
        agent_angular_vel = self.get_angular_vel()

        err_joint_pos = norm(np.sin(self.ur5_reset_joint_pos) - np.sin(joint_pos)) + \
                        norm(np.cos(self.ur5_reset_joint_pos) - np.cos(joint_pos))
        err_joint_vel = norm(np.array(joint_vel) - np.array(self.ur5_reset_joint_vel))
        err_agent_pos = norm(agent_pos - self.ur5_start_pos)
        err_agent_quat = cal_quat_error(agent_quat, self.ur5_start_quat)
        err_linear_vel = norm(agent_linear_vel)
        err_angular_vel = norm(agent_angular_vel)

        return err_joint_pos + err_joint_vel + err_agent_pos + err_agent_quat + err_linear_vel + err_angular_vel < 3e-3

    def make_action(self, action):
        joint_vel = np.hstack((action, np.zeros(self.action_dim - action.shape[0])))
        p.setJointMotorControlArray(self.id, [0, 1, 2, 3, 4, 5], controlMode=self.ur5_control_mode,
                                    targetVelocities=joint_vel.tolist(), forces=self.ur5_force_limit)

    def get_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.id)[0])

    def get_euler(self):
        return np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.id)[1]))

    def get_quat(self):
        return np.array(p.getBasePositionAndOrientation(self.id)[1])

    def get_end_effector_pos(self):
        return np.array(p.getLinkState(self.id, self.action_dim)[0])

    def get_end_effector_euler(self):
        return np.array(p.getEulerFromQuaternion(p.getLinkState(self.id, self.action_dim)[1]))

    def get_end_effector_quat(self):
        return np.array(p.getLinkState(self.id, self.action_dim)[1])

    def get_joint_pos(self):
        x = []
        for i in range(self.action_dim):
            x.append(p.getJointState(self.id, i)[0])
        return x

    def get_joint_vel(self):
        x = []
        for i in range(self.action_dim):
            x.append(p.getJointState(self.id, i)[1])
        return x

    def get_joint_pos_np(self):
        return np.array(self.get_joint_pos())

    def get_joint_vel_np(self):
        return np.array(self.get_joint_vel())

    def get_linear_vel(self):
        return np.array(p.getBaseVelocity(self.id)[0])

    def get_angular_vel(self):
        return np.array(p.getBaseVelocity(self.id)[1])

class SpaceObject:
    def __init__(self, start_pos, start_quat, urdf_path):
        '''
        :param start_pos:       [0, 0, 0] (satellite) / [-0.031, -0.944, 1.114] (mug)
        :param start_quat:      [0., 0., 0., 1.]
        :param urdf_path:       'urdf/satellite.urdf' (satellite) / 'urdf/mug.urdf' (mug)
        '''
        self.start_pos = start_pos
        self.start_quat = start_quat
        self.urdf_path = urdf_path

        self.id = p.loadURDF(self.urdf_path, self.start_pos, self.start_quat)

    def reset(self, pos, quat):
        self.start_pos = pos
        self.start_quat = quat
        p.resetBasePositionAndOrientation(self.id, self.start_pos, self.start_quat)

    def check_init(self):
        object_pos = self.get_pos()
        object_quat = self.get_quat()

        err_object_pos = norm(object_pos - self.start_pos)
        err_object_quat = cal_quat_error(object_quat, self.start_quat)

        return err_object_pos + err_object_quat < 1e-4

    def get_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.id)[0])

    def get_euler(self):
        return np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.id)[1]))

    def get_quat(self):
        return np.array(p.getBasePositionAndOrientation(self.id)[1])
