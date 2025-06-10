'''
@author: Andrius Bernatavicius, 2018
'''
import os
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
# import sim
import time
import numpy as np
from numpy import pi
from numpy import e
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=3, suppress=True)
from numpy.linalg import norm
import platform
import torch

import math
import sys
sys.path.append('..')

# from Utils.robot_cal import *
from Utils.utils import *
from gym.utils import seeding

import pybullet as p
import pybullet_data

from Envs.objects import UR5_Agent, SpaceObject
import copy

_2pi = 2 * pi
# abbreviation
#     pose: pose;
#     pos:  position(s);
#     ori:  orientation;
#     quat: quaternion;
#     vel:  velocity(ies);

class PybulletSpaceManipulatorReacherMultiagent:
    def __init__(self, args, headless=None, plat='wsl', pos_bias=0):
        super(PybulletSpaceManipulatorReacherMultiagent, self).__init__()
        ''' DATA FROM MAIN CODE '''
        self.noa = args.noa
        self.plat = plat
        if self.plat in ['wsl', 'ubuntu']:
            headless = args.headless if headless is None else headless
        else:
            headless = True

        self.action_dim = args.action_dim
        self.time_interval = args.time_interval

        physicsClient = p.connect(p.DIRECT) if headless else p.connect(p.GUI,)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 关闭整个调试GUI，包括网格
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, 0)

        self.UR5_START_POS = np.array([0, -0.4, 0.607])
        self.UR5_START_QUAT = np.array(p.getQuaternionFromEuler([0, 0, 0]))
        self.UR5_RESET_JOINT_POS = np.array([pi/2, -pi/2, pi/2, -pi/2, 0, 0, ])
        self.UR5_RESET_JOINT_VEL = np.array([0, 0, 0, 0, 0, 0, ])
        self.UR5_FORCE_LIMIT = np.array([150, 150, 150, 28, 28, 28])

        self.SATELLITE_START_POS = np.array([0, 0, 0])
        self.SATELLITE_START_QUAT = np.array(p.getQuaternionFromEuler([0, 0, 0]))
        self.MUG_START_POS = np.array([-0.031, -0.944, 1.114])
        self.MUG_START_QUAT = np.array(p.getQuaternionFromEuler([0, 0, 0]))

        self.bias_list_x = [-12, -6, 0, 6, 12]
        self.bias_list_y = [ -6, -3, 0, 3, 6 ]
        self.bias_list_z = [ -6, -3, 0, 3, 6 ]

        self.ur5_start_pos_np           = np.zeros([self.noa, 3])
        self.ur5_start_quat_np          = np.zeros([self.noa, 4])
        self.satellite_start_pos_np     = np.zeros([self.noa, 3])
        self.satellite_start_quat_np    = np.zeros([self.noa, 4])
        self.mug_start_pos_np           = np.zeros([self.noa, 3])
        self.mug_start_quat_np          = np.zeros([self.noa, 4])

        self.ur5_start_pos_np           += self.UR5_START_POS
        self.ur5_start_quat_np          += self.UR5_START_QUAT
        self.satellite_start_pos_np     += self.SATELLITE_START_POS
        self.satellite_start_quat_np    += self.SATELLITE_START_QUAT
        self.mug_start_pos_np           += self.MUG_START_POS
        self.mug_start_quat_np          += self.MUG_START_QUAT

        for i in range(self.noa):
            j = i + pos_bias
            self.ur5_start_pos_np[i][0]         += self.bias_list_x[j % len(self.bias_list_x)]
            self.ur5_start_pos_np[i][1]         += self.bias_list_y[int(j / len(self.bias_list_x)) % len(self.bias_list_y)]
            self.ur5_start_pos_np[i][2]         += self.bias_list_z[int(j / len(self.bias_list_x) / len(self.bias_list_y))]
            self.satellite_start_pos_np[i][0]   += self.bias_list_x[j % len(self.bias_list_x)]
            self.satellite_start_pos_np[i][1]   += self.bias_list_y[int(j / len(self.bias_list_x)) % len(self.bias_list_y)]
            self.satellite_start_pos_np[i][2]   += self.bias_list_z[int(j / len(self.bias_list_x) / len(self.bias_list_y))]
            self.mug_start_pos_np[i][0]         += self.bias_list_x[j % len(self.bias_list_x)]
            self.mug_start_pos_np[i][1]         += self.bias_list_y[int(j / len(self.bias_list_x)) % len(self.bias_list_y)]
            self.mug_start_pos_np[i][2]         += self.bias_list_z[int(j / len(self.bias_list_x) / len(self.bias_list_y))]

        self.agent_list, self.floating_base_list, self.target_list = [], [], []

        # load model of satellite / arm / target
        for i in range(self.noa):
            self.agent_list.append(
                UR5_Agent(
                    ur5_start_pos=self.ur5_start_pos_np[i],
                    ur5_start_quat=self.ur5_start_quat_np[i],
                    ur5_reset_joint_pos=self.UR5_RESET_JOINT_POS,
                    ur5_reset_joint_vel=self.UR5_RESET_JOINT_VEL,
                    action_dim=self.action_dim,
                    ur5_control_mode='velocity',
                    ur5_force_limit=self.UR5_FORCE_LIMIT,
                    urdf_path='urdf/ur5.urdf'
                )
            )
            self.floating_base_list.append(
                SpaceObject(self.satellite_start_pos_np[i], self.satellite_start_quat_np[i], 'urdf/satellite.urdf')
            )
            self.target_list.append(
                SpaceObject(self.mug_start_pos_np[i], self.mug_start_quat_np[i], 'urdf/mug.urdf')
            )

            # 设置无阻尼和无摩擦力
            p.changeDynamics(self.agent_list[i].id, -1, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0)
            p.changeDynamics(self.floating_base_list[i].id, -1, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0)
            # 固定 floating_base 和 agent
            p.createConstraint(
                parentBodyUniqueId=self.floating_base_list[i].id,
                parentLinkIndex=-1,  # 连接根链接
                childBodyUniqueId=self.agent_list[i].id,
                childLinkIndex=-1,   # 连接根链接
                jointType=p.JOINT_FIXED,  # 旋转关节
                jointAxis=[0, 0, 0],  # 关节绕 Z 轴旋转
                parentFramePosition=self.UR5_START_POS.tolist(),  # 关节在 cubeA 上的位置
                childFramePosition=[0, 0, 0]  # 关节在 cubeB 上的位置
            )
            # 设置碰撞组
            p.setCollisionFilterGroupMask(
                bodyUniqueId=self.floating_base_list[i].id,
                linkIndexA=-1,
                collisionFilterGroup=i,
                collisionFilterMask=i
            )
            p.setCollisionFilterGroupMask(
                bodyUniqueId=self.agent_list[i].id,
                linkIndexA=-1,
                collisionFilterGroup=i,
                collisionFilterMask=i
            )
            for j in range(p.getNumJoints(self.floating_base_list[i].id)):
                p.setCollisionFilterGroupMask(
                    bodyUniqueId=self.floating_base_list[i].id,
                    linkIndexA=j,
                    collisionFilterGroup=i,
                    collisionFilterMask=i
                )
            for j in range(p.getNumJoints(self.agent_list[i].id)):
                p.setCollisionFilterGroupMask(
                    bodyUniqueId=self.agent_list[i].id,
                    linkIndexA=j,
                    collisionFilterGroup=i,
                    collisionFilterMask=i
                )

        p.setTimeStep(self.time_interval / 1000.)    # 50 ms

        # region
        self.max_episode_steps = args.max_episode_steps

        self.is_train = args.is_train

        target_pos_reinit_methods = ['random', 'sequence']
        self.target_pos_reinit_method = args.target_pos_reinit_method
        assert (self.target_pos_reinit_method in target_pos_reinit_methods)

        # M, N, Q: split r/theta/phi into N pieces
        MNQ = args.MNQ.split()
        self.M, self.N, self.Q = int(MNQ[0]), int(MNQ[1]), int(MNQ[2])
        self.MNQ = self.M * self.N * self.Q
        self.r_max, self.r_min = args.r_max, args.r_min
        self.d_r, self.d_theta, self.d_phi = (self.r_max - self.r_min) / self.M, pi / 2 / self.N, _2pi / self.Q

        self.target_reinit_pos_freq = args.target_reinit_pos_freq

        self.episode_num = 0

        self.action_np = np.zeros([self.noa, self.action_dim])
        self.last_action_np = copy.copy(self.action_np)

        self.success_interval = args.success_interval

        self.pos_err_stable_times   = args.pos_err_stable_times
        self.pos_err_thres          = args.pos_err_thres
        self.arm_angle_0_to_2pi     = args.arm_angle_0_to_2pi
        self.success_sample_rate    = args.success_sample_rate
        self.success_min_rate       = args.success_min_rate

        self.ori_err_stable_times   = args.pos_err_stable_times
        self.ori_err_thres          = args.ori_err_thres
        self.ori_inherent_rate      = args.ori_inherent_rate
        self.ori_penalty_rate       = args.ori_penalty_rate

        self.pos_related_to_finish  = args.pos_related_to_finish
        self.ori_related_to_finish  = args.ori_related_to_finish

        self.decrease_threshold             = args.decrease_threshold
        self.success_sample_rate_idx        = args.success_sample_rate_idx
        self.pos_err_stable_times_increase  = args.pos_err_stable_times_increase
        # Discrete Curriculum Learning Parameters
        self.pos_err_thres_idx              = args.pos_err_thres_idx
        self.ori_err_thres_idx              = args.ori_err_thres_idx

        self._sphere_reach_fail_prob_reset()

        self.norm_reach_fail_prob = (self.sphere_reach_fail_prob / np.sum(self.sphere_reach_fail_prob)).reshape(-1)
        self.norm_reach_fail_prob[-1] += 1

        reward_types = ['dense', 'sparse']
        self.reward_type = args.reward_type
        assert (self.reward_type in reward_types)

        ''' position error penalty define '''
        self.use_pos_err_penalty    = args.use_pos_err_penalty

        ''' orientation error penalty define '''
        self.use_ori_err_penalty    = args.use_ori_err_penalty

        ''' orientation decrease reward define '''
        self.use_ori_decrease_reward    = args.use_ori_decrease_reward
        self.ori_decrease_reward_rate   = args.ori_decrease_reward_rate

        ''' other penalty define '''
        self.use_smooth_penalty                     = args.use_smooth_penalty

        ''' done reward define '''
        self.use_done_reward    = args.use_done_reward
        self.done_reward_rate   = args.done_reward_rate

        ''' position zero origin define '''
        pos_zero_origins        = ['ee', 'ab']
        self.pos_zero_origin    = args.pos_zero_origin
        assert (self.pos_zero_origin in pos_zero_origins)

        ''' state space define '''
        state_dim_chooses       = ['eterrcprdpopv', 'eterr3p4qdpopv', 'eterr3p3edpopv',]
        self.state_dim_choose   = args.state_dim_choose
        assert (self.state_dim_choose in state_dim_chooses)
        self.state_dim          = args.state_dim

        ''' her cache define '''
        if args.use_her:
            self.her_cache_init()
        # endregion

        self.end_effector_axis_list = []

        # env reset
        for i in range(self.noa):
            self.floating_base_list[i].reset(self.satellite_start_pos_np[i], self.satellite_start_quat_np[i])
            self.agent_list[i].reset()
            self.target_list[i].reset(self.mug_start_pos_np[i], self.mug_start_quat_np[i])

        self.init_floating_base_pos_np          = np.empty((0, 3))
        self.init_floating_base_quat_np         = np.empty((0, 4))
        self.init_floating_base_euler_np        = np.empty((0, 3))

        self.init_agent_pos_np                  = np.empty((0, 3))
        self.init_agent_quat_np                 = np.empty((0, 4))
        self.init_agent_euler_np                = np.empty((0, 3))

        self.init_joint_pos_np                  = np.empty((0, self.action_dim))
        self.init_joint_vel_np                  = np.empty((0, self.action_dim))

        self.init_target_pos_np                 = np.empty((0, 3))
        self.init_target_quat_np                = np.empty((0, 4))
        self.init_target_euler_np               = np.empty((0, 3))

        self.init_end_effector_pos_np           = np.empty((0, 3))
        self.init_end_effector_quat_np          = np.empty((0, 4))
        self.init_end_effector_euler_np         = np.empty((0, 3))

        for i in range(self.noa):
            ######### get floating_base pos/quat/euler #########
            self.init_floating_base_pos_np = np.vstack((self.init_floating_base_pos_np, self.floating_base_list[i].get_pos()))
            self.init_floating_base_quat_np = np.vstack((self.init_floating_base_quat_np, self.floating_base_list[i].get_quat()))
            self.init_floating_base_euler_np = np.vstack((self.init_floating_base_euler_np, self.floating_base_list[i].get_euler()))
            ######### get agent pos/quat/euler #########
            self.init_agent_pos_np = np.vstack((self.init_agent_pos_np, self.agent_list[i].get_pos()))
            self.init_agent_quat_np = np.vstack((self.init_agent_quat_np, self.agent_list[i].get_quat()))
            self.init_agent_euler_np = np.vstack((self.init_agent_euler_np, self.agent_list[i].get_euler()))
            ######### get joint pos/vel #########
            self.init_joint_pos_np = np.vstack((self.init_joint_pos_np, self.agent_list[i].get_joint_pos_np()))
            self.init_joint_vel_np = np.vstack((self.init_joint_vel_np, self.agent_list[i].get_joint_vel_np()))
            ######### get target pos/quat/euler #########
            self.init_target_pos_np = np.vstack((self.init_target_pos_np, self.target_list[i].get_pos()))
            self.init_target_quat_np = np.vstack((self.init_target_quat_np, self.target_list[i].get_quat()))
            self.init_target_euler_np = np.vstack((self.init_target_euler_np, self.target_list[i].get_euler()))
            ######### get end_effector pos/quat/euler #########
            self.init_end_effector_pos_np = np.vstack((self.init_end_effector_pos_np, self.agent_list[i].get_end_effector_pos()))
            self.init_end_effector_quat_np = np.vstack((self.init_end_effector_quat_np, self.agent_list[i].get_end_effector_quat()))
            self.init_end_effector_euler_np = np.vstack((self.init_end_effector_euler_np, self.agent_list[i].get_end_effector_euler()))

        if self.arm_angle_0_to_2pi:
            self.init_joint_pos_np %= _2pi

        if self.pos_zero_origin == 'ee':
            self.pos_zero_origin_np = self.init_end_effector_pos_np - self.init_floating_base_pos_np
        elif self.pos_zero_origin == 'ab':
            self.pos_zero_origin_np = self.init_agent_pos_np - self.init_floating_base_pos_np

        ''' copy '''
        self.floating_base_pos_np           = copy.copy(self.init_floating_base_pos_np)
        self.floating_base_quat_np          = copy.copy(self.init_floating_base_quat_np)
        self.floating_base_euler_np         = copy.copy(self.init_floating_base_euler_np)

        self.agent_pos_np                   = copy.copy(self.init_agent_pos_np)
        self.agent_quat_np                  = copy.copy(self.init_agent_quat_np)
        self.agent_euler_np                 = copy.copy(self.init_agent_euler_np)

        self.joint_pos_np                   = copy.copy(self.init_joint_pos_np)
        self.joint_vel_np                   = copy.copy(self.init_joint_vel_np)

        self.start_target_pos_np            = copy.copy(self.init_target_pos_np)
        self.start_target_quat_np           = copy.copy(self.init_target_quat_np)
        self.start_target_euler_np          = copy.copy(self.init_target_euler_np)
        self.target_pos_np                  = copy.copy(self.init_target_pos_np)
        self.target_quat_np                 = copy.copy(self.init_target_quat_np)
        self.target_euler_np                = copy.copy(self.init_target_euler_np)

        self.target_floating_base_quat_np   = copy.copy(self.init_floating_base_quat_np)
        self.target_floating_base_euler_np  = copy.copy(self.init_floating_base_euler_np)

        self.end_effector_pos_np            = copy.copy(self.init_end_effector_pos_np)
        self.end_effector_quat_np           = copy.copy(self.init_end_effector_quat_np)

        self.end_effector_euler_np          = copy.copy(self.init_end_effector_euler_np)

        self.done_np = np.zeros(self.noa).astype(int)
        self.his_obs_np = {}
        self._history_observation_reset_np(self.his_obs_np)
        self._make_observation()
        self._last_refresh_np()

        self._wandb_init(mode=args.wandb_mode, project_name=args.project_name, name=args.wandb_name)

    def _wandb_init(self, mode='online', project_name='free-flying', name=None):
        self.wandb = wandb.init(
            mode=mode,
            project=project_name,
            name=name,
        )

    def her_cache_init(self):
        self.her_his_obs_list_cache_np = {}
        self._history_observation_reset_zero_np(self.her_his_obs_list_cache_np, cache_size=self.noa*2)
        self.r_cahce_np, self.theta_cahce_np, self.phi_cache_np = np.empty(0).astype(int), np.empty(0).astype(int), np.empty(0).astype(int)
        self.her_his_obs_list_cache_np_len = 0

    def her_cache_post_process(self):
        ''' FROM _MAKE_OBSERVATION '''
        self.her_his_obs_list_cache_np['floating_base_pos']         = \
            concat((self.her_his_obs_list_cache_np['floating_base_pos'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)
        self.her_his_obs_list_cache_np['floating_base_quat']        = \
            concat((self.her_his_obs_list_cache_np['floating_base_quat'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 4))), axis=1)
        self.her_his_obs_list_cache_np['floating_base_euler']       = \
            concat((self.her_his_obs_list_cache_np['floating_base_euler'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)

        self.her_his_obs_list_cache_np['floating_base_target_quat']  = \
            concat((self.her_his_obs_list_cache_np['floating_base_target_quat'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 4))), axis=1)
        self.her_his_obs_list_cache_np['floating_base_target_euler'] = \
            concat((self.her_his_obs_list_cache_np['floating_base_target_euler'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)

        self.her_his_obs_list_cache_np['agent_pos'] = \
            concat((self.her_his_obs_list_cache_np['agent_pos'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)
        self.her_his_obs_list_cache_np['agent_quat'] = \
            concat((self.her_his_obs_list_cache_np['agent_quat'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 4))), axis=1)
        self.her_his_obs_list_cache_np['agent_euler'] = \
            concat((self.her_his_obs_list_cache_np['agent_euler'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)

        self.her_his_obs_list_cache_np['joint_pos'] = \
            concat((self.her_his_obs_list_cache_np['joint_pos'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, self.action_dim))), axis=1)
        self.her_his_obs_list_cache_np['joint_vel'] = \
            concat((self.her_his_obs_list_cache_np['joint_vel'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, self.action_dim))), axis=1)

        self.her_his_obs_list_cache_np['target_pos'] = \
            concat((self.her_his_obs_list_cache_np['target_pos'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)
        self.her_his_obs_list_cache_np['target_quat'] = \
            concat((self.her_his_obs_list_cache_np['target_quat'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 4))), axis=1)
        self.her_his_obs_list_cache_np['target_euler'] = \
            concat((self.her_his_obs_list_cache_np['target_euler'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)

        self.her_his_obs_list_cache_np['end_effector_pos'] = \
            concat((self.her_his_obs_list_cache_np['end_effector_pos'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)
        self.her_his_obs_list_cache_np['end_effector_quat'] = \
            concat((self.her_his_obs_list_cache_np['end_effector_quat'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 4))), axis=1)
        self.her_his_obs_list_cache_np['end_effector_euler'] = \
            concat((self.her_his_obs_list_cache_np['end_effector_euler'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)

        ''' FROM _GET_STATE '''
        self.her_his_obs_list_cache_np['err_pos'] = \
            concat((self.her_his_obs_list_cache_np['err_pos'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)
        self.her_his_obs_list_cache_np['err_quat'] = \
            concat((self.her_his_obs_list_cache_np['err_quat'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 4))), axis=1)
        self.her_his_obs_list_cache_np['err_euler'] = \
            concat((self.her_his_obs_list_cache_np['err_euler'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)

        self.her_his_obs_list_cache_np['err_floating_base_quat'] = \
            concat((self.her_his_obs_list_cache_np['err_floating_base_quat'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 4))), axis=1)
        self.her_his_obs_list_cache_np['err_floating_base_euler'] = \
            concat((self.her_his_obs_list_cache_np['err_floating_base_euler'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, 3))), axis=1)

        self.her_his_obs_list_cache_np['dis_pos'] = \
            concat((self.her_his_obs_list_cache_np['dis_pos'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa))), axis=1)
        self.her_his_obs_list_cache_np['dis_ori'] = \
            concat((self.her_his_obs_list_cache_np['dis_ori'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa))), axis=1)
        self.her_his_obs_list_cache_np['dis_floating_base_ori'] = \
            concat((self.her_his_obs_list_cache_np['dis_floating_base_ori'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa))), axis=1)

        self.her_his_obs_list_cache_np['state'] = \
            concat((self.her_his_obs_list_cache_np['state'][:, -self.noa:], np.zeros((self.max_episode_steps+1, self.noa, self.state_dim))), axis=1)

        ''' FROM STEP '''
        self.her_his_obs_list_cache_np['action'] = \
            concat((self.her_his_obs_list_cache_np['action'][:, -self.noa:], np.zeros((self.max_episode_steps, self.noa, self.action_dim))), axis=1)

        self.r_cahce_np = self.r_cahce_np[self.noa:]
        self.theta_cahce_np = self.theta_cahce_np[self.noa:]
        self.phi_cache_np = self.phi_cache_np[self.noa:]

        self.her_his_obs_list_cache_np_len      -= self.noa

    def reset(self, target_use_random_init=False):
        # history observation
        self.his_obs_np = {}
        self._history_observation_reset_np(self.his_obs_np)

        self.step_count = 0
        self.episode_num += 1

        self.pos_reach_np = np.empty((0, self.noa)).astype(int)
        self.ori_reach_np = np.empty((0, self.noa)).astype(int)

        target_pose_init_finish_np = np.zeros(self.noa)
        floating_base_pose_init_finish_np = np.zeros(self.noa)
        agent_init_finish_np = np.zeros(self.noa)

        self.action_np = np.zeros([self.noa, self.action_dim])
        self.last_action_np = copy.copy(self.action_np)

        reset_finish = False
        break_times = 0

        while not reset_finish:
            target_init_times, floating_base_init_times, agent_init_times = 0, 0, 0
            if self.target_reinit_pos_freq != 0 and self.episode_num % self.target_reinit_pos_freq == 0:
                self._target_reinit(target_use_random_init)
                self.start_target_pos_np += self.ur5_start_pos_np

            while True:
                self._set_floating_base_np()
                self._set_agent_np()
                self._set_target_np()

                for _ in range(5):
                    self.pybullet_step(2)
                    target_pose_init_finish_np = self._check_target_init_np()
                    floating_base_pose_init_finish_np = self._check_floating_base_init_np()
                    agent_init_finish_np = self._check_agent_init_np()
                    if all(target_pose_init_finish_np) and all(floating_base_pose_init_finish_np) and all(agent_init_finish_np):
                        break

                if all(target_pose_init_finish_np) and all(floating_base_pose_init_finish_np) and all(agent_init_finish_np):
                    break
                else:
                    if not all(target_pose_init_finish_np):
                        target_init_times += 1
                        if target_init_times > 10:
                            print('\033[31mTarget pose init may failure!\033[0m')
                            target_use_random_init = True

                    if not all(floating_base_pose_init_finish_np):
                        floating_base_init_times += 1
                        if floating_base_init_times > 10:
                            print('Floating Base pose init may failure!')

                    if not all(agent_init_finish_np):
                        agent_init_times += 1
                        if agent_init_times > 10:
                            print('Agent init may failure!')

                if target_init_times + floating_base_init_times + agent_init_times > 10:
                    break_times += 1
                    break

            reset_finish = self._reset_finish()
            if break_times > 10:
                print('\033[31mInit failed! Init cancelled!\033[0m')
                break

        for i in range(self.noa):
            self.init_end_effector_pos_np[i] = self.agent_list[i].get_end_effector_pos()
            self.init_agent_pos_np[i] = self.agent_list[i].get_pos()

        self._make_observation()  # Update state
        self._last_refresh_np()

        return_state_np = copy.copy(self.state_np)
        return return_state_np

    def _set_target_np(self):
        for i in range(self.noa):
            self.target_list[i].reset(self.start_target_pos_np[i], self.start_target_quat_np[i])
        # print('target position has been set to {}'.format(self.target_pos))
        # print('target quaternion has been set to {}'.format(self.start_target_quat))

    def _check_target_init_np(self):
        t = []
        for i in self.target_list:
            t.append(i.check_init())
        return t

    def _set_floating_base_np(self):
        for i in range(self.noa):
            self.floating_base_list[i].reset(self.satellite_start_pos_np[i], self.satellite_start_quat_np[i])

    def _check_floating_base_init_np(self):
        t = []
        for i in self.floating_base_list:
            t.append(i.check_init())
        return t

    def _set_agent_np(self):
        for i in self.agent_list:
            i.reset()

    def _check_agent_init_np(self):
        t = []
        for i in self.agent_list:
            t.append(i.check_init())
        return t

    def _reset_finish(self):
        for i in range(self.noa):
            distance = norm(self.target_list[i].get_pos() - self.agent_list[i].get_pos())
            if distance > (self.r_max + 0.05) or distance < self.r_min:
                return False
        return True

    def _set_max_episode_steps(self, max_episode_steps):
        self.max_episode_steps = max_episode_steps

    def _reach_check(self):
        self.pos_reach_np = concat((self.pos_reach_np, expdim(self.dis_pos_np < self.pos_err_thres, axis=0)))
        self.ori_reach_np = concat((self.ori_reach_np, expdim(self.dis_ori_np < self.ori_err_thres, axis=0)))
        self.done_np = self.pos_reach_np[-1] & self.ori_reach_np[-1]

    def step(self, action_np):
        '''
        Step the vrep simulation by one frame, make actions and observations and calculate the resulting
        rewards
        action: rotate_angles
        '''
        self.step_count += 1
        self.action_np = action_np

        self._make_action()
        self._save_history_action(self.his_obs_np)

        self.pybullet_step()
        self._make_observation()  # Update state
        self._reach_check()
        reward_np = self._cal_reward_np()
        self._last_refresh_np()

        dw_np = copy.copy(self.done_np)
        if self.step_count == self.max_episode_steps:
            self.done_np = np.ones(self.noa).astype(int)

        self.last_action_np = copy.copy(self.action_np)

        return_state_np = copy.copy(self.state_np)
        return_done_np = copy.copy(self.done_np)
        return return_state_np, reward_np, return_done_np, dw_np

    def _make_observation(self):
        '''
        Make an observation: measure the position and orientation of NAO,
        and get current joint angles from motionProxy.
        '''
        for i in range(self.noa):
            ######### get floating_base pos/quat/euler #########
            self.floating_base_pos_np[i]           = self.floating_base_list[i].get_pos()
            self.floating_base_quat_np[i]          = self.floating_base_list[i].get_quat()
            self.floating_base_euler_np[i]         = self.floating_base_list[i].get_euler()
            ######### get agent pos/quat/euler #########
            self.agent_pos_np[i]   = self.agent_list[i].get_pos()
            self.agent_quat_np[i]  = self.agent_list[i].get_quat()
            self.agent_euler_np[i] = self.agent_list[i].get_euler()
            ######### get joint pos/vel #########
            self.joint_pos_np[i]   = self.agent_list[i].get_joint_pos()
            self.joint_vel_np[i]   = self.agent_list[i].get_joint_vel()
            ######### get target pos/quat/euler #########
            self.target_pos_np[i]      = self.target_list[i].get_pos()
            self.target_quat_np[i]     = self.target_list[i].get_quat()
            self.target_euler_np[i]    = self.target_list[i].get_euler()
            ######### get end_effector pos/quat/euler #########
            self.end_effector_pos_np[i]            = self.agent_list[i].get_end_effector_pos()
            self.end_effector_quat_np[i]           = self.agent_list[i].get_end_effector_quat()
            self.end_effector_euler_np[i]          = self.agent_list[i].get_end_effector_euler()

        # move to Zero point
        self.floating_base_pos_np  -= self.init_floating_base_pos_np
        self.agent_pos_np          -= self.init_floating_base_pos_np
        self.target_pos_np         -= self.init_floating_base_pos_np
        self.end_effector_pos_np   -= self.init_floating_base_pos_np

        if self.arm_angle_0_to_2pi:
            self.joint_pos_np %= _2pi

        self._get_state()
        self._save_history_observation_from_make_observation_np(self.his_obs_np)

    def _get_state(self):
        # end_effector, target and error position
        end_effector_pos_np     = self.end_effector_pos_np - self.pos_zero_origin_np
        target_pos_np           = self.target_pos_np - self.pos_zero_origin_np
        self.err_pos_np         = target_pos_np - end_effector_pos_np
        # end_effector, target and error quaternion
        end_effector_quat_np    = self.end_effector_quat_np
        target_quat_np          = self.target_quat_np
        self.err_quat_np        = q2q(end_effector_quat_np, target_quat_np)
        # end_effector, target and error euler
        end_effector_euler_np   = self.end_effector_euler_np
        target_euler_np         = self.target_euler_np
        self.err_euler_np       = R.from_quat(self.err_quat_np).as_euler('xyz')

        # floating_base, target and error quaternion
        floating_base_quat_np           = self.floating_base_quat_np
        floating_base_target_quat_np    = self.target_floating_base_quat_np
        self.err_floating_base_quat_np  = q2q(floating_base_quat_np, floating_base_target_quat_np)
        # floating_base, target and error euler
        self.err_floating_base_euler_np = R.from_quat(self.err_floating_base_quat_np).as_euler('xyz')

        # end_effector position and orientation distance; floating_base orientation distance
        self.dis_pos_np         = norm(self.err_pos_np, axis=1)
        self.dis_ori_np         = R.from_quat(self.err_quat_np).magnitude()
        self.dis_floating_base_ori_np = R.from_quat(self.err_floating_base_quat_np).magnitude()

        joint_pos_np            = self.joint_pos_np
        joint_vel_np            = self.joint_vel_np

        if self.state_dim_choose == 'eterrcprdpopv':
            # state:[End{x,y,z}, Tar{x,y,z}, Err{x,y,z},
            # End{6D_cont_repre}, Tar{6D_cont_repre}, Err{6D_cont_repre},
            # Dis_pos, Dis_ori,
            # Joint_Pos[6], Joint_Vel[6]]
            end_effector_cr_np = quat_to_cont_repre_np(end_effector_quat_np)
            target_cr_np = quat_to_cont_repre_np(target_quat_np)
            err_cr_np = quat_to_cont_repre_np(self.err_quat_np)
            self.state_np = concat((end_effector_pos_np, target_pos_np, self.err_pos_np,
                                    end_effector_cr_np, target_cr_np, err_cr_np,
                                    self.dis_pos_np.reshape(self.noa, -1), self.dis_ori_np.reshape(self.noa, -1),
                                    joint_pos_np, joint_vel_np), axis=1)
        elif self.state_dim_choose == 'eterr3p3edpopv':
            # state:[End{x,y,z}, Tar{x,y,z}, Err{x,y,z},
            # End{x,y,z}, Tar{x,y,z}, Err{x,y,z},
            # Dis_pos, Dis_ori,
            # Joint_Pos[6], Joint_Vel[6]]
            self.state_np = concat((end_effector_pos_np, target_pos_np, self.err_pos_np,
                                    end_effector_euler_np, target_euler_np, self.err_euler_np,
                                    self.dis_pos_np.reshape(self.noa, -1), self.dis_ori_np.reshape(self.noa, -1),
                                    joint_pos_np, joint_vel_np), axis=1)
        elif self.state_dim_choose == 'eterr3p4qdpopv':
            # state:[End{x,y,z}, Tar{x,y,z}, Err{x,y,z},
            # End{x,y,z,w}, Tar{x,y,z,w}, Err{x,y,z,w},
            # Dis_pos, Dis_ori,
            # Joint_Pos[6], Joint_Vel[6]]
            self.state_np = concat((end_effector_pos_np, target_pos_np, self.err_pos_np,
                                    end_effector_quat_np, target_quat_np, self.err_quat_np,
                                    self.dis_pos_np.reshape(self.noa, -1), self.dis_ori_np.reshape(self.noa, -1),
                                    joint_pos_np, joint_vel_np), axis=1)

        # print(self.state_list)

    def _history_observation_reset_np(self, obs):
        ''' FROM _MAKE_OBSERVATION '''
        obs['floating_base_pos']             = np.empty((0, self.noa, 3))
        obs['floating_base_quat']            = np.empty((0, self.noa, 4))
        obs['floating_base_euler']           = np.empty((0, self.noa, 3))

        obs['floating_base_target_quat']     = np.empty((0, self.noa, 4))
        obs['floating_base_target_euler']    = np.empty((0, self.noa, 3))

        obs['agent_pos']                     = np.empty((0, self.noa, 3))
        obs['agent_quat']                    = np.empty((0, self.noa, 4))
        obs['agent_euler']                   = np.empty((0, self.noa, 3))

        obs['joint_pos']                     = np.empty((0, self.noa, self.action_dim))
        obs['joint_vel']                     = np.empty((0, self.noa, self.action_dim))

        obs['target_pos']                    = np.empty((0, self.noa, 3))
        obs['target_quat']                   = np.empty((0, self.noa, 4))
        obs['target_euler']                  = np.empty((0, self.noa, 3))

        obs['end_effector_pos']              = np.empty((0, self.noa, 3))
        obs['end_effector_quat']             = np.empty((0, self.noa, 4))
        obs['end_effector_euler']            = np.empty((0, self.noa, 3))

        ''' FROM _GET_STATE '''
        obs['err_pos']                       = np.empty((0, self.noa, 3))
        obs['err_quat']                      = np.empty((0, self.noa, 4))
        obs['err_euler']                     = np.empty((0, self.noa, 3))

        obs['err_floating_base_quat']        = np.empty((0, self.noa, 4))
        obs['err_floating_base_euler']       = np.empty((0, self.noa, 3))

        obs['dis_pos']                       = np.empty((0, self.noa))
        obs['dis_ori']                       = np.empty((0, self.noa))
        obs['dis_floating_base_ori']         = np.empty((0, self.noa))

        obs['state']                         = np.empty((0, self.noa, self.state_dim))

        ''' FROM STEP '''
        obs['action']                    = np.empty((0, self.noa, self.action_dim))

    def _history_observation_reset_zero_np(self, obs, cache_size):
        ''' FROM _MAKE_OBSERVATION '''
        obs['floating_base_pos']            = np.zeros((self.max_episode_steps+1, cache_size, 3))
        obs['floating_base_quat']           = np.zeros((self.max_episode_steps+1, cache_size, 4))
        obs['floating_base_euler']          = np.zeros((self.max_episode_steps+1, cache_size, 3))

        obs['floating_base_target_quat']    = np.zeros((self.max_episode_steps+1, cache_size, 4))
        obs['floating_base_target_euler']   = np.zeros((self.max_episode_steps+1, cache_size, 3))

        obs['agent_pos']                    = np.zeros((self.max_episode_steps+1, cache_size, 3))
        obs['agent_quat']                   = np.zeros((self.max_episode_steps+1, cache_size, 4))
        obs['agent_euler']                  = np.zeros((self.max_episode_steps+1, cache_size, 3))

        obs['joint_pos']                    = np.zeros((self.max_episode_steps+1, cache_size, self.action_dim))
        obs['joint_vel']                    = np.zeros((self.max_episode_steps+1, cache_size, self.action_dim))

        obs['target_pos']                   = np.zeros((self.max_episode_steps+1, cache_size, 3))
        obs['target_quat']                  = np.zeros((self.max_episode_steps+1, cache_size, 4))
        obs['target_euler']                 = np.zeros((self.max_episode_steps+1, cache_size, 3))

        obs['end_effector_pos']             = np.zeros((self.max_episode_steps+1, cache_size, 3))
        obs['end_effector_quat']            = np.zeros((self.max_episode_steps+1, cache_size, 4))
        obs['end_effector_euler']           = np.zeros((self.max_episode_steps+1, cache_size, 3))

        ''' FROM _GET_STATE '''
        obs['err_pos']                      = np.zeros((self.max_episode_steps+1, cache_size, 3))
        obs['err_quat']                     = np.zeros((self.max_episode_steps+1, cache_size, 4))
        obs['err_euler']                    = np.zeros((self.max_episode_steps+1, cache_size, 3))

        obs['err_floating_base_quat']       = np.zeros((self.max_episode_steps+1, cache_size, 4))
        obs['err_floating_base_euler']      = np.zeros((self.max_episode_steps+1, cache_size, 3))

        obs['dis_pos']                      = np.zeros((self.max_episode_steps+1, cache_size))
        obs['dis_ori']                      = np.zeros((self.max_episode_steps+1, cache_size))
        obs['dis_floating_base_ori']        = np.zeros((self.max_episode_steps+1, cache_size))

        obs['state']                        = np.zeros((self.max_episode_steps+1, cache_size, self.state_dim))

        ''' FROM STEP '''
        obs['action']                       = np.zeros((self.max_episode_steps, cache_size, self.action_dim))

    def _save_history_observation_from_make_observation_np(self, obs):
        ''' FROM _MAKE_OBSERVATION '''
        obs['floating_base_pos']            = concat((obs['floating_base_pos'], expdim(self.floating_base_pos_np, axis=0)))
        obs['floating_base_quat']           = concat((obs['floating_base_quat'], expdim(self.floating_base_quat_np, axis=0)))
        obs['floating_base_euler']          = concat((obs['floating_base_euler'], expdim(self.floating_base_euler_np, axis=0)))

        obs['floating_base_target_quat']    = concat((obs['floating_base_target_quat'], expdim(self.target_floating_base_quat_np, axis=0)))
        obs['floating_base_target_euler']   = concat((obs['floating_base_target_euler'], expdim(self.target_floating_base_euler_np, axis=0)))

        obs['agent_pos']                    = concat((obs['agent_pos'], expdim(self.agent_pos_np, axis=0)))
        obs['agent_quat']                   = concat((obs['agent_quat'], expdim(self.agent_quat_np, axis=0)))
        obs['agent_euler']                  = concat((obs['agent_euler'], expdim(self.agent_euler_np, axis=0)))

        obs['joint_pos']                    = concat((obs['joint_pos'], expdim(self.joint_pos_np, axis=0)))
        obs['joint_vel']                    = concat((obs['joint_vel'], expdim(self.joint_vel_np, axis=0)))

        obs['target_pos']                   = concat((obs['target_pos'], expdim(self.target_pos_np, axis=0)))
        obs['target_quat']                  = concat((obs['target_quat'], expdim(self.target_quat_np, axis=0)))
        obs['target_euler']                 = concat((obs['target_euler'], expdim(self.target_euler_np, axis=0)))

        obs['end_effector_pos']             = concat((obs['end_effector_pos'], expdim(self.end_effector_pos_np, axis=0)))
        obs['end_effector_quat']            = concat((obs['end_effector_quat'], expdim(self.end_effector_quat_np, axis=0)))
        obs['end_effector_euler']           = concat((obs['end_effector_euler'], expdim(self.end_effector_euler_np, axis=0)))

        ''' FROM _GET_STATE '''
        obs['err_pos']                      = concat((obs['err_pos'], expdim(self.err_pos_np, axis=0)))
        obs['err_quat']                     = concat((obs['err_quat'], expdim(self.err_quat_np, axis=0)))
        obs['err_euler']                    = concat((obs['err_euler'], expdim(self.err_euler_np, axis=0)))

        obs['err_floating_base_quat']       = concat((obs['err_floating_base_quat'], expdim(self.err_floating_base_quat_np, axis=0)))
        obs['err_floating_base_euler']      = concat((obs['err_floating_base_euler'], expdim(self.err_floating_base_euler_np, axis=0)))

        obs['dis_pos']                      = concat((obs['dis_pos'], expdim(self.dis_pos_np, axis=0)))
        obs['dis_ori']                      = concat((obs['dis_ori'], expdim(self.dis_ori_np, axis=0)))
        obs['dis_floating_base_ori']        = concat((obs['dis_floating_base_ori'], expdim(self.dis_floating_base_ori_np, axis=0)))

        obs['state']                        = concat((obs['state'], expdim(self.state_np, axis=0)))

    def _save_history_action(self, obs):
        obs['action']                       = concat((obs['action'], expdim(self.action_np, axis=0)))

    def _make_action(self):
        '''
        Perform an action - move each joint by a specific amount
        '''
        # Update velocities
        for i in range(self.noa):
            self.agent_list[i].make_action(self.action_np[i])

    def _cal_reward_np(self):
        reward_np = np.zeros(self.noa)
        pos_error_penalty_np = np.zeros(self.noa)
        ori_error_penalty_np = np.zeros(self.noa)
        ori_decrease_reward_np = np.zeros(self.noa)

        smooth_penalty_np = np.zeros(self.noa)
        done_reward_np = np.zeros(self.noa)

        pos_error_np = -self.dis_pos_np
        ori_error_np = -self.dis_ori_np

        if self.use_pos_err_penalty:
            pos_error_penalty_np = pos_error_np * (1 - self.ori_penalty_rate)
            reward_np += pos_error_penalty_np

        if self.use_ori_err_penalty:
            ori_error_penalty_np = ori_error_np * self.ori_inherent_rate * self.ori_penalty_rate
            reward_np += ori_error_penalty_np

        if self.use_ori_decrease_reward:
            ori_decrease_reward_np += self.last_dis_ori_np - self.dis_ori_np
            ori_decrease_reward_np *= self.ori_decrease_reward_rate
            reward_np += ori_decrease_reward_np

        if self.use_smooth_penalty:
            target_current_delta_np = np.sum(
                np.max([np.abs(self.action_np - self.joint_vel_np) - 0.5,
                        np.zeros_like(self.joint_vel_np)],
                    axis=0
                ),
                axis=1
            )
            smooth_penalty_np -= 0.15 * target_current_delta_np
            reward_np += smooth_penalty_np

        if self.use_done_reward:
            if self.reward_type == 'sparse':
                reward_np = self.done_np
            else:
                done_reward_np += (self.pos_err_thres + pos_error_np) * self.pos_reach_np[-1] / self.pos_err_thres
                done_reward_np += (self.ori_err_thres + ori_error_np) * self.ori_reach_np[-1] / self.ori_err_thres
                done_reward_np *= self.done_reward_rate
                reward_np += done_reward_np


        return reward_np

    def _last_refresh_np(self):
        self.last_state_np = self.state_np

        self.last_err_pos_np = self.err_pos_np
        self.last_err_quat_np = self.err_quat_np
        self.last_err_euler_np = self.err_euler_np
        self.last_err_floating_base_quat_np = self.err_floating_base_quat_np
        self.last_err_floating_base_euler_np = self.err_floating_base_euler_np

        self.last_dis_pos_np = self.dis_pos_np
        self.last_dis_ori_np = self.dis_ori_np
        self.last_dis_floating_base_ori_np = self.dis_floating_base_ori_np

    def her_enable_check_np(self):
        target_pos_np = self.his_obs_np['end_effector_pos'][-1] - self.his_obs_np['agent_pos'][0]
        r_np = np.sqrt(target_pos_np[:, 0] ** 2 + target_pos_np[:, 1] ** 2 + (target_pos_np[:, 2] - 0.05) ** 2)
        theta_np = np.arccos((target_pos_np[:, 2] - 0.05) / r_np)
        phi_np = np.arctan2(target_pos_np[:, 1], target_pos_np[:, 0]) % _2pi
        self.r_np_ = np.floor(((r_np - self.r_min) / (self.r_max - self.r_min)) * self.M).astype(int)
        self.theta_np_ = np.floor((theta_np / (pi / 2)) * self.N).astype(int)
        self.phi_np_ = np.floor((phi_np / _2pi) * self.Q).astype(int)

        for i in range(self.noa):
            if 0 <= self.r_np_[i] < self.M and 0 <= self.theta_np_[i] < self.N and 0 <= self.phi_np_[i] < self.Q:
                self._her_copy_observation_np(i)

        return self.her_his_obs_list_cache_np_len >= self.noa

    def _her_copy_observation_np(self, i):
        self._her_obs_copy(i)
        self.r_cahce_np = np.hstack((self.r_cahce_np, self.r_np_[i]))
        self.theta_cahce_np = np.hstack((self.theta_cahce_np, self.theta_np_[i]))
        self.phi_cache_np = np.hstack((self.phi_cache_np, self.phi_np_[i]))

    def _her_obs_copy(self, i):
        length = self.her_his_obs_list_cache_np_len
        ''' FROM _MAKE_OBSERVATION '''
        self.her_his_obs_list_cache_np['floating_base_pos'][:, length]          = self.his_obs_np['floating_base_pos'][:, i]
        self.her_his_obs_list_cache_np['floating_base_quat'][:, length]         = self.his_obs_np['floating_base_quat'][:, i]
        self.her_his_obs_list_cache_np['floating_base_euler'][:, length]        = self.his_obs_np['floating_base_euler'][:, i]

        self.her_his_obs_list_cache_np['floating_base_target_quat'][:, length]  = self.his_obs_np['floating_base_target_quat'][:, i]
        self.her_his_obs_list_cache_np['floating_base_target_euler'][:, length] = self.his_obs_np['floating_base_target_euler'][:, i]

        self.her_his_obs_list_cache_np['agent_pos'][:, length]                  = self.his_obs_np['agent_pos'][:, i]
        self.her_his_obs_list_cache_np['agent_quat'][:, length]                 = self.his_obs_np['agent_quat'][:, i]
        self.her_his_obs_list_cache_np['agent_euler'][:, length]                = self.his_obs_np['agent_euler'][:, i]

        self.her_his_obs_list_cache_np['joint_pos'][:, length]                  = self.his_obs_np['joint_pos'][:, i]
        self.her_his_obs_list_cache_np['joint_vel'][:, length]                  = self.his_obs_np['joint_vel'][:, i]

        self.her_his_obs_list_cache_np['target_pos'][:, length]                 = self.his_obs_np['target_pos'][:, i]
        self.her_his_obs_list_cache_np['target_quat'][:, length]                = self.his_obs_np['target_quat'][:, i]
        self.her_his_obs_list_cache_np['target_euler'][:, length]               = self.his_obs_np['target_euler'][:, i]

        self.her_his_obs_list_cache_np['end_effector_pos'][:, length]           = self.his_obs_np['end_effector_pos'][:, i]
        self.her_his_obs_list_cache_np['end_effector_quat'][:, length]          = self.his_obs_np['end_effector_quat'][:, i]
        self.her_his_obs_list_cache_np['end_effector_euler'][:, length]         = self.his_obs_np['end_effector_euler'][:, i]

        ''' FROM _GET_STATE '''
        self.her_his_obs_list_cache_np['err_pos'][:, length]                    = self.his_obs_np['err_pos'][:, i]
        self.her_his_obs_list_cache_np['err_quat'][:, length]                   = self.his_obs_np['err_quat'][:, i]
        self.her_his_obs_list_cache_np['err_euler'][:, length]                  = self.his_obs_np['err_euler'][:, i]

        self.her_his_obs_list_cache_np['err_floating_base_quat'][:, length]     = self.his_obs_np['err_floating_base_quat'][:, i]
        self.her_his_obs_list_cache_np['err_floating_base_euler'][:, length]    = self.his_obs_np['err_floating_base_euler'][:, i]

        self.her_his_obs_list_cache_np['dis_pos'][:, length]                    = self.his_obs_np['dis_pos'][:, i]
        self.her_his_obs_list_cache_np['dis_ori'][:, length]                    = self.his_obs_np['dis_ori'][:, i]
        self.her_his_obs_list_cache_np['dis_floating_base_ori'][:, length]      = self.his_obs_np['dis_floating_base_ori'][:, i]

        self.her_his_obs_list_cache_np['state'][:, length]                      = self.his_obs_np['state'][:, i]

        ''' FROM STEP '''
        self.her_his_obs_list_cache_np['action'][:, length]                     = self.his_obs_np['action'][:, i]
        self.her_his_obs_list_cache_np_len += 1

    def her_reset(self):
        # history observation
        self.her_his_obs_np = {}
        self._history_observation_reset_np(self.her_his_obs_np)

        self.step_count = 0
        self.episode_num += 1

        self.pos_reach_np = np.empty((0, self.noa)).astype(int)
        self.ori_reach_np = np.empty((0, self.noa)).astype(int)

        self.action_np = np.zeros([self.noa, self.action_dim])

        self.last_action_np = copy.copy(self.action_np)

        self.r_np_ = self.r_cahce_np[:self.noa]
        self.theta_np_ = self.theta_cahce_np[:self.noa]
        self.phi_np_ = self.phi_cache_np[:self.noa]

        self._her_make_observation_np(self.her_his_obs_list_cache_np, self.her_his_obs_np)  # Update state
        self._last_refresh_np()

        return_state_np = copy.copy(self.state_np)
        return return_state_np

    def _her_make_observation_np(self, obs1, obs2):
        '''
        Make an observation: measure the position and orientation of NAO,
        and get current joint angles from motionProxy.
        '''
        ######### get floating_base pos/quat/euler #########
        self.floating_base_pos_np          = copy.copy(obs1['floating_base_pos'][self.step_count, :self.noa])
        self.floating_base_quat_np         = copy.copy(obs1['floating_base_quat'][self.step_count, :self.noa])
        self.floating_base_euler_np        = copy.copy(obs1['floating_base_euler'][self.step_count, :self.noa])
        ######### get agent pos/quat #########
        self.agent_pos_np                  = copy.copy(obs1['agent_pos'][self.step_count, :self.noa])
        self.agent_quat_np                 = copy.copy(obs1['agent_quat'][self.step_count, :self.noa])
        self.agent_euler_np                = copy.copy(obs1['agent_euler'][self.step_count, :self.noa])
        ######### get joint pos/vel #########
        self.joint_pos_np                   = copy.copy(obs1['joint_pos'][self.step_count, :self.noa])
        self.joint_vel_np                   = copy.copy(obs1['joint_vel'][self.step_count, :self.noa])
        ######### get target pos/quat/euler #########
        self.target_pos_np                  = copy.copy(obs1['end_effector_pos'][-1, :self.noa])
        self.target_quat_np                 = copy.copy(obs1['end_effector_quat'][-1, :self.noa])
        self.target_euler_np                = copy.copy(obs1['end_effector_euler'][-1, :self.noa])
        ######### get end_effector pos/quat/euler #########
        self.end_effector_pos_np            = copy.copy(obs1['end_effector_pos'][self.step_count, :self.noa])
        self.end_effector_quat_np           = copy.copy(obs1['end_effector_quat'][self.step_count, :self.noa])
        self.end_effector_euler_np          = copy.copy(obs1['end_effector_euler'][self.step_count, :self.noa])

        self._get_state()
        self._save_history_observation_from_make_observation_np(obs2)

    def her_step(self, action_np):
        '''
        Step the vrep simulation by one frame, make actions and observations and calculate the resulting
        rewards
        action: rotate_angles
        '''
        self.step_count += 1
        self.action_np = action_np

        self._save_history_action(self.her_his_obs_np)

        self._her_make_observation_np(self.her_his_obs_list_cache_np, self.her_his_obs_np)  # Update state
        self._reach_check()
        reward_np = self._cal_reward_np()
        self._last_refresh_np()

        dw_np = copy.copy(self.done_np)
        if self.step_count == self.max_episode_steps:
            self.done_np = np.ones(self.noa).astype(int)

        self.last_action_np = copy.copy(self.action_np)

        return_state_np = copy.copy(self.state_np)
        return_done_np = copy.copy(self.done_np)
        return return_state_np, reward_np, return_done_np, dw_np

    def _target_reinit(self, pos_random_reinit=False):
        if not pos_random_reinit:
            if self.target_pos_reinit_method == 'random':
                self.start_target_pos_np = self._target_sphere_random_reinit_np()
            elif self.target_pos_reinit_method == 'sequence':
                self.start_target_pos_np = self._target_sphere_sequence_reinit_np()
        else:
            self.start_target_pos_np = self._target_sphere_random_reinit()

        self.start_target_quat_np = self._target_quaternion_reinit_np()

    def _target_sphere_random_reinit_np(self):
        # 将整个球坐标系分为 (M*N*Q) 份
        i_np = np.ones(self.noa) * -1
        while 1:
            tmp_np = np.random.rand(self.noa)
            for i in range(self.MNQ):
                tmp_np -= self.norm_reach_fail_prob[i]
                if any((i_np == -1) & (tmp_np < 0)):
                    i_np += (i_np == -1) * ((tmp_np < 0) * (i + 1))
                    if all(i_np >= 0):
                        break
            self.r_np_ = (i_np / self.N / self.Q).astype(int)
            self.theta_np_ = (i_np / self.Q % self.N).astype(int)
            self.phi_np_ = (i_np % self.Q).astype(int)

            r1_np, r2_np, r3_np = np.random.rand(3, self.noa)
            r_np = (r1_np + self.r_np_) * self.d_r + self.r_min  # 生成在一定范围内的距离 r
            theta_np = (r2_np + self.theta_np_) * self.d_theta
            phi_np = (r3_np + self.phi_np_) * self.d_phi
            y_np_ = np.array([r_np * np.sin(theta_np) * np.cos(phi_np), r_np * np.sin(theta_np) * np.sin(phi_np),
                              r_np * np.cos(theta_np) + 0.05]).T
            t = norm(y_np_, axis=1)
            if (self.r_min < t.min()) and (t.max() < (self.r_max + 0.05)):
                break

        return y_np_

    def _target_sphere_sequence_reinit_np(self):
        # 将整个球坐标系分为 (M*N*Q) 份
        # r: 0.25 ~ 0.65; theta: 0 ~ 90, phi: 0 ~ 360
        while 1:
            # d_r, d_theta, d_phi = (self.r_max - self.r_min) / self.M, pi / 2 / self.N, _2pi / self.Q
            self.r_np_ = (np.ones(self.noa) * (self.episode_num % self.MNQ / self.N / self.Q)).astype(int)
            self.theta_np_ = (np.ones(self.noa) * (self.episode_num % self.MNQ / self.Q % self.N)).astype(int)
            self.phi_np_ = (np.ones(self.noa) * (self.episode_num % self.MNQ % self.Q)).astype(int)
            r_np = (self.r_np_) * self.d_r + self.r_min  # 生成在一定范围内的距离 r
            theta_np = (self.theta_) * self.d_theta
            phi_np = (self.phi_) * self.d_phi
            y_np_ = np.array([r_np * np.sin(theta_np) * np.cos(phi_np), r_np * np.sin(theta_np) * np.sin(phi_np), r_np * np.cos(theta_np) + 0.05]).T
            t = norm(y_np_, axis=1)
            if all(self.r_min < t) and all(t < (self.r_max + 0.05)):
                break

        return y_np_

    def _target_quaternion_reinit_np(self):
        return R.random(self.noa).as_quat()

    def reach_fail_prob_maintain(self):
        if self.sphere_reach_fail_prob.max() < self.decrease_threshold:
            self._sphere_reach_fail_prob_reset()
            self.pos_err_stable_times += self.pos_err_stable_times_increase
            self.pos_err_stable_times = min(self.pos_err_stable_times, self.pos_err_stable_times_limit)
            self.pos_err_thres *= self.pos_err_thres_idx
            self.pos_err_thres = max(self.pos_err_thres, self.pos_err_thres_limit)
            self.ori_err_thres *= self.ori_err_thres_idx
            self.ori_err_thres = max(self.ori_err_thres, self.ori_err_thres_limit)

            if self.ori_err_thres == self.ori_err_thres_limit:
                self.target_pos_reinit_method = 'random'

            with open(os.path.join(self.maintain_save_dir, 'reach_fail_prob_maintain.txt'.format(self.episode_num)), 'a+') as f:
                f.write('In episode {}, total step = {}:\n'.format(
                    self.episode_num, (self.episode_num * self.max_episode_steps + self.step_count)))
                f.write('pos_err_thres = {}, ori_err_thres = {};\n'.format(
                    self.pos_err_thres, self.ori_err_thres))
                f.write('error_stable_times = {}, ori_penalty_rate = {};\n\n'.format(
                    self.pos_err_stable_times, self.ori_penalty_rate))

    def _norm_reach_fail_prob_maintain(self):
        self.norm_reach_fail_prob = (self.sphere_reach_fail_prob / np.sum(self.sphere_reach_fail_prob)).reshape(-1)
        self.norm_reach_fail_prob[-1] += 1

    def _sphere_reach_fail_prob_reset(self):
        self.sphere_reach_fail_prob = np.ones([self.M, self.N, self.Q])
        self.sphere_reach_fail_prob[2, 0, 2:7] = self.decrease_threshold * self.success_sample_rate * 0.1
        self.sphere_reach_fail_prob[0:2, 1, 4:6] = self.decrease_threshold * self.success_sample_rate * 0.1

        # print(self.sphere_reach_fail_prob)

    def set_episode_num(self, episode_num):
        self.episode_num = episode_num

    def set_maintain_save_dir(self, maintain_save_dir):
        self.maintain_save_dir = maintain_save_dir

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def pybullet_step(self, times=1):
        for i in range(times):
            p.stepSimulation()
            time.sleep(1. / 240000.)
