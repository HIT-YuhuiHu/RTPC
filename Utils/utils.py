import numpy as np
import matplotlib
import pandas as pd

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import glob
import tkinter as tk
from tkinter import messagebox
import shlex
import platform
import argparse
import csv
import imageio
import torch
import ast
import time
import copy
import wandb

from scipy.spatial.transform import Rotation as R

from Utils.normalization import Normalization, RewardScaling
from numpy.linalg import norm
from numpy import concatenate as concat
from numpy import expand_dims as expdim


def str2BoolNone(str):
    if str.lower() == 'true':
        return True
    elif str.lower() == 'false':
        return False
    elif str.lower() == 'none':
        return None
    else:
        return str.lower()

def float2int(f):
    return int(f)

def str2list(str):
    return ast.literal_eval(str)

def get_model_path(log_dir, model_name=None, type='latest'):
    # type: # ['latest', 'earliest']
    if model_name is None:
        model_paths = glob.glob(os.path.join(log_dir, 'model_ep_*.pth'))
        if len(model_paths) > 0:
            print('=> found {} models in {}'.format(len(model_paths), log_dir))
            created_times = [os.path.getmtime(path) for path in model_paths]
            if type == 'latest':
                latest_path = model_paths[np.argmax(created_times)]
            elif type == 'earliest':
                latest_path = model_paths[np.argmin(created_times)]
            print('=> the latest model path: {}'.format(latest_path))
            return latest_path
        else:
            raise ValueError('No pre-trained model found!')
    else:
        model_path = os.path.join(log_dir, model_name)
        if os.path.exists(model_path):
            return model_path
        else:
            raise ValueError('No pre-trained model found!')

def get_model_path_and_nums(log_dir, model_name=None, type='latest'):
    # type: # ['latest', 'earliest']
    if model_name is None:
        model_paths = glob.glob(os.path.join(log_dir, 'model_ep_*.pth'))
        if len(model_paths) > 0:
            print('=> found {} models in {}'.format(len(model_paths), log_dir))
            created_times = [os.path.getmtime(path) for path in model_paths]
            if type == 'latest':
                latest_path = model_paths[np.argmax(created_times)]
            elif type == 'earliest':
                latest_path = model_paths[np.argmin(created_times)]
            print('=> the latest model path: {}'.format(latest_path))
            return latest_path, len(model_paths)
        else:
            raise ValueError('No pre-trained model found!')
    else:
        model_path = os.path.join(log_dir, model_name)
        if os.path.exists(model_path):
            return model_path, len(model_paths)
        else:
            raise ValueError('No pre-trained model found!')

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_saving_dirs(args):
    save_dir = os.path.join('save_files', args.save_dir)
    create_dir(save_dir)
    if args.is_train:
        with open(os.path.join(save_dir, 'args.txt'), 'w+') as f:
            for arg in vars(args):
                f.write(f'{arg} {getattr(args, arg)}\n')

    log_dir = os.path.join(save_dir, 'log')
    create_dir(log_dir)

    increment_dir = os.path.join(log_dir, 'increment')
    create_dir(increment_dir)

    log_dir_ = os.path.join(log_dir, '_')
    create_dir(log_dir_)

    dirs = {}
    dirs['save'] = save_dir
    dirs['log'] = log_dir
    dirs['log_'] = log_dir_
    dirs['increment'] = increment_dir
    return dirs

def platform_check():
    platform_is_wsl = (platform.platform().find('WSL') != -1)
    platform_is_centos = (platform.platform().find('centos') != -1)
    platform_is_ubuntu = (platform.platform().find('generic') != -1)
    platform_is_windows = (platform.platform().find('Windows') != -1)
    assert (platform_is_wsl ^ platform_is_centos ^ platform_is_ubuntu ^ platform_is_windows)
    if platform_is_wsl:
        return 'wsl'
    elif platform_is_centos:
        return 'centos'
    elif platform_is_ubuntu:
        return 'ubuntu'
    elif platform_is_windows:
        return 'windows'

def check_files_with_suffix(directory, suffix):
    '''
        检查指定目录中是否有文件以特定后缀结尾。
        :param directory: 要检查的目录
        :param suffix: 文件后缀，例如 '.pth'
        :return: 如果找到至少一个文件则返回 True，否则返回 False
        '''
    # 遍历指定目录
    for item in os.listdir(directory):
        # 拼接完整的文件路径
        item_path = os.path.join(directory, item)
        # 检查是否为文件并且后缀匹配
        if os.path.isfile(item_path) and item_path.endswith(suffix):
            return True
    return False

def q2q(q1, q2):
    return (R.from_quat(q1).inv() * R.from_quat(q2)).as_quat()

def cal_quat_error(q1, q2):
    '''
    计算两个四元数的姿态误差。
    :param q1, q2: 四元数。
    :return: 四元数姿态误差。
    '''
    return (R.from_quat(q1).inv() * R.from_quat(q2)).magnitude()

def quat_to_cont_repre_np(data):
    return R.from_quat(data).as_matrix()[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)

def set_seed(seed, env):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def success_check(dw_flag_np, success_interval):
    success_np = np.zeros(dw_flag_np.shape[-1])

    for i in range(dw_flag_np.shape[-1]):
        t = success_interval
        for j in dw_flag_np[:, i]:
            t = t - 1 if j == 1 else success_interval
            if t == 0:
                success_np[i] = 1
                break

    return success_np

def train_post_process_multiagent_np(env, tracking_success_times_np,
        episode_err_pos_np, episode_dis_pos_np, episode_err_euler_np, episode_dis_ori_np,
        episode_err_floating_base_euler_np, episode_dis_floating_base_ori_np, episode_rewards_np, stable_step_num,
        stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, stable_dis_ori_np,
        stable_err_floating_base_euler_np, stable_dis_floating_base_ori_np, stable_rewards_np, args, use_wandb=True):
    if any(tracking_success_times_np):
        env._norm_reach_fail_prob_maintain()

    stable_err_pos_np = concat((stable_err_pos_np, expdim(np.mean(episode_err_pos_np[-stable_step_num:], axis=0), axis=0)), axis=0)
    stable_dis_pos_np = concat((stable_dis_pos_np, expdim(np.mean(episode_dis_pos_np[-stable_step_num:], axis=0), axis=0)), axis=0)
    stable_err_euler_np = concat((stable_err_euler_np, expdim(np.mean(episode_err_euler_np[-stable_step_num:], axis=0), axis=0)), axis=0)
    stable_dis_ori_np = concat((stable_dis_ori_np, expdim(np.mean(episode_dis_ori_np[-stable_step_num:], axis=0), axis=0)), axis=0)
    stable_err_floating_base_euler_np = concat((stable_err_floating_base_euler_np, expdim(np.mean(episode_err_floating_base_euler_np[-stable_step_num:], axis=0), axis=0)), axis=0)
    stable_dis_floating_base_ori_np = concat((stable_dis_floating_base_ori_np, expdim(np.mean(episode_dis_floating_base_ori_np[-stable_step_num:], axis=0), axis=0)), axis=0)

    stable_rewards_np = concat((stable_rewards_np, expdim(np.mean(episode_rewards_np[-stable_step_num:], axis=0), axis=0)), axis=0)

    if use_wandb:
        env.wandb.log({
            'tAPE': np.mean(stable_dis_pos_np[-int(args.sf_num / args.noa):]),
            'tAOE': np.mean(stable_dis_ori_np[-int(args.sf_num / args.noa):]),
            'tAFBOE': np.mean(stable_dis_floating_base_ori_np[-int(args.sf_num / args.noa):]),
            'tAR': np.mean(stable_rewards_np[-int(args.sf_num / args.noa):])
        })

    return stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, stable_dis_ori_np, \
           stable_err_floating_base_euler_np, stable_dis_floating_base_ori_np, stable_rewards_np

# save and reload data
def save_data_stable_increment_np(stable_rewards_np, stable_err_pos_np, stable_dis_pos_np,
                     stable_err_euler_np, stable_dis_ori_np, actor_loss_list, critic_loss_list,
                     actor_lr_list, critic_lr_list, episode_num_list, success_num_list, data_path=''):
    print('=> saving data to {} ...'.format(data_path))
    checkpoint = {'stable_rewards_np': stable_rewards_np,
                  'stable_err_pos_np': stable_err_pos_np,
                  'stable_dis_pos_np': stable_dis_pos_np,
                  'stable_err_euler_np': stable_err_euler_np,
                  'stable_dis_ori_np': stable_dis_ori_np,
                  'actor_loss_list': actor_loss_list,
                  'critic_loss_list': critic_loss_list,
                  'actor_lr_list': actor_lr_list,
                  'critic_lr_list': critic_lr_list,
                  'episode_num_list': episode_num_list,
                  'success_num_list': success_num_list,
                  }
    torch.save(checkpoint, data_path, _use_new_zipfile_serialization=False)
    print('=> data saved!')
    return

def reload_data_stable_increment_np(root_data_path=''):
    if os.path.exists(root_data_path):
        stable_rewards_np, stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, \
            stable_dis_ori_np = None, None, None, None, None
        actor_loss_list, critic_loss_list, actor_lr_list, critic_lr_list, episode_num_list, \
            success_num_list = [], [], [], [], [], []
        print('=> reloading data from {} ...'.format(root_data_path))
        data_path_list = sorted(os.listdir(root_data_path))

        for data_path in data_path_list:
            checkpoint = torch.load(os.path.join(root_data_path, data_path), map_location=lambda storage, loc: storage.cpu())

            stable_rewards_np = checkpoint['stable_rewards_np'] if stable_rewards_np is None else \
                concat((stable_rewards_np, checkpoint['stable_rewards_np']))

            stable_err_pos_np = checkpoint['stable_err_pos_np'] if stable_err_pos_np is None else \
                concat((stable_err_pos_np, checkpoint['stable_err_pos_np']))
            stable_dis_pos_np = checkpoint['stable_dis_pos_np'] if stable_dis_pos_np is None else \
                concat((stable_dis_pos_np, checkpoint['stable_dis_pos_np']))
            stable_err_euler_np = checkpoint['stable_err_euler_np'] if stable_err_euler_np is None else \
                concat((stable_err_euler_np, checkpoint['stable_err_euler_np']))
            stable_dis_ori_np = checkpoint['stable_dis_ori_np'] if stable_dis_ori_np is None else \
                concat((stable_dis_ori_np, checkpoint['stable_dis_ori_np']))

            if isinstance(checkpoint['actor_loss_list'], list):
                actor_loss_list.extend(checkpoint['actor_loss_list'])
            elif isinstance(checkpoint['actor_loss_list'], tuple):
                actor_loss_list.extend(checkpoint['actor_loss_list'][0])

            if isinstance(checkpoint['critic_loss_list'], list):
                critic_loss_list.extend(checkpoint['critic_loss_list'])
            elif isinstance(checkpoint['critic_loss_list'], tuple):
                critic_loss_list.extend(checkpoint['critic_loss_list'][0])

            if isinstance(checkpoint['actor_lr_list'], list):
                actor_lr_list.extend(checkpoint['actor_lr_list'])
            elif isinstance(checkpoint['actor_lr_list'], tuple):
                actor_lr_list.extend(checkpoint['actor_lr_list'][0])

            if isinstance(checkpoint['critic_lr_list'], list):
                critic_lr_list.extend(checkpoint['critic_lr_list'])
            elif isinstance(checkpoint['critic_lr_list'], tuple):
                critic_lr_list.extend(checkpoint['critic_lr_list'][0])

            if isinstance(checkpoint['episode_num_list'], list):
                episode_num_list.extend(checkpoint['episode_num_list'])
            elif isinstance(checkpoint['episode_num_list'], tuple):
                episode_num_list.extend(checkpoint['episode_num_list'][0])

            if isinstance(checkpoint['success_num_list'], list):
                success_num_list.extend(checkpoint['success_num_list'])
            elif isinstance(checkpoint['success_num_list'], tuple):
                success_num_list.extend(checkpoint['success_num_list'][0])

    else:
        raise ValueError('No data file is found in {}'.format(data_path))

    return stable_rewards_np, \
           stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, stable_dis_ori_np, \
           actor_loss_list, critic_loss_list, actor_lr_list, critic_lr_list, episode_num_list, success_num_list

def get_gpuidx_from_dir(args):
    '''
    :param args: args
    :return: -1 (for error) or none (change args.gpu_idx directly)
    :description:
        根据 args.save_dir 和 gpu_num 计算 gpu_idx, 直接修改 args.gpu_idx
    '''
    try:
        save_dir_num = int(os.path.basename(args.save_dir)[1:])
    except:
        return -1

    if args.gpu_num <= 0:
        return -1

    args.gpu_idx = save_dir_num % args.gpu_num
