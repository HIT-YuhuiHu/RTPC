import copy

import torch
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from torch.utils.tensorboard import SummaryWriter
import argparse
from Utils.normalization import Normalization, RewardScaling
from Utils.replaybuffer import PPO_ReplayBuffer_np
from Utils.utils import *
from Agents.PPO_continuous import PPO_continuous
import os

import sys
sys.path.append('..')

import time

def train(args, env):
    set_seed(args.seed, env)

    print('state_dim={}'.format(args.state_dim))
    print('action_dim={}'.format(args.action_dim))
    print('max_episode_steps={}'.format(args.max_episode_steps))
    print('mode:', 'train' if args.is_train else 'evaluation')

    dirs = create_saving_dirs(args)
    env.wandb.save(os.path.join(dirs['save'], 'args.txt'))
    print('Create directory successful!\nROOT FOLDER: >>>> {} <<<<'.format(args.save_dir))

    reinit_freq = args.target_reinit_pos_freq

    # model_path = None
    stable_step_num = args.stable_step_num

    agent = PPO_continuous(args)
    # print(agent.actor)

    state_norm_np = Normalization(shape=args.state_dim)  # Trick 2: state normalization
    state_norm = Normalization(shape=args.state_dim)
    reward_scaling_np = RewardScaling(shape=1, gamma=args.reward_scaling_gamma)

    actor_loss_list, critic_loss_list, actor_lr_list, critic_lr_list = [], [], [], []
    episode_num_list, success_num_list = [], []

    stable_err_pos_np = np.empty((0, args.noa, 3))
    stable_dis_pos_np = np.empty((0, args.noa))
    stable_err_euler_np = np.empty((0, args.noa, 3))
    stable_dis_ori_np = np.empty((0, args.noa))
    stable_err_floating_base_euler_np = np.empty((0, args.noa, 3))
    stable_dis_floating_base_ori_np = np.empty((0, args.noa))

    stable_rewards_np = np.empty((0, args.noa))

    total_steps = 0  # Record the total steps during the training
    episode_num = 0
    update_time = 0

    replay_buffer_np = PPO_ReplayBuffer_np(args)

    total_steps_last = total_steps
    data_length = stable_rewards_np.shape[0]

    env.set_maintain_save_dir(dirs['save'])
    update_flag = False

    t_last = time.time()
    her_episode_num, her_episode_num_last, episode_num_last = 0, 0, 0
    success_num, success_num_last = 0, 0
    while total_steps < args.max_train_steps:
        her_flag = args.use_her
        s_np = env.reset()

        tracking_success_times_np = np.zeros(args.noa)
        episode_rewards_np = np.zeros((args.max_episode_steps, args.noa))

        if args.use_state_norm:
            if args.use_one_state_norm:
                for i in range(args.noa):
                    state_norm(s_np[i])
            s_np = state_norm_np(s_np)

        episode_num += 1
        for episode_steps in range(args.max_episode_steps):
            action, a_logprob_np = agent.choose_action_np(s_np)
            s_np_, reward_np, done_np, dw_np = env.step(action)

            if args.use_state_norm:
                if args.use_one_state_norm:
                    for i in range(args.noa):
                        state_norm(s_np_[i])
                s_np_ = state_norm_np(s_np_)

            replay_buffer_np.store(s_np, action, a_logprob_np, reward_np, s_np_, dw_np, done_np)

            episode_rewards_np[episode_steps] = reward_np
            s_np = s_np_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer_np.count == args.batch_size:
                update_flag, her_flag = True, False
                env.her_cache_init()
                break

            tracking_success_times_np += dw_np

        tracking_success_np = (tracking_success_times_np > 0)

        episode_err_pos_np                  = env.his_obs_np['err_pos']
        episode_dis_pos_np                  = env.his_obs_np['dis_pos']
        episode_err_euler_np                = env.his_obs_np['err_euler']
        episode_dis_ori_np                  = env.his_obs_np['dis_ori']
        episode_err_floating_base_euler_np  = env.his_obs_np['err_floating_base_euler']
        episode_dis_floating_base_ori_np    = env.his_obs_np['dis_floating_base_ori']

        success_num += sum(tracking_success_np)

        if args.use_state_norm and args.use_one_state_norm:
            for i in range(args.noa):
                state_norm_np.running_ms.mean[i]    = state_norm.running_ms.mean
                state_norm_np.running_ms.std[i]     = state_norm.running_ms.std
                state_norm_np.running_ms.S[i]       = state_norm.running_ms.S
            state_norm_np.running_ms.n              = state_norm.running_ms.n

        stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, stable_dis_ori_np, stable_err_floating_base_euler_np,\
            stable_dis_floating_base_ori_np, stable_rewards_np = train_post_process_multiagent_np(
                env, tracking_success_times_np,
                episode_err_pos_np, episode_dis_pos_np, episode_err_euler_np, episode_dis_ori_np,
                episode_err_floating_base_euler_np, episode_dis_floating_base_ori_np, episode_rewards_np, stable_step_num,
                stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, stable_dis_ori_np,
                stable_err_floating_base_euler_np, stable_dis_floating_base_ori_np, stable_rewards_np, args)

        '''HER'''
        if her_flag:
            if env.her_enable_check_np():
                s_np = env.her_reset()

                episode_rewards_np = np.zeros((args.max_episode_steps, args.noa))
                tracking_success_times_np = np.zeros(args.noa)

                if args.use_state_norm:
                    if args.use_one_state_norm:
                        for i in range(args.noa):
                            state_norm(s_np[i])
                    s_np = state_norm_np(s_np)

                episode_num += 1
                her_episode_num += 1
                for episode_steps in range(args.max_episode_steps):
                    action = env.her_his_obs_list_cache_np['action'][episode_steps, :args.noa]
                    a_logprob_np = agent.get_logprob(s_np, action)
                    s_np_, reward_np, done_np, dw_np = env.her_step(action)

                    if args.use_state_norm:
                        if args.use_one_state_norm:
                            for i in range(args.noa):
                                state_norm(s_np_[i])
                        s_np_ = state_norm_np(s_np_)

                    replay_buffer_np.store(s_np, action, a_logprob_np, reward_np, s_np_, dw_np, done_np)

                    episode_rewards_np[episode_steps] = reward_np
                    s_np = s_np_
                    total_steps += 1

                    # When the number of transitions in buffer reaches batch_size,then update
                    if replay_buffer_np.count == args.batch_size:
                        update_flag, her_flag = True, False
                        env.her_cache_init()
                        break

                    tracking_success_times_np += dw_np

                tracking_success_np = (tracking_success_times_np > 0)

                episode_err_pos_np                  = env.her_his_obs_np['err_pos']
                episode_dis_pos_np                  = env.her_his_obs_np['dis_pos']
                episode_err_euler_np                = env.her_his_obs_np['err_euler']
                episode_dis_ori_np                  = env.her_his_obs_np['dis_ori']
                episode_err_floating_base_euler_np  = env.her_his_obs_np['err_floating_base_euler']
                episode_dis_floating_base_ori_np    = env.her_his_obs_np['dis_floating_base_ori']

                env.her_cache_post_process()

                if args.use_state_norm and args.use_one_state_norm:
                    for i in range(args.noa):
                        state_norm_np.running_ms.mean[i]    = state_norm.running_ms.mean
                        state_norm_np.running_ms.std[i]     = state_norm.running_ms.std
                        state_norm_np.running_ms.S[i]       = state_norm.running_ms.S
                    state_norm_np.running_ms.n = state_norm.running_ms.n

                stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, stable_dis_ori_np, stable_err_floating_base_euler_np, \
                    stable_dis_floating_base_ori_np, stable_rewards_np = train_post_process_multiagent_np(
                        env, tracking_success_times_np,
                        episode_err_pos_np, episode_dis_pos_np, episode_err_euler_np, episode_dis_ori_np,
                        episode_err_floating_base_euler_np, episode_dis_floating_base_ori_np, episode_rewards_np, stable_step_num,
                        stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, stable_dis_ori_np,
                        stable_err_floating_base_euler_np, stable_dis_floating_base_ori_np, stable_rewards_np, args)

        if update_flag:
            update_flag = False
            env.her_his_obs_list_cache = []
            reward_scaling_np = agent.update_np(replay_buffer_np, total_steps, update_time, reward_scaling_np)
            actor_loss_list.append(agent.actor_loss_save)
            critic_loss_list.append(agent.critic_loss_save)
            actor_lr_list.append(agent.optimizer_actor.param_groups[0]['lr'])
            critic_lr_list.append(agent.optimizer_critic.param_groups[0]['lr'])
            episode_num_list.append(args.noa*((episode_num-episode_num_last)-(her_episode_num-her_episode_num_last)))
            success_num_list.append(success_num - success_num_last)
            episode_num_last, her_episode_num_last, success_num_last = episode_num, her_episode_num, success_num
            replay_buffer_np.reset()
            update_time += 1
            env.reach_fail_prob_maintain()
            if update_time > args.her_update_times:
                args.use_her = False
                if args.after_her_reset_reward_scaling:
                    reward_scaling_np = RewardScaling(shape=1, gamma=args.reward_scaling_gamma)   # Trick 11.6.2: after her, reset reward_scaling

            if update_time % args.model_save_freq == 0:
                model_path = os.path.join(dirs['log'], 'model_ep_{:03d}.pth'.format(update_time))

                if args.use_one_state_norm:
                    agent.save_model(state_norm=state_norm, model_path=model_path,)
                else:
                    agent.save_model(state_norm=state_norm_np, model_path=model_path,)

                if args.data_save:
                    increment_data_path = os.path.join(dirs['increment'], 'data_{:03d}.pth'.format(update_time))
                    save_data_stable_increment_np(stable_rewards_np[data_length:],
                        stable_err_pos_np[data_length:], stable_dis_pos_np[data_length:],
                        stable_err_euler_np[data_length:], stable_dis_ori_np[data_length:],
                        [actor_loss_list[-1]], [critic_loss_list[-1]], [actor_lr_list[-1]], [critic_lr_list[-1]],
                        [episode_num_list[-1]], [success_num_list[-1]], increment_data_path)

                if stable_rewards_np.shape[0] >= args.sf_num:
                    stable_err_pos_np = stable_err_pos_np[-int(args.sf_num / args.noa):]
                    stable_dis_pos_np = stable_dis_pos_np[-int(args.sf_num / args.noa):]
                    stable_err_euler_np = stable_err_euler_np[-int(args.sf_num / args.noa):]
                    stable_dis_ori_np = stable_dis_ori_np[-int(args.sf_num / args.noa):]
                    stable_rewards_np = stable_rewards_np[-int(args.sf_num / args.noa):]

                data_length = stable_rewards_np.shape[0]

                # fps calculate
                t = time.time()
                fps = (total_steps - total_steps_last) / (t - t_last)
                f_hours = (args.max_train_steps - total_steps) / fps / 3600
                with open(os.path.join(dirs['save'], 'fps.txt'), 'a+') as f:
                    f.write('Update {},\t fps: {:.2f},\tfinish after {:.2f} hours\n'.format(
                        update_time, fps * args.noa, f_hours))
                    if args.target_pos_reinit_method == 'distribution':
                        print(env.sphere_reach_fail_prob)

                total_steps_last, t_last = total_steps, t

                # update to W & B
                env.wandb.log({
                    'actor_loss': actor_loss_list[-1],
                    'critic_loss': critic_loss_list[-1],
                    'tASR': success_num_list[-1] / episode_num_list[-1],
                    'fps': fps * args.noa,
                    'f_hours': f_hours,
                })

def eval(args, env, prefix=''):
    set_seed(args.seed, env)

    args.state_dim = env.state_np.shape[-1]

    dirs = create_saving_dirs(args)
    env.wandb.save(os.path.join(dirs['save'], 'args.txt'))
    # print('Create directory successful!\nROOT FOLDER: >>>> {} <<<<'.format(args.save_dir))

    reinit_freq = args.target_reinit_pos_freq
    stable_step_num = args.stable_step_num

    agent = PPO_continuous(args)

    state_norm = Normalization(shape=args.state_dim)
    eval_episodes = args.eval_episodes

    t_last = time.time()
    while check_files_with_suffix(dirs['log'], '.pth'):
        model_path, model_nums = get_model_path_and_nums(dirs['log'], model_name=args.model_name, type='earliest')
        state_norm = agent.reload_model(model_path)
        # print('Reload model from', model_path)

        stable_err_pos_np = np.empty((0, args.noa, 3))
        stable_dis_pos_np = np.empty((0, args.noa))
        stable_err_euler_np = np.empty((0, args.noa, 3))
        stable_dis_ori_np = np.empty((0, args.noa))
        stable_err_floating_base_euler_np = np.empty((0, args.noa, 3))
        stable_dis_floating_base_ori_np = np.empty((0, args.noa))

        stable_rewards_np = np.empty((0, args.noa))
        success_flag_np = np.empty(0)

        for episode_num in range(int(np.ceil(eval_episodes / args.noa))):
            # print('episode_number={}'.format(episode_num+1))
            episode_steps = 0
            s_np = env.reset()
            if args.use_state_norm:
                s_np = state_norm(s_np, update=False)

            dw_flag_np = np.zeros((args.max_episode_steps, args.noa))
            episode_rewards_np = np.zeros((args.max_episode_steps, args.noa))

            for episode_steps in range(args.max_episode_steps):
                action, _ = agent.choose_action_np(s_np)  # Action and the corresponding log probability

                s_np_, reward_np, done_np, dw_np = env.step(action)
                dw_flag_np[episode_steps] = dw_np
                if args.use_state_norm:
                    s_np_ = state_norm(s_np_, update=False)

                episode_rewards_np[episode_steps] = reward_np
                s_np = s_np_

            episode_err_pos_np = env.his_obs_np['err_pos']
            episode_dis_pos_np = env.his_obs_np['dis_pos']
            episode_err_euler_np = env.his_obs_np['err_euler']
            episode_dis_ori_np = env.his_obs_np['dis_ori']
            episode_err_floating_base_euler_np = env.his_obs_np['err_floating_base_euler']
            episode_dis_floating_base_ori_np = env.his_obs_np['dis_floating_base_ori']

            stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, stable_dis_ori_np, stable_err_floating_base_euler_np, \
                stable_dis_floating_base_ori_np, stable_rewards_np = train_post_process_multiagent_np(
                    env, [],
                    episode_err_pos_np, episode_dis_pos_np, episode_err_euler_np, episode_dis_ori_np,
                    episode_err_floating_base_euler_np, episode_dis_floating_base_ori_np, episode_rewards_np, stable_step_num,
                    stable_err_pos_np, stable_dis_pos_np, stable_err_euler_np, stable_dis_ori_np,
                    stable_err_floating_base_euler_np, stable_dis_floating_base_ori_np, stable_rewards_np, args, use_wandb=False)

            success_flag_np = np.hstack((success_flag_np, success_check(dw_flag_np, args.success_interval)))

        stable_dis_pos_np               = stable_dis_pos_np.flatten()[:args.eval_episodes]
        stable_dis_ori_np               = stable_dis_ori_np.flatten()[:args.eval_episodes]
        stable_rewards_np               = stable_rewards_np.flatten()[:args.eval_episodes]
        stable_dis_floating_base_ori_np = stable_dis_floating_base_ori_np.flatten()[:args.eval_episodes]
        success_flag_np                 = success_flag_np[:args.eval_episodes]

        t = time.time()
        dt = t - t_last
        t_last = t
        fps = args.eval_episodes * args.max_episode_steps / dt
        f_hours = model_nums * dt / 3600

        env.wandb.log({
            prefix+'eAPE': np.mean(stable_dis_pos_np),
            prefix+'eAOE': np.mean(stable_dis_ori_np),
            prefix+'eAFBOE': np.mean(stable_dis_floating_base_ori_np),
            prefix+'eASR': np.mean(success_flag_np),
            prefix+'eAR': np.mean(stable_rewards_np),
            'fps': fps,
            'f_hours': f_hours,
        })

        os.rename(model_path, os.path.join(dirs['log_'], os.path.basename(model_path)))


# region
parser = argparse.ArgumentParser('RTPC')

parser.add_argument('--max_train_steps', type=float2int, default=18e6, help='Maximum number of training steps')
parser.add_argument('--noa', type=float2int, default=24, help='Num of Agent')
parser.add_argument('--batch_size', type=int, default=60000, help='Batch size')
parser.add_argument('--mini_batch_size', type=int, default=6000, help='Minibatch size')
parser.add_argument('--hidden_structure', type=str2list, default='[256, 256, 128]',
                    help='The structure of hidden layers of the neural network')
parser.add_argument('--lr_a', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--lr_c', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--gamma', type=float, default=0.94, help='Discount factor')
parser.add_argument('--reward_scaling_gamma', type=float, default=0.94, help='Reward Scaling Discount factor')
parser.add_argument('--lamda', type=float, default=0.95, help='GAE parameter')
parser.add_argument('--epsilon', type=float, default=0.2, help='PPO clip parameter')
parser.add_argument('--K_epochs', type=int, default=45, help='PPO parameter')
parser.add_argument('--use_adv_norm', type=str2BoolNone, default=True, help='Trick 1:advantage normalization')
parser.add_argument('--use_state_norm', type=str2BoolNone, default=True, help='Trick 2:state normalization')
parser.add_argument('--use_one_state_norm', type=str2BoolNone, default=False, help='Trick 2.5:one state normalization')
parser.add_argument('--use_reward_scaling', type=str2BoolNone, default=True, help='Trick 4:reward scaling')
parser.add_argument('--entropy_coef', type=float, default=0.01, help='Trick 5: policy entropy')

parser.add_argument('--use_grad_clip', type=str2BoolNone, default=True, help='Trick 7: Gradient clip')
parser.add_argument('--set_adam_eps', type=str2BoolNone, default=True, help='Trick 9: set Adam epsilon=1e-5')

parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
parser.add_argument('--gpu_num', type=int, default=-1, help='GPU numbers, 3 for ass01 & ass02; 2 for ass03')

parser.add_argument('--project_name', type=str2BoolNone, default='RTPC', help='project name')
parser.add_argument('--wandb_mode', type=str2BoolNone, default='offline', help='wandb mode: online / offline')
parser.add_argument('--wandb_name', type=str2BoolNone, default=None, help='wandb name')

parser.add_argument('--lr_scheduler', type=str2BoolNone, default='singlecyclic', choices=['const', 'cyclic', 'singlecyclic'],
                    help='Trick 6.1: learning rate scheduler, type = const, cyclic, singlecyclic, exp (useless)')
parser.add_argument('--lr_cycle', type=int, default=15, help='Trick 6.21: learning rate change cycle, for cyclic')
parser.add_argument('--lr_ac_minrate', type=float, default=0.1,
                    help='Trick 6.4.3 lr_a/lr_c minrate: lr_a/c_now = max(lr_a/c_now, lr_a/c * lr_ac_minrate)')
parser.add_argument('--lr_minrate', type=float, default=0.05,
                    help='Trick 6.5 lr minrate: if use cyclic, this can make sure: minimum peak = lr * lr_minrate')

parser.add_argument('--use_her', type=str2BoolNone, default=True, help='Trick 11: Use HER')
parser.add_argument('--her_update_times', type=int, default=100, help='Trick 11.5 HER update times, after this update times, stop her')
parser.add_argument('--after_her_reset_reward_scaling', type=str2BoolNone, default=True, help='Trick 11.6.2 when her finish, reset reward_scaling')

parser.add_argument('--save_dir', type=str, default='RTPC', help='Data save path')

parser.add_argument('--action_dim', type=int, default=6, help='Dimension of action')
parser.add_argument('--seed', type=int, default=0, help='The seed for training')
parser.add_argument('--fps_check_times', type=int, default=300, help='FPS check per times')
parser.add_argument('--model_save_freq', type=int, default=1, help='Saving model after __ update')
parser.add_argument('--eval_episodes', type=int, default=300, help='eval episodes')
parser.add_argument('--sf_num', type=int, default=300, help='sf_num')
parser.add_argument('--joint_max_vel', type=float, default=3, help='Joint max velocity')
parser.add_argument('--time_interval', type=float, default=100., help='Simulate environment time interval')
parser.add_argument('--MNQ', type=str, default='5 6 10', help='Sphere distribution reinit numbers')
parser.add_argument('--r_max', type=float, default=0.65, help='Max distance from agent to target')
parser.add_argument('--r_min', type=float, default=0.25, help='Min distance from agent to target')
parser.add_argument('--target_reinit_pos_freq', type=int, default=1, help='After this times, target will reinit position')

# '''
parser.add_argument('--is_train', type=str2BoolNone, default=True, help='Train or evaluation')
parser.add_argument('--target_pos_reinit_method', type=str2BoolNone, default='random', choices=['random', 'sequence'],
                    help='Target position reinit by random/sequence/distribution')
parser.add_argument('--pos_err_thres', type=float, default=0.1, help='Tracking position error threshold')
parser.add_argument('--ori_err_thres', type=float, default=0.3, help='Tracking quaternion error threshold')
'''
parser.add_argument('--is_train', type=str2BoolNone, default=False, help='Train or evaluation')
parser.add_argument('--target_pos_reinit_method', type=str2BoolNone, default='random', choices=['random', 'sequence', 'distribution'],
                    help='Target position reinit by random/sequence/distribution')
parser.add_argument('--pos_err_thres', type=float, default=0.05, help='Tracking position error threshold')
parser.add_argument('--ori_err_thres', type=float, default=0.1, help='Tracking quaternion error threshold')
# '''

parser.add_argument('--headless', type=str2BoolNone, default=True, help='Use CoppeliaSim with headless mode')
# parser.add_argument('--headless', type=str2BoolNone, default=False, help='Use CoppeliaSim without headless mode')

parser.add_argument('--data_save', type=str2BoolNone, default=False, help='Save data or not')
parser.add_argument('--model_name', type=str2BoolNone, default=None, help='Evaluation model name')

parser.add_argument('--finish_after_reach', type=str2BoolNone, default=False, help='Finish episode after reach')
parser.add_argument('--success_interval', type=int, default=5, help='If no finish after reach, interval this to calculate next success')

parser.add_argument('--max_episode_steps', type=int, default=50, help='The max step num per episode')
parser.add_argument('--stable_step_num', type=int, default=10, help='Stable step number')
parser.add_argument('--pos_err_stable_times', type=int, default=5, help='Tracking stable if position error < threshold more than this times')
parser.add_argument('--arm_angle_0_to_2pi', type=str2BoolNone, default=True, help='Arm angle just limited in 0 to 2pi')
parser.add_argument('--success_sample_rate', type=float, default=0.95, help='Success sample rate')
parser.add_argument('--success_min_rate', type=float, default=0, help='Success minimum rate')

parser.add_argument('--ori_inherent_rate', type=float, default=0.25, help='The inherent rate between pos_error and ori_error')
parser.add_argument('--ori_penalty_rate', type=float, default=0.5, help='The rate of orientation error penalty')
parser.add_argument('--pos_related_to_finish', type=str2BoolNone, default=True, help='Position error related to finish')
parser.add_argument('--ori_related_to_finish', type=str2BoolNone, default=True, help='Orientation error related to finish')

parser.add_argument('--decrease_threshold', type=float, default=1e-2, help='Decrease if fail prob less than threshold')
parser.add_argument('--success_sample_rate_idx', type=float, default=1, help='Success sample rate decrease index')
parser.add_argument('--pos_err_stable_times_increase', type=int, default=1, help='Stable times increase number')

parser.add_argument('--pos_err_thres_idx', type=float, default=0.85, help='Position error threshold decrease index')
parser.add_argument('--ori_err_thres_idx', type=float, default=0.85, help='Quaternion error threshold decrease index')

# reward define
parser.add_argument('--reward_type', type=str2BoolNone, default='dense', choices=['dense', 'sparse'], help='Reward type')

parser.add_argument('--use_pos_err_penalty', type=str2BoolNone, default=True, help='Reward use position error penalty')
parser.add_argument('--use_ori_err_penalty', type=str2BoolNone, default=True, help='Reward use orientation error penalty')

parser.add_argument('--use_ori_decrease_reward', type=str2BoolNone, default=True, help='Reward use orientation error decrease reward')
parser.add_argument('--ori_decrease_reward_rate', type=float, default=0.1, help='Ori decrease reward rate')

parser.add_argument('--use_smooth_penalty', type=str2BoolNone, default=True, help='Reward use smooth penalty')

parser.add_argument('--use_done_reward', type=str2BoolNone, default=True, help='Reward use done reward')
parser.add_argument('--done_reward_rate', type=float, default=0.1, help='Done reward rate')

parser.add_argument('--pos_zero_origin', type=str2BoolNone, default='ee', choices=['ee', 'ab'],
                    help='ee: end effector; ab: agent base')
parser.add_argument('--state_dim_choose', type=str2BoolNone, default='eterrcprdpopv_41',
                    choices=['eterrcprdpopv_41', 'eterr3p3edpopv_32', 'eterr3p4qdpopv_35',],
                    help='E: End, T: Target, Err: Error'
                    'eterrcprdpopv_41:  (Ex~z, Tx~z, Errx~z(CPR),              dis_pos, dis_ori, pj1, vj1)'
                    'eterr3p3edpopv_32: (Ex~z, Tx~z, Errx~z(3d-pos, 3d-euler), dis_pos, dis_ori, pj1, vj1)'
                    'eterr3p4qdpopv_35: (Ex~z, Tx~z, Errx~z(3d-pos, 4d-quat),  dis_pos, dis_ori, pj1, vj1)')

parser.add_argument('--state_dim', type=int, default=0, help='Useless')

args = parser.parse_args()
# endregion

if __name__ == '__main__':

    from Envs.pybullet_SpaceManipulatorReacherMultiagent import PybulletSpaceManipulatorReacherMultiagent

    args.max_train_steps = int(args.max_train_steps / args.noa)
    args.batch_size = int(args.batch_size / args.noa)
    plat = platform_check()
    print(plat)

    args.state_dim = int(args.state_dim_choose.split('_')[-1])
    args.state_dim_choose = args.state_dim_choose.split('_')[0]

    if args.reward_type == 'sparse':    # if use sparse reward, close dense items
        args.use_pos_err_penalty = False
        args.use_ori_err_penalty = False
        args.use_ori_decrease_reward = False
        args.use_smooth_penalty = False

    get_gpuidx_from_dir(args)
    env = PybulletSpaceManipulatorReacherMultiagent(args, plat=plat)

    if args.is_train:
        train(args, env)
    else:
        eval(args, env)
