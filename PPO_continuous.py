import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import sys
sys.path.append('..')
from Models.PPO_actor_critic import Actor_Gaussian, Critic
import os
import numpy as np

class PPO_continuous():
    def __init__(self, args):
        self.joint_max_vel = args.joint_max_vel
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.use_reward_scaling = args.use_reward_scaling

        lr_schedulers = ['const', 'cyclic', 'singlecyclic']
        self.lr_scheduler = args.lr_scheduler
        assert (self.lr_scheduler in lr_schedulers)
        self.lr_cycle = args.lr_cycle

        self.lr_ac_minrate = args.lr_ac_minrate
        self.lr_minrate = args.lr_minrate

        if self.lr_cycle % 2:
            self.cyclic_array = np.linspace(1, self.lr_ac_minrate, int((self.lr_cycle + 1) / 2))
            self.cyclic_array = np.hstack((self.cyclic_array, np.linspace(self.cyclic_array[-2], 1, int((self.lr_cycle - 1) / 2))))
        else:
            self.cyclic_array = np.linspace(1, self.lr_ac_minrate, int((self.lr_cycle) / 2))
            self.cyclic_array = np.hstack((self.cyclic_array, np.linspace(self.cyclic_array[-1], 1, int(self.lr_cycle / 2))))
        self.singlecyclic_array = np.linspace(1, self.lr_ac_minrate, self.lr_cycle)

        self.use_adv_norm = args.use_adv_norm
        self.gpu_idx = args.gpu_idx

        self.device = torch.device('cuda', self.gpu_idx) if 0 <= self.gpu_idx <= torch.cuda.device_count() \
            else torch.device('cpu')
        print(self.device)
        self.is_train = args.is_train

        self.actor = Actor_Gaussian(args).to(self.device)
        self.critic = Critic(args).to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        if self.device.type == 'cuda':
            self.choose_action_np   = self.cuda_choose_action_np
            self.get_logprob        = self.cuda_get_logprob
        elif self.device.type == 'cpu':
            self.choose_action_np   = self.cpu_choose_action_np
            self.get_logprob        = self.cpu_get_logprob

    def cuda_choose_action_np(self, s):
        # 确保输入张量在GPU上
        s = (torch.tensor(s, dtype=torch.float)).to(self.device)

        with torch.no_grad():
            # 确保模型在GPU上
            if self.is_train:
                dist = self.actor.get_dist(s)
                a = dist.sample()  # 根据概率分布采样动作
                a = torch.clamp(a, -self.joint_max_vel, self.joint_max_vel)  # [-max,max]
                a_logprob = dist.log_prob(a)  # 动作的对数概率密度
            else:
                a = self.actor(s)
                a = torch.clamp(a, -self.joint_max_vel, self.joint_max_vel)  # [-max,max]

        # 在返回前将张量从GPU移到CPU，并转换为NumPy数组
        if self.is_train:
            return a.detach().cpu().numpy(), a_logprob.detach().cpu().numpy()
        else:
            return a.detach().cpu().numpy(), 1

    def cpu_choose_action_np(self, s):
        # 确保输入张量在GPU上
        s = (torch.tensor(s, dtype=torch.float))

        with torch.no_grad():
            # 确保模型在GPU上
            if self.is_train:
                dist = self.actor.get_dist(s)
                a = dist.sample()  # 根据概率分布采样动作
                a = torch.clamp(a, -self.joint_max_vel, self.joint_max_vel)  # [-max,max]
                a_logprob = dist.log_prob(a)  # 动作的对数概率密度
            else:
                a = self.actor(s)
                a = torch.clamp(a, -self.joint_max_vel, self.joint_max_vel)  # [-max,max]

        # 在返回前将张量从GPU移到CPU，并转换为NumPy数组
        if self.is_train:
            return a.numpy().squeeze(), a_logprob.numpy().squeeze()
        else:
            return a.squeeze(), 1

    def cuda_get_logprob(self, s, a):
        s = torch.tensor(s, dtype=torch.float).to(self.device)
        a = torch.tensor(a, dtype=torch.float).to(self.device)

        with torch.no_grad():
            # 确保模型在GPU上
            dist = self.actor.get_dist(s)
            a = torch.clamp(a, -self.joint_max_vel, self.joint_max_vel)  # [-max,max]
            a_logprob = dist.log_prob(a)  # 动作的对数概率密度
        # 在返回前将张量从GPU移到CPU，并转换为NumPy数组
        return a_logprob.detach().cpu().numpy()

    def cpu_get_logprob(self, s, a):
        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.float)

        with torch.no_grad():
            # 确保模型在GPU上
            self.actor.to(self.device)
            dist = self.actor.get_dist(s)
            a = torch.clamp(a, -self.joint_max_vel, self.joint_max_vel)  # [-max,max]
            a_logprob = dist.log_prob(a)  # 动作的对数概率密度
        # 在返回前将张量从GPU移到CPU，并转换为NumPy数组
        if self.device.type == 'cpu':
            return a_logprob.numpy()
        else:
            return a_logprob.detach().cpu().numpy()

    def update_np(self, replay_buffer_np, total_steps, update_time, reward_scaling):
        if self.use_reward_scaling:
            reward_scaling = replay_buffer_np.scaling_for_reward(reward_scaling)
        s, a, a_logprob, r, s_, dw, done = replay_buffer_np.numpy_to_tensor()
        self.noa = s.shape[0] if s.dim() == 3 else 1
        s = s.reshape(-1, self.state_dim).to(self.device)
        a = a.reshape(-1, self.action_dim).to(self.device)
        a_logprob = a_logprob.reshape(-1, self.action_dim).to(self.device)
        r = r.reshape(-1).to(self.device).unsqueeze(1)
        s_ = s_.reshape(-1, self.state_dim).to(self.device)
        dw = dw.reshape(-1).to(self.device).unsqueeze(1)
        done = done.reshape(-1).to(self.device).unsqueeze(1)
        '''
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        '''

        # region
        if update_time % self.lr_cycle == 0:
            tmp = 1 - total_steps * (1 - self.lr_minrate) / self.max_train_steps
            self.singlecyclic_array = np.linspace(tmp, self.lr_ac_minrate, self.lr_cycle)
            if self.lr_cycle % 2:
                self.cyclic_array = np.linspace(tmp, self.lr_ac_minrate, int((self.lr_cycle + 1) / 2))
                self.cyclic_array = np.hstack(
                    (self.cyclic_array, np.linspace(self.cyclic_array[-2], tmp, int((self.lr_cycle - 1) / 2))))
            else:
                self.cyclic_array = np.linspace(tmp, self.lr_ac_minrate, int((self.lr_cycle) / 2))
                self.cyclic_array = np.hstack(
                    (self.cyclic_array, np.linspace(self.cyclic_array[-1], tmp, int(self.lr_cycle / 2))))

        lr_rate = 1
        if self.lr_scheduler == 'const':
            lr_a_now = self.lr_a
            lr_c_now = self.lr_c
        elif self.lr_scheduler == 'cyclic':
            lr_a_now = self.lr_a * self.cyclic_array[update_time % self.lr_cycle]
            lr_c_now = self.lr_c * self.cyclic_array[update_time % self.lr_cycle]
            lr_rate = 1 - self.lr_minrate
        elif self.lr_scheduler == 'singlecyclic':
            lr_a_now = self.lr_a * self.singlecyclic_array[update_time % self.lr_cycle]
            lr_c_now = self.lr_c * self.singlecyclic_array[update_time % self.lr_cycle]
            lr_rate = 1 - self.lr_minrate

        if self.lr_scheduler != 'singlecyclic' and self.lr_scheduler != 'cyclic':
            lr_a_now = lr_a_now * (1 - lr_rate * total_steps / self.max_train_steps)
            lr_c_now = lr_c_now * (1 - lr_rate * total_steps / self.max_train_steps)

        for p in self.optimizer_actor.param_groups:
            p['lr'] = np.max([lr_a_now, self.lr_a * self.lr_ac_minrate])
        for p in self.optimizer_critic.param_groups:
            p['lr'] = np.max([lr_c_now, self.lr_c * self.lr_ac_minrate])
        # endregion

        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs     # TD Error
            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size * self.noa)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                self.actor_loss_save = actor_loss.mean().detach().cpu().tolist()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = torch.nn.functional.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.critic_loss_save = critic_loss.detach().cpu().tolist()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        return reward_scaling

    def save_model(self, state_norm, model_path=''):
        print('=> saving network to {} ...'.format(model_path))
        checkpoint = {
            'actor_net': self.actor.state_dict(),
            'critic_net': self.critic.state_dict(),
            'state_norm': state_norm,
        }
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print('=> model params is saved!')
        return

    def reload_model(self, model_path=''):
        if os.path.exists(model_path):
            print('=> reloading model from {} ...'.format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
            self.actor.load_state_dict(checkpoint['actor_net'])
            self.critic.load_state_dict(checkpoint['critic_net'])
            state_norm = checkpoint['state_norm']
        else:
            raise ValueError('No model file is found in {}'.format(model_path))
        return state_norm
