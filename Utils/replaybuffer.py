import torch
import numpy as np

class PPO_ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0
        self.wo_r_count = 0
        self.r_count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def store_wo_r(self, s, a, a_logprob, s_, dw, done):
        self.s[self.wo_r_count] = s
        self.a[self.wo_r_count] = a
        self.a_logprob[self.wo_r_count] = a_logprob
        self.s_[self.wo_r_count] = s_
        self.dw[self.wo_r_count] = dw
        self.done[self.wo_r_count] = done
        self.wo_r_count += 1

    def store_r(self, r):
        self.r[self.r_count] = r
        self.r_count += 1

    def store_balance(self):
        return self.r_count == self.wo_r_count

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done

class PPO_ReplayBuffer_np:
    def __init__(self, args):
        self.noa = args.noa
        self.batch_size = args.batch_size
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.max_episode_steps = args.max_episode_steps
        self.reset()

    def reset(self):
        self.s          = np.zeros((self.noa, self.batch_size, self.state_dim))
        self.a          = np.zeros((self.noa, self.batch_size, self.action_dim))
        self.a_logprob  = np.zeros((self.noa, self.batch_size, self.action_dim))
        self.r          = np.zeros((self.noa, self.batch_size))
        self.s_         = np.zeros((self.noa, self.batch_size, self.state_dim))
        self.dw         = np.zeros((self.noa, self.batch_size))
        self.done       = np.zeros((self.noa, self.batch_size))
        self.count      = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[:, self.count, :]            = s
        self.a[:, self.count, :]            = a
        self.a_logprob[:, self.count, :]    = a_logprob
        self.r[:, self.count,]              = r
        self.s_[:, self.count, :]           = s_
        self.dw[:, self.count,]             = dw
        self.done[:, self.count,]           = done
        self.count += 1

    def scaling_for_reward(self, reward_scaling):
        r = np.array(np.split(self.r, int(self.r.shape[-1]/self.max_episode_steps), axis=1)).reshape(-1, self.max_episode_steps)
        r_ = []
        for i in range(r.shape[0]):
            r_.append([])
            reward_scaling.reset()
            for j in range(r.shape[1]):
                r_[i].append(reward_scaling(r[i, j])[0])

        r_ = np.array(np.split(np.array(r_), int(self.r.shape[-1]/self.max_episode_steps)))
        self.r = r_.transpose(1, 0, 2).reshape(self.noa, -1)

        return reward_scaling

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done

class Off_Policy_ReplayBuffer(object):
    def __init__(self, args):
        self.max_size = int(args.replaybuffer_max_size)
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.count = 0
        self.wo_r_count = 0
        self.r_count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, self.state_dim))
        self.a = np.zeros((self.max_size, self.action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, self.state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def store_wo_r(self, s, a, s_, dw):
        self.s[self.wo_r_count] = s
        self.a[self.wo_r_count] = a
        self.s_[self.wo_r_count] = s_
        self.dw[self.wo_r_count] = dw
        self.wo_r_count = (self.wo_r_count + 1) % self.max_size

    def store_r(self, r):
        self.r[self.r_count] = r
        self.r_count = (self.r_count + 1) % self.max_size

    def store_balance(self):
        return self.r_count == self.wo_r_count

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)

        return s, a, r, s_, dw

    def sample(self, batch_size, device):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float).to(device)
        batch_a = torch.tensor(self.a[index], dtype=torch.float).to(device)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).to(device)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(device)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw

class Off_Policy_ReplayBuffer_torch_one(object):
    def __init__(self, args):
        self.noa = args.noa
        self.max_size = int(np.ceil(args.replaybuffer_max_size / self.noa) * self.noa)
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.count = 0
        self.wo_r_count = 0
        self.r_count = 0
        self.size = 0
        self.device = torch.device('cuda', args.gpu_idx) if 0 <= args.gpu_idx <= torch.cuda.device_count() else torch.device('cpu')

        self.s  = torch.zeros((self.max_size, self.state_dim)).to(self.device)
        self.a  = torch.zeros((self.max_size, self.action_dim)).to(self.device)
        self.r  = torch.zeros((self.max_size)).to(self.device)
        self.s_ = torch.zeros((self.max_size, self.state_dim)).to(self.device)
        self.dw = torch.zeros((self.max_size)).to(self.device)

    def store(self, s, a, r, s_, dw):
        count_ = self.count + self.noa
        self.s[self.count:count_]  = torch.tensor(s).to(self.device)
        self.a[self.count:count_]  = torch.tensor(a).to(self.device)
        self.r[self.count:count_]  = torch.tensor(r).to(self.device)
        self.s_[self.count:count_] = torch.tensor(s_).to(self.device)
        self.dw[self.count:count_] = torch.tensor(dw).to(self.device)
        self.count = count_ % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + self.noa, self.max_size)  # Record the number of  transitions

    def store_wo_r(self, s, a, s_, dw):
        self.s[self.wo_r_count] = s
        self.a[self.wo_r_count] = a
        self.s_[self.wo_r_count] = s_
        self.dw[self.wo_r_count] = dw
        self.wo_r_count = (self.wo_r_count + 1) % self.max_size

    def store_r(self, r):
        self.r[self.r_count] = r
        self.r_count = (self.r_count + 1) % self.max_size

    def store_balance(self):
        return self.r_count == self.wo_r_count

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)

        return s, a, r, s_, dw

    def sample(self, batch_size, device):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s  = self.s[index]
        batch_a  = self.a[index]
        batch_r  = self.r[index].unsqueeze(1)
        batch_s_ = self.s_[index]
        batch_dw = self.dw[index].unsqueeze(1)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw

