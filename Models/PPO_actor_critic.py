import torch
import torch.nn as nn
from torch.distributions import Normal

class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_structure = args.hidden_structure
        self.joint_max_vel = args.joint_max_vel

        self.actor = nn.Sequential()
        self.actor.add_module('L1', nn.Linear(self.state_dim, self.hidden_structure[0]))
        self.actor.add_module('T1', nn.Tanh())

        for i in range(len(self.hidden_structure) - 1):
            self.hidden_layer = nn.Linear(self.hidden_structure[i], self.hidden_structure[i + 1])
            self.actor.add_module('L{}'.format(i + 2), self.hidden_layer)

            if i < len(self.hidden_structure) - 1:
                self.actor.add_module('T{}'.format(i + 2), nn.Tanh())

        self.mean_layer = nn.Linear(self.hidden_structure[-1], self.action_dim)
        self.actor.add_module('L_mean', self.mean_layer)

        self.actor.add_module('T_mean', nn.Tanh())

        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self, state):
        out = self.actor(state)
        mean = self.joint_max_vel * out

        return mean

    def get_dist(self, state):
        mean = self.forward(state)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state_dim = args.state_dim
        self.hidden_structure = args.hidden_structure

        self.critic = nn.Sequential()
        self.layer = nn.Linear(self.state_dim, self.hidden_structure[0])
        self.critic.add_module('L1', self.layer)

        self.critic.add_module('T1', nn.Tanh())

        for i in range(len(self.hidden_structure)-1):
            self.layer = nn.Linear(self.hidden_structure[i], self.hidden_structure[i+1])
            self.critic.add_module('L{}'.format(i+2), self.layer)
            self.critic.add_module('T{}'.format(i+2), nn.Tanh())

        self.layer = nn.Linear(self.hidden_structure[-1], 1)
        self.critic.add_module('Le', self.layer)

    def forward(self, state):
        out = self.critic(state)
        return out
