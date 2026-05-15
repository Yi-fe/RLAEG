#!/usr/bin/env python
# coding=utf-8
import math
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=128):
        """Fully connected Q-network."""
        super(MLP, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(n_states, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.advantage_1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.value_1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.advantage_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.value_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.advantage_3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.value_3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.advantage_4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.value_4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.advantage_5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.value_5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.sigma = nn.Parameter(torch.Tensor(n_actions))
        self.sigma.data.fill_(math.log(0.1))

    def net1_forward(self, x):
        x = self.hidden(x)
        advantage = self.advantage_1(x)
        value = self.value_1(x)
        return value + advantage - advantage.mean()

    def net2_forward(self, x):
        x = self.hidden(x)
        advantage = self.advantage_2(x)
        value = self.value_2(x)
        return value + advantage - advantage.mean()

    def net3_forward(self, x):
        x = self.hidden(x)
        advantage = self.advantage_3(x)
        value = self.value_3(x)
        return value + advantage - advantage.mean()

    def net4_forward(self, x):
        x = self.hidden(x)
        advantage = self.advantage_4(x)
        value = self.value_4(x)
        return value + advantage - advantage.mean()

    def net5_forward(self, x):
        x = self.hidden(x)
        advantage = self.advantage_5(x)
        value = self.value_5(x)
        return value + advantage - advantage.mean()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store one transition in the replay buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        """Return the number of stored transitions."""
        return len(self.buffer)


def get_kl(teacher_dist_info, student_dist_info):
    pi = Normal(loc=teacher_dist_info[0], scale=teacher_dist_info[1])
    pi_new = Normal(student_dist_info[0], scale=student_dist_info[1])
    kl = torch.mean(kl_divergence(pi, pi_new))
    return kl


class DQN:
    def __init__(self, n_states, n_actions, cfg):
        self.n_actions = n_actions
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.target_net = MLP(n_states, n_actions).to(self.device)
        self.policy_net = MLP(n_states, n_actions).to(self.device)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr, eps=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.5,
            total_iters=100000,
        )
        self.memory_1 = ReplayBuffer(cfg.memory_capacity)
        self.memory_2 = ReplayBuffer(cfg.memory_capacity)
        self.memory_3 = ReplayBuffer(cfg.memory_capacity)
        self.memory_4 = ReplayBuffer(cfg.memory_capacity)
        self.memory_5 = ReplayBuffer(cfg.memory_capacity)
        self.iteration = 0

    def choose_action(self, state, index):
        """Choose the action with the maximum predicted Q value."""
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            if index == 0:
                q_values = self.policy_net.net1_forward(state)
            elif index == 1:
                q_values = self.policy_net.net2_forward(state)
            elif index == 2:
                q_values = self.policy_net.net3_forward(state)
            elif index == 3:
                q_values = self.policy_net.net4_forward(state)
            else:
                q_values = self.policy_net.net5_forward(state)
            action = q_values.max(1)[1].item()
        return action

    def update(self, teacher, index):
        if index == 0:
            if len(self.memory_1) < self.batch_size:
                return
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_1.sample(
                self.batch_size)
        elif index == 1:
            if len(self.memory_2) < self.batch_size:
                return
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_2.sample(
                self.batch_size)
        elif index == 2:
            if len(self.memory_3) < self.batch_size:
                return
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_3.sample(
                self.batch_size)
        elif index == 3:
            if len(self.memory_4) < self.batch_size:
                return
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_4.sample(
                self.batch_size)
        else:
            if len(self.memory_5) < self.batch_size:
                return
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_5.sample(
                self.batch_size)

        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        if hasattr(teacher, "critic_net"):
            q_values_mean = teacher.critic_net(state_batch).detach()
            teacher_params = OrderedDict(teacher.critic_net.named_parameters())
            q_values_std = torch.exp(torch.clamp(teacher_params['sigma'], min=math.log(1e-6)))
        else:
            q_values_mean = teacher.upper_critic_net(state_batch).detach()
            q_values_std = None

        if index == 0:
            dqn_q_values_mean = self.policy_net.net1_forward(state_batch)
        elif index == 1:
            dqn_q_values_mean = self.policy_net.net2_forward(state_batch)
        elif index == 2:
            dqn_q_values_mean = self.policy_net.net3_forward(state_batch)
        elif index == 3:
            dqn_q_values_mean = self.policy_net.net4_forward(state_batch)
        else:
            dqn_q_values_mean = self.policy_net.net5_forward(state_batch)
        dqn_params = OrderedDict(self.policy_net.named_parameters())
        dqn_q_values_std = torch.exp(torch.clamp(dqn_params['sigma'], min=math.log(1e-6)))
        if q_values_std is None:
            loss = F.mse_loss(dqn_q_values_mean, q_values_mean)
        else:
            loss = get_kl([q_values_mean, q_values_std], [dqn_q_values_mean, dqn_q_values_std])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
