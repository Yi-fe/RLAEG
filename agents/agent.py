import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# critic network
class CriticNet(nn.Module):
    def __init__(self, n_actions, init_w=3e-3):
        super(CriticNet, self).__init__()

        self.linear1 = nn.Linear(37, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, n_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        linear1_out = F.relu(self.linear1(state))
        linear2_out = F.relu(self.linear2(linear1_out))
        linear3_out = self.linear3(linear2_out)
        return linear3_out



# actor network
class ActorNet(nn.Module):

    def __init__(self, n_actions, init_w=3e-3):
        super(ActorNet, self).__init__()

        self.linear1 = nn.Linear(37, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, n_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        linear1_out = F.relu(self.linear1(state))
        linear2_out = F.relu(self.linear2(linear1_out))
        linear3_out = self.linear3(linear2_out)
        out = F.softmax(linear3_out, dim=-1)
        return out


# SAC algorithm
class SAC:

    def __init__(self, env, cfg):
        self.upbatch_size = cfg.upper_batch_size
        self.device = cfg.device
        self.env = env
        self.gamma = cfg.gamma

        n_upper_actions = cfg.upper_action
        # Replay_buffer
        self.upper_memory = ReplayBuffer(cfg.upper_memory_capacity)
        # upper Q Network
        self.upper_actor_net = ActorNet(n_upper_actions).to(self.device)
        self.upper_actor_target_net = ActorNet(n_upper_actions).to(self.device)
        self.upper_critic_net = CriticNet(n_upper_actions).to(self.device)
        self.upper_critic_target_net = CriticNet(n_upper_actions).to(self.device)

        # optimizer
        self.upper_critic_optimizer = optim.Adam(self.upper_critic_net.parameters(), lr=cfg.critic_lr)
        self.upper_actor_optimizer = optim.Adam(self.upper_actor_net.parameters(), lr=cfg.actor_lr)

        # copy the parameters
        for target_param, param in zip(self.upper_critic_target_net.parameters(),
                                       self.upper_critic_net.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.upper_actor_target_net.parameters(),
                                       self.upper_actor_net.parameters()):
            target_param.data.copy_(param.data)

        self.critic_criterion = nn.MSELoss()
        self.actor_criterion = nn.MSELoss()
        self.upper_target_entropy = -np.log((1.0 / n_upper_actions)) * 0.5
        self.upper_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.upper_alpha = self.upper_log_alpha.exp()
        self.upper_alpha_optim = Adam([self.upper_log_alpha], lr=3e-4, eps=1e-4)

    def actor_pick_action(self, state=None):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, _ = self.produce_action_and_action_info(state, self.upper_actor_net)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state, net):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        action_probabilities = net(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_critic_parameters(self, critic_loss):
        """Updates the parameters for both critics"""
        self.upper_critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.upper_critic_net.parameters(), 5)
        self.upper_critic_optimizer.step()
        self.soft_update_of_target_network(self.upper_critic_net, self.upper_critic_target_net, tau=0.005)

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.upper_actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.upper_actor_net.parameters(), 5)
        self.upper_actor_optimizer.step()
        self.soft_update_of_target_network(self.upper_actor_net, self.upper_actor_target_net, tau=0.005)
        if alpha_loss is not None:
            self.upper_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.upper_alpha_optim.step()
            self.upper_alpha = self.upper_log_alpha.exp()

    def update(self):
        if len(self.upper_memory) < self.upbatch_size:
            return
        state, action, reward, next_state, done = self.upper_memory.sample(self.upbatch_size)
        state = np.array(state)
        next_state = np.array(next_state)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.tensor(action, device=self.device).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        with torch.no_grad():
            next_state_action, (
            action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(
                next_state, self.upper_actor_target_net)
            qf_next_target = self.upper_critic_target_net(next_state)
            min_qf_next_target = action_probabilities * (qf_next_target - self.upper_alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward + (1.0 - done) * self.gamma * min_qf_next_target
        qf = self.upper_critic_net(state).gather(dim=1, index=action)
        qf_loss = self.critic_criterion(qf, next_q_value)
        self.update_critic_parameters(qf_loss)
        new_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state,
                                                                                                              self.upper_actor_net)
        qf_pi = self.upper_critic_net(state)
        inside_term = self.upper_alpha * log_action_probabilities - qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        alpha_loss = -(self.upper_log_alpha * (log_action_probabilities + self.upper_target_entropy).detach()).mean()
        self.update_actor_parameters(policy_loss, alpha_loss)
        return qf_loss, policy_loss, alpha_loss

    def save(self, path):
        torch.save(self.upper_critic_net.state_dict(), path + "upper_critic")
        torch.save(self.upper_critic_optimizer.state_dict(), path + "upper_critic_optimizer")

        torch.save(self.upper_actor_net.state_dict(), path + "upper_actor")
        torch.save(self.upper_actor_optimizer.state_dict(), path + "upper_actor_optimizer")

    def load(self, path):
        self.upper_critic_net.load_state_dict(torch.load(path + "upper_critic"))
        self.upper_critic_optimizer.load_state_dict(torch.load(path + "upper_critic_optimizer"))
        self.upper_critic_target_net = copy.deepcopy(self.upper_critic_net)

        self.upper_actor_net.load_state_dict(torch.load(path + "upper_actor"))
        self.upper_actor_optimizer.load_state_dict(torch.load(path + "upper_actor_optimizer"))
        self.upper_actor_target_net = copy.deepcopy(self.upper_actor_net)
