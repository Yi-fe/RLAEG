import argparse
import datetime
import os
import sys

import gym
import numpy as np
import torch

from agents.agent import SAC
from envs.utils.make_results import make_dir, save_results
try:
    from scripts.Train_test import train, test
except ImportError:
    from Train_test import train, test


curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
algo_name = 'SAC'
env_name = 'Mal_origin'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlotConfig:
    def __init__(self) -> None:
        self.algo_name = algo_name
        self.env_name = env_name
        self.device = device
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'
        self.save_fig = True


def get_score_args():
    """Return score-environment hyperparameters."""
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='SAC', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='malware-score-v0', type=str, help="name of environment")
    parser.add_argument('--gamma', default=0.8, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=3e-4, type=float, help="critic network learning rate")
    parser.add_argument('--actor_lr', default=3e-4, type=float, help="actor network learning rate")
    parser.add_argument('--upper_lr', default=3e-4, type=float, help="upper network learning rate")
    parser.add_argument('--lower_lr', default=3e-4, type=float, help="lower network learning rate")
    parser.add_argument('--epsilon_start', default=0.95, type=float, help="actor network learning rate")
    parser.add_argument('--epsilon_end', default=0.01, type=float, help="actor network learning rate")
    parser.add_argument('--epsilon_decay', default=10000, type=float, help="actor network learning rate")
    parser.add_argument('--upper_memory_capacity', default=10000, type=int, help="upper memory capacity")
    parser.add_argument('--lower_memory_capacity', default=10000, type=int, help="upper memory capacity")
    parser.add_argument('--upper_action', default=12, type=int, help="Number of upper level actions")
    parser.add_argument('--lower_action', default=20, type=int, help="Number of upper level actions except import")
    parser.add_argument('--upper_batch_size', default=256, type=int)
    parser.add_argument('--lower_batch_size', default=256, type=int)
    parser.add_argument('--lr', default=7e-4, type=float, help="learning rate")
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name +
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name +
                                                '/' + curr_time + '/models/')
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def get_label_args():
    """Return label-environment hyperparameters."""
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='SAC', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='malware-score-v0', type=str, help="name of environment")
    parser.add_argument('--gamma', default=0.85, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=3e-4, type=float, help="critic network learning rate")
    parser.add_argument('--actor_lr', default=3e-4, type=float, help="actor network learning rate")
    parser.add_argument('--upper_memory_capacity', default=10000, type=int, help="upper memory capacity")
    parser.add_argument('--lower_memory_capacity', default=10000, type=int, help="upper memory capacity")
    parser.add_argument('--upper_action', default=4, type=int, help="Number of upper level actions")
    parser.add_argument('--lower_action', default=30, type=int, help="Number of upper level actions except import")
    parser.add_argument('--upper_batch_size', default=256, type=int)
    parser.add_argument('--lower_batch_size', default=256, type=int)
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name +
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name +
                                                '/' + curr_time + '/models/')
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def env_agent_config(cfg, seed=1):
    """Create the environment and agent."""
    env = gym.make(cfg.env_name)
    agent = SAC(env, cfg)
    if seed != 0:
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
    return env, agent


if __name__ == "__main__":
    cfg = get_score_args()
    plot_cfg = PlotConfig()
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)
    agent.save(path=plot_cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)

    env, agent = env_agent_config(cfg, seed=2)
    agent.load(path=plot_cfg.model_path)
    input_per, dic, turn_per, time_per = test(cfg, env, agent)
