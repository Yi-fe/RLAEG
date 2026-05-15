#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()
    plt.title("learning curve on {} of {} for {}".format(
        cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('episodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_rewards_curve".format(tag))
    plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('episodes')
    plt.plot(losses, label='losses')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


def save_results_1(dic, tag='train', path='./results'):
    """Save reward dictionaries."""
    for key, value in dic.items():
        np.save(path + '{}_{}.npy'.format(tag, key), value)
    print('Results saved!')


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    """Save rewards and moving-average rewards."""
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')


def make_dir(*paths):
    """Create directories if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    """Remove empty child directories under the provided paths."""
    for path in paths:
        dirs = os.listdir(path)
        for dir_name in dirs:
            child = os.path.join(path, dir_name)
            if not os.listdir(child):
                os.removedirs(child)


def save_args(args):
    """Save command-line arguments to the result directory."""
    args_dict = args.__dict__
    with open(args.result_path + 'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for each_arg, value in args_dict.items():
            f.writelines(each_arg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")
