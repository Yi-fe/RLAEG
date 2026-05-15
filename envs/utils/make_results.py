# coding=utf-8
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mpl.use('tkagg')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADV_LABEL_MALWARE_DIR = Path(os.environ.get(
    "RLAEG_ADV_LABEL_MALWARE_DIR",
    PROJECT_ROOT / "data" / "label_adv_malware",
))
ADV_SCORE_MALWARE_DIR = Path(os.environ.get(
    "RLAEG_ADV_SCORE_MALWARE_DIR",
    PROJECT_ROOT / "data" / "score_adv_malware",
))
ADV_SCORE_TEST_MALWARE_DIR = Path(os.environ.get(
    "RLAEG_ADV_SCORE_TEST_MALWARE_DIR",
    PROJECT_ROOT / "data" / "score_adv_test_malware",
))
ADV_LABEL_TEST_MALWARE_DIR = Path(os.environ.get(
    "RLAEG_ADV_LABEL_TEST_MALWARE_DIR",
    PROJECT_ROOT / "data" / "label_adv_test_malware",
))


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


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    """Save rewards and moving-average rewards."""
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')


def save_data(input_per, dic, turn_per, time_per, tag='test', path='./results'):
    np.save(path + '{}_input_per.npy'.format(tag), input_per)
    np.save(path + '{}_dic.npy'.format(tag), dic)
    np.save(path + '{}_turn_per.npy'.format(tag), turn_per)
    np.save(path + '{}_time_per.npy'.format(tag), time_per)


def save_adv_malware(bytez, label, file_name):
    """Save adversarial malware samples for training."""
    output_dir = ADV_LABEL_MALWARE_DIR if label == 0 else ADV_SCORE_MALWARE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / file_name, 'wb') as outfile:
        outfile.write(bytez)


def save_adv_malware_test(bytez, label, file_name):
    """Save adversarial malware samples for testing."""
    output_dir = ADV_LABEL_TEST_MALWARE_DIR if label == 0 else ADV_SCORE_TEST_MALWARE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / file_name, 'wb') as outfile:
        outfile.write(bytez)


def make_dir(*paths):
    """Create directories if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
