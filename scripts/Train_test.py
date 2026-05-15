import os
import sys
import time

import numpy as np
from tensorboardX import SummaryWriter

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

try:
    from scripts.make_results import save_result, write_results
except ImportError:
    from make_results import save_result, write_results

training_malicious_path = os.environ.get(
    "RLAEG_TRAINING_MALWARE_DIR",
    os.path.join(parent_path, "data", "Train_Malicious"),
)
testing_malicious_path = os.environ.get(
    "RLAEG_TESTING_MALWARE_DIR",
    os.path.join(parent_path, "data", "Test_Malicious"),
)
maxturn = 10

ACTION_NAMES = (
    "dos_change",
    "section_append",
    "overlay_append",
    "section_add",
    "section_rename",
    "change_time_stamp",
    "upx_pack",
    "upx_unpack",
    "overlay_replace",
    "shift_header",
    "header_disrupt",
    "shift_content",
)


def _empty_action_counts():
    return {action_name: 0 for action_name in ACTION_NAMES}


def _iter_malware_files(path):
    for root, _, files in os.walk(path):
        for file_name in sorted(files):
            yield file_name, os.path.join(root, file_name)


def train(cfg, env, agent):
    print("Start training!")
    print(f"Environment:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}")
    writer = SummaryWriter(comment=cfg.env_name)
    episode_index = 1
    update_index = 1
    rewards = []
    ma_rewards = []

    for file_name, file_path in _iter_malware_files(training_malicious_path):
        state = env.reset(file_name, file_path, test=False)
        if state is None:
            continue

        done = False
        upper_turn = 0
        ep_reward = 0
        while not done and upper_turn < maxturn:
            action = agent.actor_pick_action(state)
            next_state, reward, done, _ = env.step(action, test=False)
            if next_state is None:
                break

            upper_turn += 1
            ep_reward += reward
            agent.upper_memory.push(state, action, reward, next_state, done)
            if len(agent.upper_memory) >= agent.upbatch_size:
                qf_loss, policy_loss, alpha_loss = agent.update()
                writer.add_scalar("critic_loss", qf_loss, update_index)
                writer.add_scalar("actor_loss", policy_loss, update_index)
                writer.add_scalar("alpha_loss", alpha_loss, update_index)
                writer.add_scalar("alpha", agent.upper_alpha, update_index)
                update_index += 1
            state = next_state

        writer.add_scalar("ep_reward", ep_reward, episode_index)
        episode_index += 1
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

    print("Training is over!")
    writer.close()
    return rewards, ma_rewards


def test(cfg, env, agent):
    print("Start testing!")
    print(f"Environment: {cfg.env_name}, Algorithm:{cfg.algo_name}, Device: {cfg.device}")
    input_per = []
    action_counts = _empty_action_counts()
    turn_per = []
    time_per = []
    fail_list = []

    for file_name, file_path in _iter_malware_files(testing_malicious_path):
        file_action_counts = _empty_action_counts()
        action_list = []
        file_size = os.path.getsize(file_path)
        state = env.reset(file_name, file_path, test=True)
        origin_state = state
        start_time = time.perf_counter()
        if state is None:
            continue

        done = False
        turn = 0
        episode_actions = []
        evaded = False
        while not done and turn < maxturn:
            action = agent.actor_pick_action(state)
            action_list.append(action)
            episode_actions.append(action)
            turn += 1
            next_state, reward, done, flag = env.step(action, test=True)
            if next_state is None:
                break

            if reward > 0:
                elapsed = time.perf_counter() - start_time
                time_per.append(elapsed)
                action_seq = flag[1]
                input_size = flag[0]
                input_per.append(input_size / file_size)
                for action_name in action_seq:
                    action_counts[action_name] += 1
                    file_action_counts[action_name] += 1
                turn_per.append(turn)
                write_results(origin_state, file_action_counts, state, action_list)
                evaded = True
            state = next_state

        if not evaded:
            fail_list.append(episode_actions)

    print("Testing is over!")
    save_result()
    fail_list = np.array(fail_list, dtype=object)
    np.save("fail_arr.npy", fail_list)
    print(fail_list)
    return input_per, action_counts, turn_per, time_per
