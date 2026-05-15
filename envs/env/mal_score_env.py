import gym
from gym.utils import seeding
import os
import lief
import random
import tempfile
import subprocess
import numpy as np
from envs.utils import interface
import envs.controls.pe_manipulate as manipulate
from envs.utils.make_results import save_adv_malware, save_adv_malware_test
from functionality_verification.cfg_verification import (
    CFGVerificationError,
    extract_cfg_signature,
    is_cfg_verification_enabled,
    verify_modified_cfg,
)

action_table = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.values())}


class FileRetrievalFailure(Exception):
    pass


def check_pack(bytez):
    random.seed(1)
    tmpfilename = os.path.join(
        tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
    with open(tmpfilename, 'wb') as outfile:
        outfile.write(bytez)

    options = ['--force', '--overlay=copy']
    compression_level = random.randint(1, 9)
    options += ['-{}'.format(compression_level)]

    with open(os.devnull, 'w') as DEVNULL:
        retcode = subprocess.call(
            ['upx'] + options + [tmpfilename, '-o', tmpfilename + '_packed'], stdout=DEVNULL, stderr=DEVNULL)

    os.unlink(tmpfilename)

    if retcode == 0:
        # success, 0
        return 0
    else:
        # failed, 1
        return 1


def check_unpack(bytez):
    tmpfilename = os.path.join(
        tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))

    with open(tmpfilename, 'wb') as outfile:
        outfile.write(bytez)

    with open(os.devnull, 'w') as DEVNULL:
        retcode = subprocess.call(
            ['upx', tmpfilename, '-d', '-o', tmpfilename + '_unpacked'], stdout=DEVNULL, stderr=DEVNULL)

    os.unlink(tmpfilename)

    if retcode == 0:
        # success, 0
        return 0
    else:
        # failed, 1
        return 1


def check_secappend(binary):
    avail_len = 0
    for s in binary.sections:
        section_size = s.size
        section_vsize = s.virtual_size
        avail_len = section_size - section_vsize
        if avail_len > 0:
            break
    if avail_len > 0:
        return 0
    else:
        return 1


def check_error(action, obs):
    if action == 1 and obs[29] == 0:
        return 1
    if action == 6 and obs[27] == 0:
        return 1
    if action == 7 and obs[28] == 0:
        return 1

    if action == 5 and obs[30] == 1:
        return 1
    if action == 4 and obs[31] == 1:
        return 1
    if action == 8 and obs[32] == 1:
        return 1
    if action == 1 and obs[33] == 1:
        return 1
    if action == 10 and obs[34] == 1:
        return 1
    return 0


class Mal_Score_Env(gym.Env):

    def __init__(self):
        self.thresholds = 0.90
        self.maxturn = 10


    def _reset(self, file_name, file_path, test):

        """
            First step: Fetch the malicious file

            Second step: Evaluation of the original malware score

            Third step: Describe the state space of the original malware

        :return: observation of the original malware

        """
        self.turn = 0
        self.input_size = 0
        self.seed()
        self.file_path = file_path
        self.file_name = file_name
        self.action_seq = []
        self.a = {}
        self.original_cfg_signature = None
        self.bytez = interface.fetch_file(self.file_name, self.file_path)
        self.score = interface.get_score_local(self.bytez)
        self.original_score = self.score
        print("The {}'s original score is {}".format(file_name, self.score))
        self.obs = interface.get_rl_obs(self.bytez)

        try:
            binary = lief.PE.parse(self.bytez)
        except:
            return None

        # manu feature
        self.obs = np.array(self.obs)
        if check_pack(self.bytez) != 0:
            self.obs = np.append(self.obs, 0)
        else:
            self.obs = np.append(self.obs, 1)
        if check_unpack(self.bytez) != 0:
            self.obs = np.append(self.obs, 0)
        else:
            self.obs = np.append(self.obs, 1)
        if check_secappend(binary) != 0:
            self.obs = np.append(self.obs, 0)
        else:
            self.obs = np.append(self.obs, 1)
        self.obs = np.append(self.obs, 0)
        self.obs = np.append(self.obs, 0)
        self.obs = np.append(self.obs, 0)
        self.obs = np.append(self.obs, 0)
        self.obs = np.append(self.obs, 0)
        self.obs = np.append(self.obs, 0)
        self.obs = np.append(self.obs, 0)

        self.ori_obs = self.obs

        if not self._capture_original_cfg():
            return None

        if self.score < self.thresholds:
            if test == True:
                save_adv_malware_test(self.bytez, self.score, self.file_name)
            return None

        return self.obs

    def _step(self, action_index, test):
        """
            First step: Make specified changes to the malware and return the changed malware

            Second step: Evaluation of the changed malware score and
            decide if it should be terminated (end of modification)

            Third step: Return the new observation,reward and done

        :param action_index:

        :return:

        """
        upper_action = action_index
        function = action_table[upper_action]
        if function in self.a:
            self.a[function] += 1
        else:
            self.a[function] = 1
        self.turn += 1
        self.bytez, length = self._take_action(action_index)
        try:
            binary = lief.PE.parse(self.bytez)
        except:
            return None, None, None, None
        self.score = interface.get_score_local(self.bytez)
        self.obs = interface.get_rl_obs(self.bytez)

        self.input_size += length

        # manu feature
        self.obs = np.array(self.obs)
        if check_pack(self.bytez) != 0:
            self.obs = np.append(self.obs, 0)
        else:
            self.obs = np.append(self.obs, 1)
        if check_unpack(self.bytez) != 0:
            self.obs = np.append(self.obs, 0)
        else:
            self.obs = np.append(self.obs, 1)
        if check_secappend(binary) != 0:
            self.obs = np.append(self.obs, 0)
        else:
            self.obs = np.append(self.obs, 1)
        if 'change_time_stamp' in self.a:
            self.obs = np.append(self.obs, 1)
        else:
            self.obs = np.append(self.obs, 0)
        if 'section_rename' in self.a:
            self.obs = np.append(self.obs, 1)
        else:
            self.obs = np.append(self.obs, 0)
        if 'overlay_replace' in self.a:
            self.obs = np.append(self.obs, 1)
        else:
            self.obs = np.append(self.obs, 0)
        if 'dos_change' in self.a:
            self.obs = np.append(self.obs, 1)
        else:
            self.obs = np.append(self.obs, 0)
        if 'header_disrupt' in self.a:
            self.obs = np.append(self.obs, 1)
        else:
            self.obs = np.append(self.obs, 0)

        resu = check_error(upper_action, self.ori_obs)
        if resu == 1:
            self.obs = np.append(self.obs, 1)
        else:
            self.obs = np.append(self.obs, 0)
        if self.score < self.thresholds:
            self.obs = np.append(self.obs, 1)
        else:
            self.obs = np.append(self.obs, 0)

        self.ori_obs = self.obs

        print(self.obs)
        print(resu)

        if self.score < self.thresholds:
            episode_over = True
            if self._current_cfg_is_valid():
                reward = 1
                print("Success! CFG is same.")
                if test is False:
                    save_adv_malware(self.bytez, self.score, self.file_name)
                else:
                    save_adv_malware_test(self.bytez, self.score, self.file_name)
            else:
                reward = -0.2
                print("Detector evasion found, but CFG changed. Treat as failed.")
        elif self.turn == self.maxturn:
            if resu == 0:
                reward = 0
            else:
                reward = -0.2
            episode_over = True

        else:
            if resu == 0:
                reward = 0
            else:
                reward = -0.2
            self.original_score = self.score
            episode_over = False

        if episode_over:
            print("Episode is over! The final score is {}, The action sequence is {}".format(self.score, self.action_seq))
        return self.obs, reward, episode_over, [self.input_size, self.action_seq]

    def _take_action(self, action_index):
        upper_action = action_index
        function = action_table[upper_action]
        print("The next modification is {}, the current score is {}".format(function, self.score))
        self.action_seq.append(function)
        # return eval(function)(self.bytez, lower_action)
        return eval(function)(self.bytez)

    def _capture_original_cfg(self):
        if not is_cfg_verification_enabled():
            return True

        try:
            self.original_cfg_signature = extract_cfg_signature(self.file_path)
        except CFGVerificationError as exc:
            print("Skip {} because original CFG extraction failed: {}".format(
                self.file_name, exc))
            return False

        print("Captured original CFG for {} with {} functions.".format(
            self.file_name, len(self.original_cfg_signature)))
        return True

    def _current_cfg_is_valid(self):
        if not is_cfg_verification_enabled():
            return True

        try:
            result = verify_modified_cfg(
                self.original_cfg_signature, self.bytez, self.file_name)
        except CFGVerificationError as exc:
            print("CFG verification failed for {}: {}".format(self.file_name, exc))
            return False

        print("CFG verification for {}: {}".format(self.file_name, result))
        return result["same"]

    def _render(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _close(self):
        pass
