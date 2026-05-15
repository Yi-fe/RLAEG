from shutil import copy
import gym
from gym.utils import seeding
import numpy as np

from envs.utils import interface
from envs.utils.make_results import save_adv_malware
import envs.controls.pe_manipulate as manipulate
from envs.utils.manipulator_pefeatures import PEFeatureExtract
from functionality_verification.cfg_verification import (
    CFGVerificationError,
    extract_cfg_signature,
    is_cfg_verification_enabled,
    verify_modified_cfg,
)
feature_extractor = PEFeatureExtract()
action_table = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.keys())}


class Mal_Label_Env(gym.Env):

    def __init__(self):
        self.turn = 0
        self.maxturn = 300

    def _reset(self, file_name, file_path):

        """
            First step: Fetch the malicious file

            Second step: Evaluation of the original malware score

            Third step: Describe the state space of the original malware

        :return: observation of the original malware

        """
        self.seed()
        self.file_path = file_path
        self.file_name = file_name
        self.action_seq = []
        self.original_cfg_signature = None
        self.bytez = interface.fetch_file(self.file_name, self.file_path)
        self.label = interface.get_label_local(self.bytez)
        if self.label == 1:
            print("The {}'s original label is malicious".format(file_name))
        else:
            print("The {}'s original label is benign".format(file_name))
        self.obs = feature_extractor.extract(self.bytez)
        # Convert a one-dimensional array to a two-dimensional matrix
        for _ in range(48 * 48 - len(self.obs)):
            self.obs = np.append(self.obs, 0)
        self.obs = self.obs.reshape(48, 48)

        if not self._capture_original_cfg():
            return None

        return self.obs

    def _step(self, action_index, retrain):
        """
            First step: Make specified changes to the malware and return the changed malware

            Second step: Evaluation of the changed malware score and
            decide if it should be terminated (end of modification)

            Third step: Return the new observation,reward and done

        :param action_index:

        :return:

        """
        self.turn += 1
        self._take_action(action_index)
        self.label = interface.get_label_local(self.bytez)
        self.obs = feature_extractor.extract(self.bytez)
        # Convert a one-dimensional array to a two-dimensional matrix
        for _ in range(48 * 48 - len(self.obs)):
            self.obs = np.append(self.obs, 0)
        self.obs = self.obs.reshape(48, 48)

        if self.label == 0:
            episode_over = True
            if self._current_cfg_is_valid():
                reward = 10
                save_adv_malware(self.bytez, self.label, self.file_name)
            else:
                reward = -1

        elif self.turn > self.maxturn:
            reward = -1
            if retrain is False:
                copy(self.file_path, retrain_path)
            episode_over = True
        else:
            reward = -1
            episode_over = False

        if episode_over:
            if self.label == 1:
                print("episode is over! The {} malware's final label is malicious!".format(self.file_name))
            else:
                print("episode is over! The {} malware's final label is benign!".format(self.file_name))

        return self.obs, reward, episode_over, {}

    def _take_action(self, action_index):
        upper_action = action_index[0]
        lower_action = action_index[1]
        function = action_table[upper_action]
        print("The next modification for {} is {}".format(self.file_name, function))
        self.action_seq.append(function)
        self.bytez = eval(function)(self.bytez, lower_action)

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
