import numpy as np
import torch
from pathlib import Path
import gzip
import joblib

from envs.utils.manipulator_pefeatures import PEFeatureExtract

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_DIR = PROJECT_ROOT / "models" / "classifiers"

# GBDT
class_extractor = PEFeatureExtract()
local_model = joblib.load(CLASSIFIER_DIR / 'gradient_boosting.pkl')
local_model_threshold = 0.90

# LGBM, not used
# lgbm_extractor = PEFeatureExtractor()
# lgbm_model = joblib.load(CLASSIFIER_DIR / 'lgbm_classifier.pkl')
# lgbm_model_threshold = 0.90

# Malconv
# malconv_model = MalConv()
# malconv_model.load_state_dict(torch.load(CLASSIFIER_DIR / 'pretrained_malconv.pth'))
# malconv_model_threshold = 0.50
# Malconv_model = torch.load(CLASSIFIER_DIR / 'Malconv.pth')


class FileRetrievalFailure(Exception):
    pass


def fetch_file(file_name, file_path):

    try:
        with open(file_path, 'rb') as infile:
            bytez = infile.read()
    except IOError:
        raise FileRetrievalFailure(
            "Unable to read {} from {}".format(file_name, file_path))

    '''
    try:
        with gzip.open(file_path, 'rb') as f:
            bytez = f.read()
    except OSError:
        # OK, you are not a gziped file. Just read in raw bytes from disk.
        with open(file_path, 'rb') as f:
            bytez = f.read()
    '''
    return bytez


def get_rl_obs(bytez):

    RL_OBS_INDICES = np.array([0, 1, 2, 7, 33, 47, 49, 436, 487, 515, 529, 531, 532,
                           551, 562, 563, 578, 582, 586, 588, 613, 619, 625,
                           676, 678, 734, 784], dtype=np.int64)

    return np.array(class_extractor.extract(bytez), dtype=np.float32)[RL_OBS_INDICES]


def get_score_local(bytez):
    # for machine-learning classifiers
    # GBDT
    # extract features
    features = class_extractor.extract(bytez)
    # query the model
    score = local_model.predict_proba(features.reshape(1, -1))[0, -1]
    # predict on single sample, get the malicious score

    # LGBM
    # extract features
    # features = lgbm_extractor.extract(bytez)
    # query the model
    # score = lgbm_model.predict_proba(features.reshape(1, -1))[0, -1]
    # predict on single sample, get the malicious score

    # for deep-learning classifiers
    # score = malconv_model(bytez).detach().cpu().numpy()[0][0]
    return score


def get_label_local(bytez):
    # mimic black box by thresholding here
    score = get_score_local(bytez)
    label = float(get_score_local(bytez) >= local_model_threshold)
    print("score={} (hidden), label={}".format(score, label))
    return label
