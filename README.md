# RLAEG & MPDAEG: Functionality-Verification Adversarial Malware Attack Framework

> Official implementation of **IEEE TIFS 2024** paper: *Functionality-Verification Attack Framework Based on Reinforcement Learning Against Static Malware Detectors*

This repository provides a reproducible implementation of reinforcement-learning-based adversarial malware generation frameworks: **RLAEG (RL-based Adversarial Example Generation)** and **MPDAEG (Multi-task Policy Distillation-based Adversarial Example Generation)**. The framework targets black-box static malware detectors and supports PE-file manipulation, detector feedback, expert-agent training, and multi-task policy distillation.

This code is released for educational, reproducibility, and defensive security research only. Use it in isolated environments and only with binaries that you are authorized to analyze.

## System Requirements

- Linux recommended for large-scale experiments
- Python 3.6.3 reference environment; Python >= 3.7 is recommended if dependency compatibility allows
- CUDA-capable GPU recommended for training neural policies and MalConv-style models
- UPX installed and available on `PATH` for UPX packing/unpacking actions
- Windows PE samples for malware-evasion experiments

## Installation

Create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all commands from the repository root so package imports resolve consistently. If you use CUDA, install the PyTorch build that matches your CUDA version from the official PyTorch instructions, then install the remaining dependencies from `requirements.txt`.

## Data And Model Setup

Malware samples are not included in this repository. Prepare local directories such as:

```text
data/
|-- Train_Malicious/
|-- Test_Malicious/
|-- Pure_Benign/
|-- label_adv_malware/
|-- score_adv_malware/
|-- label_adv_test_malware/
`-- score_adv_test_malware/
```

Default classifier artifacts are expected under:

```text
models/classifiers/
|-- gradient_boosting.pkl
|-- pretrained_malconv.pth
`-- AvastConv.pth
```

Useful environment variables:

```bash
RLAEG_TRAINING_MALWARE_DIR=/path/to/train_samples
RLAEG_TESTING_MALWARE_DIR=/path/to/test_samples
RLAEG_PURE_BENIGN_DIR=/path/to/benign_samples
RLAEG_TEACHER_MODEL_PATH=/path/to/teacher_models/
RLAEG_SECTION_NAMES_PATH=/path/to/section_names.txt
RLAEG_IMPORT_API_PATH=/path/to/dll_imports.json
RLAEG_ADV_LABEL_MALWARE_DIR=/path/to/save/label_adv_malware
RLAEG_ADV_SCORE_MALWARE_DIR=/path/to/save/score_adv_malware
RLAEG_ADV_LABEL_TEST_MALWARE_DIR=/path/to/save/label_adv_test_malware
RLAEG_ADV_SCORE_TEST_MALWARE_DIR=/path/to/save/score_adv_test_malware
```

## 📦 Repository Contents

`agents/`: SAC expert agent and distilled student policy implementations

`envs/`: Gym environments for malware-score and malware-label attack settings

`envs/controls/`: PE manipulation actions used to generate adversarial variants

`envs/utils/`: Feature extraction, classifier interface, and result-saving utilities

`models/`: Victim detector components, MalConv variants, and classifier artifacts

`models/MalConv2_main/`: MalConv-GCT and low-memory MalConv model implementations

`scripts/`: Training, evaluation, and multi-task policy distillation entry points

`data/`: Lightweight metadata files and expected local dataset folder structure

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Yi-fe/RLAEG.git
cd RLAEG

# Create and activate an environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run SAC-based RLAEG training against the score-based malware environment
python scripts/main_func.py --env_name malware-score-v0

# Run multi-task policy distillation with pretrained expert agents
python scripts/PD_malware.py \
  --env_name malware-score-v0 \
  --training_malicious_path data/Train_Malicious \
  --teacher_model_path scripts/teacher_models/
```

Registered Gym environments:

```text
malware-score-v0
malware-label-v0
```

## Validation

Run a lightweight syntax check for every Python file:

```bash
python -B - <<'PY'
import ast
import pathlib

for path in sorted(pathlib.Path('.').rglob('*.py')):
    if '__pycache__' not in path.parts:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print('AST parse OK')
PY
```

Before publishing or archiving results, remove generated files such as `__pycache__/`, experiment outputs, TensorBoard logs, local malware samples, and generated adversarial binaries unless they are intentionally part of the release.

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{tian2024functionality,
  title={Functionality-Verification Attack Framework Based on Reinforcement Learning Against Static Malware Detectors},
  author={Tian, Buwei and Jiang, Junyong and He, Zichen and Yuan, Xin and Dong, Lu and Sun, Changyin},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={19},
  pages={8500--8514},
  year={2024},
  publisher={IEEE},
  doi={10.1109/TIFS.2024.3453047}
}
```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

Note: This research is for educational and defensive security purposes only. The authors are not responsible for any misuse of this code.
