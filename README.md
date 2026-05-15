# RLAEG & MPDAEG: Functionality-Verification Adversarial Malware Attack Framework

> Official implementation of **IEEE TIFS 2024** paper: *Functionality-Verification Attack Framework Based on Reinforcement Learning Against Static Malware Detectors*

This repository provides a reproducible implementation of reinforcement-learning-based adversarial malware generation frameworks: **RLAEG (RL-based Adversarial Example Generation)** and **MPDAEG (Multi-task Policy Distillation-based Adversarial Example Generation)**. The framework targets black-box static malware detectors and supports PE-file manipulation, detector feedback, expert-agent training, and multi-task policy distillation.

This code is released for educational, reproducibility, and defensive security research only. Use it in isolated environments and only with binaries that you are authorized to analyze.

## System Requirements

- Linux recommended for large-scale experiments
- Python 3.6.3 reference environment; Python >= 3.7 is recommended if dependency compatibility allows
- CUDA-capable GPU recommended for training neural policies and MalConv-style models
- UPX installed and available on `PATH` for UPX packing/unpacking actions
- `angr` available for CFG-based functionality verification
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

CFG-based functionality verification is enabled by default. When a generated sample evades a victim detector, RLAEG extracts the original and modified CFGs with `angr` and saves the sample only if the two CFG signatures are identical. Set `RLAEG_ENABLE_CFG_CHECK=0` only when you intentionally want to reproduce the detector-only attack baseline.

## 📦 Repository Contents

`agents/`: SAC expert agent and distilled student policy implementations

`envs/`: Gym environments for malware-score and malware-label attack settings

`envs/controls/`: PE manipulation actions used to generate adversarial variants

`envs/utils/`: Feature extraction, classifier interface, and result-saving utilities

`functionality_verification/`: CFG extraction and comparison utilities for validating generated samples

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

Note: This research is for educational and defensive security purposes only.
