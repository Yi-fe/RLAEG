# RLAEG

RLAEG is a reinforcement-learning framework for generating adversarial malware samples against malware classifiers. It provides Gym environments for score-based and label-based malware evasion, SAC training code, and a multi-task policy-distillation workflow for distilling several teacher policies into one student policy.

This repository is intended for authorized malware-research experiments only. Use isolated analysis environments and only work with files you are allowed to inspect and modify.

## Repository Layout

```text
RLAEG/
|-- agents/
|   |-- agent.py                 # SAC teacher/attack agent
|   |-- Student_net.py           # Student DQN used by policy distillation
|   `-- common/                  # Shared model, replay buffer, plotting utilities
|-- envs/
|   |-- env/                     # Gym malware-score and malware-label environments
|   |-- controls/                # PE manipulation actions
|   `-- utils/                   # Feature extraction, classifier interface, result saving
|-- models/
|   |-- classifiers/             # Pretrained classifier artifacts
|   `-- MalConv2_main/           # MalConv model implementations and helpers
|-- scripts/
|   |-- main_func.py             # SAC training/evaluation entry point
|   |-- Train_test.py            # Shared SAC train/test loops
|   |-- PD_malware.py            # Multi-task policy distillation
|   `-- make_results.py          # CSV result writer
`-- data/
    `-- section_names.txt        # Common PE section names
```

Malware samples are not included. Create the expected local data folders before running experiments.

## Environment

Python 3.8+ is recommended. The code was syntax-checked with Python 3.10.

Install the main Python dependencies:

```bash
pip install numpy torch gym tensorboardX matplotlib seaborn scikit-learn joblib pefile lief tqdm
```

Some PE actions call the external `upx` executable. Install UPX and make sure it is available on `PATH` if you use packing or unpacking actions.

## Data And Models

Expected sample directories:

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

The default classifier artifacts are loaded from:

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

## Running SAC Training

The SAC training entry point is:

```bash
python scripts/main_func.py --env_name malware-score-v0
```

The registered Gym environments are:

```text
malware-score-v0
malware-label-v0
```

Outputs are written under `scripts/outputs/<env>/<timestamp>/`.

## Running Policy Distillation

`scripts/PD_malware.py` trains a student policy from multiple task-specific teacher policies. By default it expects teacher checkpoints in `scripts/teacher_models/`; override that with `RLAEG_TEACHER_MODEL_PATH` or `--teacher_model_path`.

```bash
python scripts/PD_malware.py \
  --env_name malware-score-v0 \
  --training_malicious_path data/Train_Malicious \
  --teacher_model_path scripts/teacher_models/
```

Important options:

```text
--task_count       Number of teacher/student task heads, default 5
--file_groups      Number of malware file groups, default 9
--max_turn         Maximum modification turns per sample, default 10
--update_steps     Student update steps after each teacher rollout, default 500
```

## Validation

A lightweight syntax check for every Python file:

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

Before publishing, remove generated files such as `__pycache__/`, experiment outputs, TensorBoard logs, and generated adversarial samples unless you intentionally want to release them.

## Notes

- The project uses Gym registration from `envs/__init__.py`.
- `envs/utils/interface.py` loads the gradient-boosting classifier at import time.
- LIEF is required by PE feature extraction and malware environment modules.
- UPX-dependent actions fail gracefully only when UPX is installed and reachable.
- Keep malware samples and generated adversarial binaries out of public commits unless your release policy explicitly allows them.
