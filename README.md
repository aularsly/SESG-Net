# SESG-Net

SESG-Net is a lithology identification framework that integrates **well-log sequence information, inter-well spatial relationships, and stratigraphic constraints**.

This repository provides the implementation for data preprocessing, model training, baseline comparison, ablation studies, and model evaluation, supporting reproducibility of the experiments reported in the accompanying study.

## Repository Structure

```text
.
|-- data/
|   `-- README.md
|-- docs/
|   `-- method_alignment.md
|-- experiments/
|   |-- ablations/component_ablation.py
|   |-- baselines/compare_models.py
|   `-- testing/evaluate.py
|-- src/SESG_Net/
|   |-- __init__.py
|   |-- cli.py
|   |-- config.py
|   |-- dataset.py
|   |-- model.py
|   |-- preprocessing.py
|   `-- training.py
|-- train.py
|-- requirements.txt
`-- pyproject.toml
```

Main files and modules are organized as follows:

| Path                                          | Description                                     |
| --------------------------------------------- | ----------------------------------------------- |
| `train.py`                                    | Main entry point for SESG-Net training          |
| `src/SESG_Net/config.py`                      | Dataset, model, and training configurations     |
| `src/SESG_Net/preprocessing.py`               | Data preprocessing and graph construction       |
| `src/SESG_Net/dataset.py`                     | Construction of well-log sequence windows       |
| `src/SESG_Net/model.py`                       | Implementation of the SESG-Net architecture     |
| `src/SESG_Net/training.py`                    | Training, prediction, and evaluation procedures |
| `experiments/baselines/compare_models.py`     | Baseline comparison experiments                 |
| `experiments/ablations/component_ablation.py` | Component ablation experiments                  |
| `experiments/testing/evaluate.py`             | Evaluation on the test set                      |
| `data/README.md`                              | Dataset source and input data description       |

## Installation

Python 3.9 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For GPU acceleration, install a PyTorch version compatible with the local CUDA environment. The code can also be executed on CPU.

## Data Preparation

The experiments are based on the **FORCE 2020 Well Log Lithology Dataset**.

Detailed information about data acquisition and file preparation is provided in:

```text
data/README.md
```

The processed dataset should be organized as follows:

```text
data/
|-- train.csv
|-- val.csv
|-- test.csv
`-- penalty_matrix.npy
```

where:

* `train.csv`: training dataset
* `val.csv`: validation dataset
* `test.csv`: test dataset
* `penalty_matrix.npy`: lithology misclassification penalty matrix

The original dataset is not included in this repository.

## Model Training

Run the following command from the repository root:

```bash
python train.py \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --penalty-matrix data/penalty_matrix.npy \
  --output-dir outputs/reference
```

Model settings and training parameters can be found in:

```text
src/SESG_Net/config.py
src/SESG_Net/cli.py
```

After training, the main outputs are saved to:

```text
outputs/reference/
|-- sesg_net_state_dict.pt
|-- validation_predictions.csv
`-- metrics.json
```

## Baseline Comparison

To evaluate the comparative models, including RF, XGBoost, CatBoost, MLP, BiLSTM, TCN, and GAT, run:

```bash
python experiments/baselines/compare_models.py
```

The corresponding results are saved in:

```text
outputs/baselines/
```

## Ablation Studies

Component ablation experiments can be executed using:

```bash
python experiments/ablations/component_ablation.py
```

The experiments include the following configurations:

* sequence information only
* spatial information only
* spatial information with stratigraphic constraints
* combined sequence and spatial information

The results are saved in:

```text
outputs/ablations/
```

## Evaluation

After training the complete SESG-Net model, evaluate it on the test set using:

```bash
python experiments/testing/evaluate.py
```

The evaluation results are saved in:

```text
outputs/test/
```

Available training options can be inspected with:

```bash
python -B train.py --help
```
