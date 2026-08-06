# SESG-Net: Intelligent Lithology Identification Based on Well-Log Sequence Encoding and a Stratigraphy-Constrained Graph Neural Network

Existing intelligent lithology identification methods often focus on only one aspect of the problem, such as pointwise well-log responses, vertical sequence information within individual wells, or spatial relationships among wells. As a result, they may not fully exploit the vertical continuity of well logs, spatial correlations among neighboring wells, and stratigraphic context simultaneously. To address this issue, SESG-Net adopts a dual-branch sequence-spatial feature fusion framework:
- a well-log sequence encoding module extracts vertical sequence features within individual wells;
- a stratigraphy-constrained spatial graph neural network extracts spatial correlation features from neighboring wells;
- the two feature types are fused for lithology classification.

## Method Overview

First, the raw well-log data are processed through missing-value indicators, missing-value imputation, logarithmic transformation, within-well standardization, and derived-feature construction to obtain consistent model inputs.

Next, a local well-log sequence window is constructed around each target depth sample. One-dimensional convolutions are used to encode combinations of well-log responses and their vertical variations, while attention pooling performs weighted aggregation across different depth positions to obtain the vertical sequence representation of the target sample.

For spatial feature modeling, stratigraphic units are used as constraints. Consecutive samples with similar well-log responses from the same well and the same stratigraphic unit are aggregated into supernodes. Candidate neighboring-well relationships are then constructed using the stratigraphic attributes, spatial locations, and well-log response statistics of the supernodes. Dynamic edge-weight learning and neighborhood aggregation are used to extract cross-well spatial features under stratigraphic constraints.

Finally, the vertical sequence features and neighboring-well spatial features are fused and passed through a classification layer to obtain lithology predictions. Model performance is evaluated using Accuracy, Precision, Recall, F1-score, and Penalty.

## Project Structure

```text
.
|-- train.py
|-- src/SESG_Net/
|   |-- config.py
|   |-- preprocessing.py
|   |-- dataset.py
|   |-- model.py
|   |-- training.py
|   `-- cli.py
|-- data/README.md
|-- requirements.txt
`-- pyproject.toml
```

## Usage

Install Python 3.9 or later and install the project dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

View available command-line arguments:

```bash
python train.py --help
```

Run training:

```bash
python train.py \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --penalty-matrix data/penalty_matrix.npy \
  --output-dir outputs/default
```
