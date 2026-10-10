# Reproduction guide

## Environment

The reference environment uses Python 3.8, PyTorch 2.1.2, NumPy 1.24.3, pandas 2.0.3, scikit-learn 1.2.2, XGBoost 2.1.4, CatBoost 1.2.8, and joblib 1.4.2.

## Data preparation

Prepare `train.csv`, `val.csv`, `test.csv`, and `penalty_matrix.npy` according to `data/README.md`. The split must be performed at the well level. Keep the same well lists for the main model, baseline models, ablation experiments, and test evaluation.

## Main model

```bash
python train.py \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --penalty-matrix data/penalty_matrix.npy \
  --output-dir outputs/reference
```

The checkpoint is selected using validation Accuracy. The test set must not be used for model selection, epoch selection, seed selection, or parameter selection.

## Baselines

```bash
python experiments/baselines/compare_models.py
```

All baseline models use the same training and validation wells and the same class mapping.

## Ablations

```bash
python experiments/ablations/component_ablation.py
```

The experiment writes one directory per configuration and a combined `summary_metrics.json`. Configurations without stratigraphic constraints rebuild both supernodes and graph candidates without group or formation restrictions.

## Test set

```bash
python experiments/testing/evaluate.py
```

The evaluator loads the selected checkpoint and its saved configuration, rebuilds preprocessing from the training wells, and evaluates the fixed model on the test wells.

## Expected files

Each main run contains a checkpoint, predictions, metrics, training configuration, preprocessing configuration, and preprocessing statistics. Baseline and ablation directories contain model-specific predictions and metrics. Test evaluation contains `test_predictions.csv` and `test_metrics.json`.
