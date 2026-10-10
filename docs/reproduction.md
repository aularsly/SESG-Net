# Reproduction guide

## Environment

The reference environment uses Python 3.8, PyTorch 2.1.2, NumPy 1.24.3, pandas 2.0.3, scikit-learn 1.2.2, XGBoost 2.1.4, CatBoost 1.2.8, and joblib 1.4.2.

## Data preparation

Prepare `train.csv`, `val.csv`, `test.csv`, and `penalty_matrix.npy` according to `data/README.md`. Use the fixed well-level assignment in `data/split_manifest.csv`: 98 wells for training, 10 for validation, and 10 for blind testing. Keep this assignment unchanged for the main model, baseline models, ablation experiments, and test evaluation.

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

## Paper experiment mapping

| Paper result | Repository entry point |
|---|---|
| SESG-Net configuration in Table 1, SESG-Net result in Table 2, and Group 6 in Table 3 | `train.py` |
| RF, XGBoost, CatBoost, MLP, BiLSTM, TCN, and GAT results in Table 2 | `experiments/baselines/compare_models.py` |
| Groups 1-6 in Table 3 | `experiments/ablations/component_ablation.py` |
| Blind-well evaluation and the predictions used in Figure 14 | `experiments/testing/evaluate.py` |

The ablation configuration names correspond to Table 3 as follows: `sequence_only` is Group 1, `spatial_only` is Group 2, `spatial_stratigraphy` is Group 3, `sequence_spatial` is Group 4, `sesg_net_without_missing` is Group 5, and `sesg_net` is Group 6.

## Expected files

Each main run contains a checkpoint, predictions, metrics, training configuration, preprocessing configuration, and preprocessing statistics. Baseline and ablation directories contain model-specific predictions and metrics. Test evaluation contains `test_predictions.csv` and `test_metrics.json`.
