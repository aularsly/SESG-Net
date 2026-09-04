import copy
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

from SESG_Net.config import DIFFERENCE_COLUMNS, PipelineConfig, TrainConfig
from SESG_Net.dataset import SegmentLithologySequenceDataset, build_well_groups
from SESG_Net.preprocessing import TrainingArtifacts, build_training_artifacts
from SESG_Net.training import fit_global_standardizer, get_feature_columns, seed_everything


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index], torch.tensor(index)


class MLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 12),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class BiLSTM(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.classifier = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 12))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(values.transpose(1, 2))
        return self.classifier(encoded.mean(dim=1))


class TCN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 12))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(values).mean(dim=2))


class GAT(nn.Module):
    def __init__(self, input_dim: int, neighbors: torch.Tensor, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("neighbors", neighbors, persistent=False)
        self.register_buffer("mask", mask, persistent=False)
        self.projection = nn.Linear(input_dim, 64, bias=False)
        self.attention = nn.Linear(128, 1)
        self.classifier = nn.Linear(64, 12)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projection(features)
        candidate = projected[self.neighbors]
        source = projected.unsqueeze(1).expand_as(candidate)
        scores = torch.nn.functional.leaky_relu(self.attention(torch.cat([source, candidate], dim=-1)).squeeze(-1), 0.2)
        scores = scores.masked_fill(~self.mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * self.mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self.classifier(torch.relu(projected + (candidate * weights.unsqueeze(-1)).sum(dim=1)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, penalty: Optional[np.ndarray]) -> Dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if penalty is not None:
        result["penalty"] = float((-penalty[y_true, y_pred]).mean())
    return result


def _point_arrays(artifacts: TrainingArtifacts) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = artifacts.train_df
    val = artifacts.val_df
    columns = [
        column
        for column in train.columns
        if column.endswith(("_NORM", "_LOG", "_MISS")) or column in ("UMA", "VSH", "DEPTH_REL", "GROUP_ID", "FORMATION_ID")
    ]
    x_train = train[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    x_val = val[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    return x_train, train["target"].to_numpy(np.int64), x_val, val["target"].to_numpy(np.int64)


def _sequence_datasets(artifacts: TrainingArtifacts, window: int):
    train = artifacts.train_df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    val = artifacts.val_df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    continuous, missing = get_feature_columns(train)
    differences = [column for column in DIFFERENCE_COLUMNS if column in continuous]
    mean, std = fit_global_standardizer(train, continuous)
    train_dataset = SegmentLithologySequenceDataset(train, build_well_groups(train), np.arange(len(train)), continuous, missing, differences, mean, std, window)
    val_dataset = SegmentLithologySequenceDataset(val, build_well_groups(val), np.arange(len(val)), continuous, missing, differences, mean, std, window)
    return train_dataset, val_dataset


def _train_torch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_adapter: Callable,
    config: TrainConfig,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray, int]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            values, target, _ = batch_adapter(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        y_true, y_pred, _ = _predict_torch(model, val_loader, batch_adapter, device)
        accuracy = float(accuracy_score(y_true, y_pred))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No model state was selected")
    model.load_state_dict(best_state)
    y_true, y_pred, _ = _predict_torch(model, val_loader, batch_adapter, device)
    return best_state, y_true, y_pred, best_epoch


def _predict_torch(model: nn.Module, loader: DataLoader, batch_adapter: Callable, device: torch.device):
    model.eval()
    true_batches = []
    prediction_batches = []
    index_batches = []
    with torch.no_grad():
        for batch in loader:
            values, target, indices = batch_adapter(batch, device)
            prediction_batches.append(model(values).argmax(dim=1).cpu().numpy())
            true_batches.append(target.cpu().numpy())
            index_batches.append(indices.cpu().numpy())
    return np.concatenate(true_batches), np.concatenate(prediction_batches), np.concatenate(index_batches)


def _point_adapter(batch, device):
    values, target, indices = batch
    return values.to(device), target.to(device), indices


def _sequence_adapter(batch, device):
    sequence, missing, target, _, _, _, indices = batch
    values = torch.cat([sequence, missing], dim=1).to(device)
    return values, target.to(device), indices


def _save_result(output_dir: Path, name: str, model_state, y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float]) -> None:
    torch.save(model_state, output_dir / f"{name}_model.pt")
    pd.DataFrame({"true": y_true, "pred": y_pred}).to_csv(output_dir / f"{name}_predictions.csv", index=False)
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def run_baselines(
    train_csv: Path,
    val_csv: Path,
    output_dir: Path,
    penalty_matrix_path: Optional[Path] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    train_config: Optional[TrainConfig] = None,
) -> Dict[str, Dict[str, float]]:
    pipeline_config = pipeline_config or PipelineConfig()
    train_config = train_config or TrainConfig()
    seed_everything(train_config.seed)
    random.seed(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = build_training_artifacts(train_csv, val_csv, pipeline_config)
    penalty = np.load(penalty_matrix_path) if penalty_matrix_path is not None and penalty_matrix_path.exists() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_val, y_val = _point_arrays(artifacts)
    summary = {}
    estimators = {
        "RF": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=train_config.seed),
        "XGBoost": XGBClassifier(n_estimators=350, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, tree_method="hist", objective="multi:softprob", num_class=12, eval_metric="mlogloss", n_jobs=-1, random_state=train_config.seed),
        "CatBoost": CatBoostClassifier(iterations=400, depth=8, learning_rate=0.05, loss_function="MultiClass", auto_class_weights="Balanced", random_seed=train_config.seed, verbose=False),
    }
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        predictions = np.asarray(estimator.predict(x_val)).reshape(-1).astype(np.int64)
        metrics = _metrics(y_val, predictions, penalty)
        joblib.dump(estimator, output_dir / f"{name}_model.joblib")
        pd.DataFrame({"true": y_val, "pred": predictions}).to_csv(output_dir / f"{name}_predictions.csv", index=False)
        (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[name] = metrics
    point_train = DataLoader(ArrayDataset(x_train, y_train), batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers)
    point_val = DataLoader(ArrayDataset(x_val, y_val), batch_size=train_config.batch_size, shuffle=False, num_workers=train_config.num_workers)
    state, true, pred, epoch = _train_torch(MLP(x_train.shape[1]), point_train, point_val, _point_adapter, train_config, device)
    summary["MLP"] = {**_metrics(true, pred, penalty), "best_epoch": epoch}
    _save_result(output_dir, "MLP", state, true, pred, summary["MLP"])
    train_sequence, val_sequence = _sequence_datasets(artifacts, train_config.model.window)
    sequence_train = DataLoader(train_sequence, batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers)
    sequence_val = DataLoader(val_sequence, batch_size=train_config.batch_size, shuffle=False, num_workers=train_config.num_workers)
    input_dim = train_sequence.main_dim + train_sequence.missing_dim
    for name, model in (("BiLSTM", BiLSTM(input_dim)), ("TCN", TCN(input_dim))):
        state, true, pred, epoch = _train_torch(model, sequence_train, sequence_val, _sequence_adapter, train_config, device)
        summary[name] = {**_metrics(true, pred, penalty), "best_epoch": epoch}
        _save_result(output_dir, name, state, true, pred, summary[name])
    segment_features = artifacts.segments[artifacts.segment_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    train_mask = artifacts.segments["TRAIN_POINTS"].to_numpy() > 0
    node_targets = np.zeros(len(artifacts.segments), dtype=np.int64)
    for segment_id, group in artifacts.train_df.groupby("SEGMENT_ID"):
        node_targets[int(segment_id)] = int(group["target"].mode().iloc[0])
    gat = GAT(segment_features.shape[1], torch.tensor(artifacts.candidate_neighbors, dtype=torch.long), torch.tensor(artifacts.candidate_mask, dtype=torch.bool)).to(device)
    features = torch.tensor(segment_features, dtype=torch.float32, device=device)
    targets = torch.tensor(node_targets, dtype=torch.long, device=device)
    mask = torch.tensor(train_mask, dtype=torch.bool, device=device)
    optimizer = torch.optim.AdamW(gat.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.epochs)
    best_state = None
    best_accuracy = -1.0
    best_epoch = 0
    val_segment_ids = artifacts.val_df["SEGMENT_ID"].to_numpy(np.int64)
    for epoch in range(1, train_config.epochs + 1):
        gat.train()
        optimizer.zero_grad(set_to_none=True)
        logits = gat(features)
        loss = nn.CrossEntropyLoss()(logits[mask], targets[mask])
        loss.backward()
        nn.utils.clip_grad_norm_(gat.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        gat.eval()
        with torch.no_grad():
            predictions = gat(features).argmax(dim=1).cpu().numpy()[val_segment_ids]
        accuracy = float(accuracy_score(y_val, predictions))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in gat.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No GAT model state was selected")
    gat.load_state_dict(best_state)
    gat.eval()
    with torch.no_grad():
        predictions = gat(features).argmax(dim=1).cpu().numpy()[val_segment_ids]
    summary["GAT"] = {**_metrics(y_val, predictions, penalty), "best_epoch": best_epoch}
    _save_result(output_dir, "GAT", best_state, y_val, predictions, summary["GAT"])
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    run_baselines(root / "data" / "train.csv", root / "data" / "val.csv", root / "outputs" / "baselines", root / "data" / "penalty_matrix.npy")
