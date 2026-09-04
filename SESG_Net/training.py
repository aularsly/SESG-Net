import copy
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .config import DIFFERENCE_COLUMNS, SEQUENCE_CONTINUOUS_COLUMNS, TrainConfig
from .dataset import SegmentLithologySequenceDataset, build_well_groups
from .model import SegmentSpatialTemporalStratModel
from .preprocessing import TrainingArtifacts


LOGGER = logging.getLogger(__name__)
PathLike = Union[str, Path]


@dataclass
class TrainingResult:
    model: SegmentSpatialTemporalStratModel
    best_epoch: int
    metrics: Dict[str, Optional[float]]
    classification_report: str
    predictions: pd.DataFrame
    state_dict: Dict[str, torch.Tensor]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def penalty_score(y_true: np.ndarray, y_pred: np.ndarray, matrix: np.ndarray) -> float:
    return float((-matrix[y_true.astype(int), y_pred.astype(int)]).mean())


def get_feature_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    continuous = [column for column in SEQUENCE_CONTINUOUS_COLUMNS if column in frame]
    missing = [column for column in frame.columns if column.endswith("_MISS")]
    if not continuous:
        raise ValueError("No sequence features were produced by preprocessing")
    return continuous, missing


def fit_global_standardizer(
    train_df: pd.DataFrame,
    continuous_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_df[continuous_columns].mean().to_numpy(dtype=np.float32)
    std = train_df[continuous_columns].std().to_numpy(dtype=np.float32)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)
    return mean, std


def build_segment_feature_matrix(
    segments: pd.DataFrame,
    feature_columns: List[str],
) -> np.ndarray:
    matrix = (
        segments[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    fit_mask = segments["TRAIN_POINTS"].to_numpy() > 0
    if not fit_mask.any():
        raise ValueError("No training supernodes are available")
    mean = matrix[fit_mask].mean(axis=0, keepdims=True)
    std = matrix[fit_mask].std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return ((matrix - mean) / (std + 1e-6)).astype(np.float32)


def prepare_context(
    artifacts: TrainingArtifacts,
    device: torch.device,
) -> Dict[str, object]:
    train_df = artifacts.train_df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    val_df = artifacts.val_df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    segments = artifacts.segments.sort_values("SEGMENT_ID").reset_index(drop=True)

    continuous_columns, missing_columns = get_feature_columns(train_df)
    difference_columns = [
        column for column in DIFFERENCE_COLUMNS if column in continuous_columns
    ]
    mean, std = fit_global_standardizer(train_df, continuous_columns)
    segment_features = build_segment_feature_matrix(
        segments,
        artifacts.segment_feature_columns,
    )

    num_groups = int(
        max(train_df["GROUP_ID"].max(), val_df["GROUP_ID"].max(), segments["GROUP_ID_MODE"].max())
    ) + 1
    num_formations = int(
        max(
            train_df["FORMATION_ID"].max(),
            val_df["FORMATION_ID"].max(),
            segments["FORMATION_ID_MODE"].max(),
        )
    ) + 1

    return {
        "train_df": train_df,
        "val_df": val_df,
        "train_well_groups": build_well_groups(train_df),
        "val_well_groups": build_well_groups(val_df),
        "train_indices": np.arange(len(train_df), dtype=np.int64),
        "val_indices": np.arange(len(val_df), dtype=np.int64),
        "continuous_columns": continuous_columns,
        "missing_columns": missing_columns,
        "difference_columns": difference_columns,
        "mean": mean,
        "std": std,
        "segment_features": torch.tensor(segment_features, dtype=torch.float32, device=device),
        "candidate_neighbors": torch.tensor(
            artifacts.candidate_neighbors,
            dtype=torch.long,
            device=device,
        ),
        "candidate_mask": torch.tensor(
            artifacts.candidate_mask,
            dtype=torch.bool,
            device=device,
        ),
        "pair_features": torch.tensor(
            artifacts.pair_features,
            dtype=torch.float32,
            device=device,
        ),
        "segment_group_ids": torch.tensor(
            segments["GROUP_ID_MODE"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=device,
        ),
        "segment_formation_ids": torch.tensor(
            segments["FORMATION_ID_MODE"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=device,
        ),
        "num_groups": num_groups,
        "num_formations": num_formations,
    }


def _is_better(
    accuracy: float,
    penalty: Optional[float],
    best_accuracy: float,
    best_penalty: Optional[float],
) -> bool:
    if accuracy > best_accuracy + 1e-8:
        return True
    if abs(accuracy - best_accuracy) > 1e-8 or penalty is None:
        return False
    return best_penalty is None or penalty > best_penalty + 1e-8


def _make_dataset(
    context: Dict[str, object],
    split: str,
    window: int,
) -> SegmentLithologySequenceDataset:
    return SegmentLithologySequenceDataset(
        frame=context[f"{split}_df"],
        well_groups=context[f"{split}_well_groups"],
        indices=context[f"{split}_indices"],
        continuous_columns=context["continuous_columns"],
        missing_columns=context["missing_columns"],
        difference_columns=context["difference_columns"],
        mean=context["mean"],
        std=context["std"],
        window=window,
    )


def build_model(
    artifacts: TrainingArtifacts,
    config: TrainConfig,
    device: torch.device,
):
    context = prepare_context(artifacts, device)
    model_config = config.model
    train_dataset = _make_dataset(context, "train", model_config.window)
    model = SegmentSpatialTemporalStratModel(
        main_dim=train_dataset.main_dim,
        missing_dim=train_dataset.missing_dim,
        num_classes=model_config.num_classes,
        segment_features=context["segment_features"],
        candidate_neighbors=context["candidate_neighbors"],
        candidate_mask=context["candidate_mask"],
        pair_features=context["pair_features"],
        segment_group_ids=context["segment_group_ids"],
        segment_formation_ids=context["segment_formation_ids"],
        num_groups=context["num_groups"],
        num_formations=context["num_formations"],
        spatial_dim=model_config.spatial_dim,
        group_embedding_dim=model_config.group_embedding_dim,
        formation_embedding_dim=model_config.formation_embedding_dim,
        base_out=model_config.base_out,
        depth=model_config.depth,
        kernel_sizes=model_config.kernel_sizes,
        bottleneck_channels=model_config.bottleneck_channels,
        dropout=model_config.dropout,
        missing_branch_dim=model_config.missing_branch_dim,
        edge_hidden_dim=model_config.edge_hidden_dim,
        final_neighbor_k=model_config.final_neighbor_k,
        use_sequence=model_config.use_sequence,
        use_spatial=model_config.use_spatial,
        use_stratigraphy=model_config.use_stratigraphy,
    ).to(device)
    return model, context, train_dataset


def train_model(
    artifacts: TrainingArtifacts,
    config: Optional[TrainConfig] = None,
    penalty_matrix: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> TrainingResult:
    config = config or TrainConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model, context, train_dataset = build_model(artifacts, config, device)
    model_config = config.model
    val_dataset = _make_dataset(context, "val", model_config.window)
    generator = torch.Generator().manual_seed(config.seed)
    persistent_workers = config.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        persistent_workers=persistent_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        persistent_workers=persistent_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_accuracy = -1.0
    best_penalty: Optional[float] = None
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        model.train()
        running_loss = 0.0
        for sequence, missing, target, group, formation, segment, _ in train_loader:
            sequence = sequence.to(device, non_blocking=True)
            missing = missing.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            group = group.to(device, non_blocking=True)
            formation = formation.to(device, non_blocking=True)
            segment = segment.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(sequence, missing, group, formation, segment)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

        y_true, y_pred = _predict(model, val_loader, device, amp_enabled)
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        penalty = (
            penalty_score(y_true, y_pred, penalty_matrix)
            if penalty_matrix is not None
            else None
        )
        if _is_better(accuracy, penalty, best_accuracy, best_penalty):
            best_accuracy = accuracy
            best_penalty = penalty
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        LOGGER.info(
            "Epoch %d/%d loss=%.4f val_accuracy=%.4f macro_f1=%.4f penalty=%s time=%.1fs",
            epoch,
            config.epochs,
            running_loss / max(len(train_loader), 1),
            accuracy,
            macro_f1,
            "n/a" if penalty is None else f"{penalty:.4f}",
            time.time() - started,
        )
        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training completed without a model state")
    model.load_state_dict(best_state)
    y_true, y_pred, row_indices = _predict(
        model,
        val_loader,
        device,
        amp_enabled,
        return_indices=True,
    )
    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "penalty": (
            penalty_score(y_true, y_pred, penalty_matrix)
            if penalty_matrix is not None
            else None
        ),
    }
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    predictions = pd.DataFrame(
        {"row_index": row_indices, "true": y_true, "pred": y_pred}
    )
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        metrics=metrics,
        classification_report=report,
        predictions=predictions,
        state_dict=best_state,
    )


def _predict(
    model: SegmentSpatialTemporalStratModel,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    return_indices: bool = False,
):
    model.eval()
    true_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []
    index_batches: List[np.ndarray] = []
    with torch.no_grad():
        for sequence, missing, target, group, formation, segment, indices in loader:
            sequence = sequence.to(device, non_blocking=True)
            missing = missing.to(device, non_blocking=True)
            group = group.to(device, non_blocking=True)
            formation = formation.to(device, non_blocking=True)
            segment = segment.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(sequence, missing, group, formation, segment)
            prediction_batches.append(torch.argmax(logits, dim=1).cpu().numpy())
            true_batches.append(target.numpy())
            index_batches.append(indices.numpy())

    y_true = np.concatenate(true_batches).astype(int)
    y_pred = np.concatenate(prediction_batches).astype(int)
    if return_indices:
        return y_true, y_pred, np.concatenate(index_batches).astype(int)
    return y_true, y_pred


def predict_model(
    model: SegmentSpatialTemporalStratModel,
    artifacts: TrainingArtifacts,
    config: TrainConfig,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = prepare_context(artifacts, device)
    dataset = _make_dataset(context, "val", config.model.window)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    model.to(device)
    y_true, y_pred, indices = _predict(
        model,
        loader,
        device,
        config.use_amp and device.type == "cuda",
        return_indices=True,
    )
    return pd.DataFrame({"row_index": indices, "true": y_true, "pred": y_pred})


def save_training_result(
    result: TrainingResult,
    output_dir: PathLike,
    config: TrainConfig,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result.state_dict, output / "sesg_net_state_dict.pt")
    payload = {
        "best_epoch": result.best_epoch,
        "metrics": result.metrics,
        "training_config": asdict(config),
    }
    payload["classification_report"] = result.classification_report
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.predictions.to_csv(output / "validation_predictions.csv", index=False)
