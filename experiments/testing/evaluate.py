import json
import sys
from pathlib import Path
from typing import Dict, Optional

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from SESG_Net.config import ModelConfig, PipelineConfig, TrainConfig
from SESG_Net.preprocessing import build_training_artifacts
from SESG_Net.training import build_model, predict_model


def load_saved_configs(checkpoint: Path):
    metrics_path = checkpoint.parent / "metrics.json"
    if not metrics_path.exists():
        return PipelineConfig(), TrainConfig()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    training_payload = dict(payload.get("training_config", {}))
    model_payload = dict(training_payload.pop("model", {}))
    model_config = ModelConfig(**model_payload) if model_payload else ModelConfig()
    train_config = TrainConfig(model=model_config, **training_payload)
    pipeline_config = PipelineConfig(**payload.get("pipeline_config", {}))
    return pipeline_config, train_config


def run_test(
    train_csv: Path,
    test_csv: Path,
    checkpoint: Path,
    output_dir: Path,
    penalty_matrix_path: Optional[Path] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    train_config: Optional[TrainConfig] = None,
) -> Dict[str, float]:
    saved_pipeline_config, saved_train_config = load_saved_configs(checkpoint)
    pipeline_config = pipeline_config or saved_pipeline_config
    train_config = train_config or saved_train_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = build_training_artifacts(train_csv, test_csv, pipeline_config)
    model, _, _ = build_model(artifacts, train_config, device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    predictions = predict_model(model, artifacts, train_config, device)
    y_true = predictions["true"].to_numpy(np.int64)
    y_pred = predictions["pred"].to_numpy(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if penalty_matrix_path is not None and penalty_matrix_path.exists():
        penalty = np.load(penalty_matrix_path)
        metrics["penalty"] = float((-penalty[y_true, y_pred]).mean())
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    run_test(
        root / "data" / "train.csv",
        root / "data" / "test.csv",
        root / "outputs" / "reference" / "sesg_net_state_dict.pt",
        root / "outputs" / "test",
        root / "data" / "penalty_matrix.npy",
    )
