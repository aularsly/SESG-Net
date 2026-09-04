import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Optional

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from SESG_Net.config import ModelConfig, PipelineConfig, TrainConfig
from SESG_Net.preprocessing import build_training_artifacts
from SESG_Net.training import save_training_result, train_model


VARIANTS = {
    "sequence_only": {"use_sequence": True, "use_spatial": False, "use_stratigraphy": False},
    "spatial_only": {"use_sequence": False, "use_spatial": True, "use_stratigraphy": False},
    "spatial_stratigraphy": {"use_sequence": False, "use_spatial": True, "use_stratigraphy": True},
    "sequence_spatial": {"use_sequence": True, "use_spatial": True, "use_stratigraphy": False},
}


def run_ablations(
    train_csv: Path,
    val_csv: Path,
    output_dir: Path,
    penalty_matrix_path: Optional[Path] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    train_config: Optional[TrainConfig] = None,
) -> Dict[str, Dict[str, float]]:
    pipeline_config = pipeline_config or PipelineConfig()
    train_config = train_config or TrainConfig()
    artifacts = build_training_artifacts(train_csv, val_csv, pipeline_config)
    penalty_matrix = (
        np.load(penalty_matrix_path)
        if penalty_matrix_path is not None and penalty_matrix_path.exists()
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, switches in VARIANTS.items():
        model_config = replace(train_config.model, **switches)
        variant_config = replace(train_config, model=model_config)
        result = train_model(artifacts, variant_config, penalty_matrix=penalty_matrix)
        variant_output = output_dir / name
        save_training_result(result, variant_output, variant_config)
        summary[name] = {
            "best_epoch": result.best_epoch,
            "metrics": result.metrics,
            "model_config": asdict(model_config),
        }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    run_ablations(
        root / "data" / "train.csv",
        root / "data" / "val.csv",
        root / "outputs" / "ablations",
        root / "data" / "penalty_matrix.npy",
    )
