from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from SESG_Net.config import PipelineConfig, TrainConfig
from SESG_Net.preprocessing import build_training_artifacts
from SESG_Net.training import build_model


def test_sample_pipeline():
    root = Path(__file__).resolve().parents[1]
    artifacts = build_training_artifacts(
        root / "data" / "sample" / "train.csv",
        root / "data" / "sample" / "val.csv",
        PipelineConfig(),
    )
    model, _, dataset = build_model(
        artifacts,
        TrainConfig(num_workers=0, use_amp=False),
        torch.device("cpu"),
    )
    missing_columns = [column for column in artifacts.train_df if column.endswith("_MISS")]
    assert len(missing_columns) == 10
    assert dataset.missing_dim == 10
    assert artifacts.candidate_neighbors.shape[1] == 64
    assert model.final_neighbor_k == 10
    rows = [dataset[index] for index in range(min(8, len(dataset)))]
    inputs = [torch.stack([row[index] for row in rows]) for index in (0, 1, 3, 4, 5)]
    model.eval()
    with torch.no_grad():
        logits = model(*inputs)
    assert logits.shape == (len(rows), 12)
    assert np.isfinite(logits.numpy()).all()


def test_sample_files_have_disjoint_wells():
    root = Path(__file__).resolve().parents[1] / "data" / "sample"
    wells = {
        split: set(pd.read_csv(root / f"{split}.csv", sep=";")["WELL"].astype(str))
        for split in ("train", "val", "test")
    }
    assert wells["train"].isdisjoint(wells["val"])
    assert wells["train"].isdisjoint(wells["test"])
    assert wells["val"].isdisjoint(wells["test"])


def test_split_manifest_has_fixed_disjoint_wells():
    manifest = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "split_manifest.csv")
    assert list(manifest.columns) == ["WELL", "split"]
    assert manifest["split"].value_counts().to_dict() == {"train": 98, "val": 10, "test": 10}
    assert manifest["WELL"].nunique() == 118
