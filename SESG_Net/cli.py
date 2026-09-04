import argparse
import logging
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True

from .config import ModelConfig, PipelineConfig, TrainConfig
from .preprocessing import build_training_artifacts
from .training import save_training_result, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--penalty-matrix", type=Path, default=Path("data/penalty_matrix.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--window", type=int, default=129)
    parser.add_argument("--search-radius", type=float, default=41110.0)
    return parser


def run_training(
    train_csv: Path,
    val_csv: Path,
    penalty_matrix_path: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 41,
    num_workers: int = 4,
    window: int = 129,
    search_radius: float = 41110.0,
):
    pipeline_config = PipelineConfig(search_radius_xyz=search_radius)
    model_config = ModelConfig(window=window)
    train_config = TrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        num_workers=num_workers,
        model=model_config,
    )
    artifacts = build_training_artifacts(train_csv, val_csv, pipeline_config)
    penalty_matrix = np.load(penalty_matrix_path) if penalty_matrix_path.exists() else None
    result = train_model(artifacts, train_config, penalty_matrix=penalty_matrix)
    save_training_result(result, output_dir, train_config)
    return result


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = run_training(
        args.train_csv,
        args.val_csv,
        args.penalty_matrix,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        window=args.window,
        search_radius=args.search_radius,
    )
    logging.info("Best epoch: %d", result.best_epoch)
    logging.info("Metrics: %s", result.metrics)
    logging.info("Final outputs: %s", args.output_dir)


if __name__ == "__main__":
    main()
