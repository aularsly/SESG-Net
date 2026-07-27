import argparse
import logging
from pathlib import Path

import numpy as np

from .config import ModelConfig, PipelineConfig, TrainConfig
from .preprocessing import build_training_artifacts
from .training import save_training_result, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--penalty-matrix", type=Path, default=Path("data/penalty_matrix.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--window", type=int, default=129)
    parser.add_argument("--search-radius", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    pipeline_config = PipelineConfig(search_radius_xyz=args.search_radius)
    model_config = ModelConfig(window=args.window)
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        model=model_config,
    )

    artifacts = build_training_artifacts(
        args.train_csv,
        args.val_csv,
        pipeline_config,
    )
    penalty_matrix = (
        np.load(args.penalty_matrix) if args.penalty_matrix.exists() else None
    )
    result = train_model(artifacts, train_config, penalty_matrix=penalty_matrix)
    save_training_result(result, args.output_dir, train_config)
    logging.info("Best epoch: %d", result.best_epoch)
    logging.info("Metrics: %s", result.metrics)
    logging.info("Final outputs: %s", args.output_dir)


if __name__ == "__main__":
    main()
