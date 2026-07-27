from dataclasses import dataclass, field
from typing import Optional, Tuple


LITHOLOGY_MAP = {
    30000: 0,
    65030: 1,
    65000: 2,
    80000: 3,
    74000: 4,
    70000: 5,
    70032: 6,
    88000: 7,
    86000: 8,
    99000: 9,
    90000: 10,
    93000: 11,
}

TARGET_COLUMN = "FORCE_2020_LITHOFACIES_LITHOLOGY"

RAW_CURVES = (
    "CALI",
    "RSHA",
    "RMED",
    "RDEP",
    "RHOB",
    "GR",
    "NPHI",
    "PEF",
    "DTC",
    "SP",
    "BS",
    "ROP",
    "DRHO",
)

LOG_CURVES = ("RSHA", "RMED", "RDEP")
NORMALIZED_CURVES = ("GR", "RHOB", "NPHI", "DTC", "SP", "ROP")

SEQUENCE_CONTINUOUS_COLUMNS = (
    "GR_NORM",
    "RHOB_NORM",
    "NPHI_NORM",
    "DTC_NORM",
    "SP_NORM",
    "ROP_NORM",
    "RSHA_LOG",
    "RMED_LOG",
    "RDEP_LOG",
    "UMA",
    "VSH",
    "DEPTH_REL",
)

DIFFERENCE_COLUMNS = (
    "GR_NORM",
    "RHOB_NORM",
    "NPHI_NORM",
    "DTC_NORM",
    "RDEP_LOG",
)

RAW_TO_CONTINUOUS = {
    "GR": "GR_NORM",
    "RHOB": "RHOB_NORM",
    "NPHI": "NPHI_NORM",
    "DTC": "DTC_NORM",
    "SP": "SP_NORM",
    "ROP": "ROP_NORM",
    "RSHA": "RSHA_LOG",
    "RMED": "RMED_LOG",
    "RDEP": "RDEP_LOG",
}


@dataclass(frozen=True)
class PipelineConfig:
    response_columns: Tuple[str, ...] = (
        "GR_NORM",
        "RHOB_NORM",
        "NPHI_NORM",
        "DTC_NORM",
        "SP_NORM",
        "ROP_NORM",
        "RSHA_LOG",
        "RMED_LOG",
        "RDEP_LOG",
        "UMA",
        "VSH",
    )
    response_diff_quantile: float = 0.98
    min_supernode_points: int = 8
    max_supernode_points: int = 64
    candidate_well_k: int = 6
    candidate_per_well: int = 4
    max_candidate_nodes: int = 20
    search_radius_quantile: float = 0.95
    search_radius_xyz: Optional[float] = None
    csv_separator: str = ";"


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 12
    window: int = 129
    kernel_sizes: Tuple[int, ...] = (9, 19, 39)
    base_out: int = 16
    depth: int = 4
    bottleneck_channels: int = 32
    dropout: float = 0.1
    missing_branch_dim: int = 16
    spatial_dim: int = 32
    group_embedding_dim: int = 16
    formation_embedding_dim: int = 16
    edge_hidden_dim: int = 64
    final_neighbor_k: int = 10


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 48
    num_workers: int = 4
    use_amp: bool = True
    pin_memory: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
