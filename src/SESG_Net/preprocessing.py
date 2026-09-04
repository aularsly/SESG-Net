from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import (
    LITHOLOGY_MAP,
    LOG_CURVES,
    NORMALIZED_CURVES,
    PipelineConfig,
    RAW_CURVES,
    TARGET_COLUMN,
)


LOGGER = logging.getLogger(__name__)
PathLike = Union[str, Path]


@dataclass
class TrainingArtifacts:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    wells: pd.DataFrame
    segments: pd.DataFrame
    segment_feature_columns: List[str]
    candidate_neighbors: np.ndarray
    candidate_mask: np.ndarray
    pair_features: np.ndarray
    preprocessing_stats: Dict[str, Any]
    search_radius_xyz: float


def safe_log10_series(values: pd.Series, lower: float = 0.1) -> pd.Series:
    clipped = values.where(values > lower, lower)
    return np.log10(clipped)


def build_strat_mapping(series: pd.Series) -> Dict[str, int]:
    values = series.fillna("Unknown").astype(str)
    ordered = ["Unknown"] + sorted(v for v in values.unique() if v != "Unknown")
    return {value: index for index, value in enumerate(ordered)}


def cast_frame_types(frame: pd.DataFrame) -> pd.DataFrame:
    float_columns = frame.select_dtypes(include=[np.float64]).columns
    if len(float_columns):
        frame[float_columns] = frame[float_columns].astype(np.float32)

    for column in ("target", "GROUP_ID", "FORMATION_ID", "WELL_ID", "SEGMENT_ID"):
        if column in frame.columns:
            frame[column] = frame[column].astype(np.int32)
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _median_or_zero(values: pd.Series) -> float:
    median = values.median()
    return float(0.0 if pd.isna(median) else median)


def _well_median_fill(
    frame: pd.DataFrame,
    column: str,
    global_fallback: float,
) -> pd.Series:
    well_median = frame.groupby("WELL", sort=False)[column].transform("median")
    return frame[column].fillna(well_median).fillna(global_fallback)


def preprocess_split(
    frame: pd.DataFrame,
    split_name: str,
    stats: Optional[Mapping[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = frame.copy()
    _require_columns(df, ("WELL", "DEPTH_MD", TARGET_COLUMN, "X_LOC", "Y_LOC"))

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(LITHOLOGY_MAP)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df.rename(columns={TARGET_COLUMN: "target"}, inplace=True)
    df["target"] = df["target"].astype(np.int32)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)

    for column in LOG_CURVES:
        if column in df.columns:
            df.loc[df[column] <= 0, column] = np.nan

    raw_columns = [column for column in RAW_CURVES if column in df.columns]
    for column in raw_columns:
        df[f"{column}_MISS"] = df[column].isna().astype(np.float32)

    if stats is None:
        curve_fill_values = {
            column: _median_or_zero(df[column]) for column in raw_columns
        }
    else:
        curve_fill_values = dict(stats["curve_fill_values"])

    for column in raw_columns:
        fallback = float(curve_fill_values.get(column, 0.0))
        df[column] = _well_median_fill(df, column, fallback).astype(np.float32)

    for column in LOG_CURVES:
        if column in df.columns:
            df[f"{column}_LOG"] = safe_log10_series(df[column]).astype(np.float32)

    if "PEF" in df.columns and "RHOB" in df.columns:
        df["UMA"] = ((df["PEF"] * (df["RHOB"] + 0.1883)) / 1.07).astype(np.float32)
    else:
        df["UMA"] = np.float32(0.0)

    if "GR" in df.columns:
        if stats is None:
            gr_min = float(df["GR"].min())
            gr_range = max(float(df["GR"].max()) - gr_min, 1e-6)
        else:
            gr_min = float(stats["gr_min"])
            gr_range = float(stats["gr_range"])
        df["VSH"] = ((df["GR"] - gr_min) / gr_range).clip(0, 1).astype(np.float32)
    else:
        gr_min, gr_range = 0.0, 1.0
        df["VSH"] = np.float32(0.0)

    depth_group = df.groupby("WELL", sort=False)["DEPTH_MD"]
    depth_min = depth_group.transform("min")
    depth_max = depth_group.transform("max")
    df["DEPTH_REL"] = (
        (df["DEPTH_MD"] - depth_min) / (depth_max - depth_min + 1e-6)
    ).astype(np.float32)

    for column in NORMALIZED_CURVES:
        if column not in df.columns:
            continue
        grouped = df.groupby("WELL", sort=False)[column]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, 1).fillna(1)
        df[f"{column}_NORM"] = ((df[column] - mean) / std).astype(np.float32)

    if stats is None:
        group_source = df.get("GROUP", pd.Series("Unknown", index=df.index))
        formation_source = df.get("FORMATION", pd.Series("Unknown", index=df.index))
        group_map = build_strat_mapping(group_source)
        formation_map = build_strat_mapping(formation_source)
    else:
        group_map = dict(stats["group_map"])
        formation_map = dict(stats["formation_map"])

    for column, mapping in (("GROUP", group_map), ("FORMATION", formation_map)):
        if column not in df.columns:
            df[column] = "Unknown"
        values = df[column].fillna("Unknown").astype(str)
        df[column] = values
        df[f"{column}_ID"] = values.map(lambda value: mapping.get(value, 0)).astype(np.int32)

    coordinate_columns = [column for column in ("X_LOC", "Y_LOC", "Z_LOC") if column in df]
    if stats is None:
        coordinate_fill_values = {
            column: _median_or_zero(df[column]) for column in coordinate_columns
        }
    else:
        coordinate_fill_values = dict(stats["coordinate_fill_values"])

    for column in coordinate_columns:
        df[column] = df[column].fillna(coordinate_fill_values.get(column, 0.0))

    if "Z_LOC" not in df.columns:
        df["Z_LOC"] = -df["DEPTH_MD"].astype(np.float32)
        coordinate_columns.append("Z_LOC")

    if stats is None:
        xyz_mean = df[["X_LOC", "Y_LOC", "Z_LOC"]].mean().to_numpy(dtype=np.float32)
        xyz_std = df[["X_LOC", "Y_LOC", "Z_LOC"]].std().to_numpy(dtype=np.float32)
        xyz_std = np.where(np.isfinite(xyz_std) & (xyz_std > 0), xyz_std, 1.0)
    else:
        xyz_mean = np.asarray(stats["xyz_mean"], dtype=np.float32)
        xyz_std = np.asarray(stats["xyz_std"], dtype=np.float32)

    for index, column in enumerate(("X_LOC", "Y_LOC", "Z_LOC")):
        normalized_name = {"X_LOC": "X_NORM", "Y_LOC": "Y_NORM", "Z_LOC": "Z_NORM"}[column]
        df[normalized_name] = ((df[column] - xyz_mean[index]) / xyz_std[index]).astype(np.float32)

    sequence_like = [
        column
        for column in df.columns
        if column.endswith(("_NORM", "_LOG", "_MISS"))
        or column in ("UMA", "VSH", "DEPTH_REL")
    ]
    df[sequence_like] = (
        df[sequence_like].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    )

    fitted_stats = {
        "curve_fill_values": curve_fill_values,
        "gr_min": gr_min,
        "gr_range": gr_range,
        "group_map": group_map,
        "formation_map": formation_map,
        "coordinate_fill_values": coordinate_fill_values,
        "xyz_mean": xyz_mean.tolist(),
        "xyz_std": xyz_std.tolist(),
    }
    LOGGER.info("Preprocessed %s split: %d rows, %d wells", split_name, len(df), df["WELL"].nunique())
    return cast_frame_types(df), fitted_stats


def attach_well_info(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_wells = set(train_df["WELL"].astype(str).unique())
    combined = pd.concat([train_df, val_df], ignore_index=True, sort=False)
    well_df = (
        combined[["WELL", "X_LOC", "Y_LOC", "Z_LOC"]]
        .groupby("WELL", as_index=False)
        .median(numeric_only=True)
        .reset_index(drop=True)
    )
    well_df["WELL_ID"] = np.arange(len(well_df), dtype=np.int32)

    fit_wells = well_df[well_df["WELL"].astype(str).isin(train_wells)]
    for column in ("X_LOC", "Y_LOC"):
        mean = float(fit_wells[column].mean())
        std = float(fit_wells[column].std())
        if not np.isfinite(std) or std == 0:
            std = 1.0
        well_df[f"{column}_NORM"] = ((well_df[column] - mean) / std).astype(np.float32)

    merge_columns = ["WELL", "WELL_ID", "X_LOC_NORM", "Y_LOC_NORM"]
    train_out = train_df.merge(well_df[merge_columns], on="WELL", how="left")
    val_out = val_df.merge(well_df[merge_columns], on="WELL", how="left")
    train_out = train_out.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    val_out = val_out.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    return cast_frame_types(train_out), cast_frame_types(val_out), cast_frame_types(well_df)


def mode_int(values: Iterable[Any]) -> int:
    series = pd.Series(values).dropna()
    return 0 if series.empty else int(series.mode().iloc[0])


def fit_response_scaler_and_threshold(
    train_df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[List[str], np.ndarray, np.ndarray, float]:
    columns = [column for column in config.response_columns if column in train_df.columns]
    if not columns:
        raise ValueError("No response columns are available for supernode construction")

    mean = train_df[columns].mean().to_numpy(dtype=np.float32)
    std = train_df[columns].std().to_numpy(dtype=np.float32)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)

    differences: List[np.ndarray] = []
    for _, well in train_df.groupby("WELL", sort=False):
        if len(well) < 2:
            continue
        response = (well[columns].to_numpy(dtype=np.float32) - mean) / std
        group = well["GROUP_ID"].to_numpy(dtype=np.int32)
        formation = well["FORMATION_ID"].to_numpy(dtype=np.int32)
        adjacent_distance = np.linalg.norm(response[1:] - response[:-1], axis=1)
        same_stratigraphy = (group[1:] == group[:-1]) & (formation[1:] == formation[:-1])
        valid = np.isfinite(adjacent_distance) & same_stratigraphy
        if valid.any():
            differences.append(adjacent_distance[valid])

    threshold = (
        float(np.quantile(np.concatenate(differences), config.response_diff_quantile))
        if differences
        else float("inf")
    )
    return columns, mean, std, threshold


def _make_supernode_row(
    segment_id: int,
    split_name: str,
    start: int,
    end: int,
    segment: pd.DataFrame,
    response_columns: Sequence[str],
    response_mean: np.ndarray,
    response_std: np.ndarray,
) -> Dict[str, Any]:
    center = segment.iloc[(len(segment) - 1) // 2]
    response = segment[list(response_columns)].to_numpy(dtype=np.float32)
    response = (response - response_mean) / response_std
    is_train = split_name == "train"

    row: Dict[str, Any] = {
        "SEGMENT_ID": segment_id,
        "WELL": center["WELL"],
        "WELL_ID": int(center["WELL_ID"]),
        "START_POS": start,
        "END_POS": end,
        "SEGMENT_NPOINTS": end - start,
        "DEPTH_MD_CENTER": float(center["DEPTH_MD"]),
        "DEPTH_REL_CENTER": float(center["DEPTH_REL"]),
        "SEGMENT_THICKNESS_MD": (
            float(segment["DEPTH_MD"].iloc[-1] - segment["DEPTH_MD"].iloc[0])
            if len(segment) > 1
            else 0.0
        ),
        "GROUP_ID_MODE": mode_int(segment["GROUP_ID"]),
        "FORMATION_ID_MODE": mode_int(segment["FORMATION_ID"]),
        "X_LOC": float(center["X_LOC"]),
        "Y_LOC": float(center["Y_LOC"]),
        "Z_LOC": float(center["Z_LOC"]),
        "X_LOC_NORM": float(center["X_LOC_NORM"]),
        "Y_LOC_NORM": float(center["Y_LOC_NORM"]),
        "Z_NORM": float(center["Z_NORM"]),
        "TRAIN_POINTS": int(len(segment) if is_train else 0),
        "VAL_POINTS": int(0 if is_train else len(segment)),
        "SOURCE_SPLIT": split_name,
    }
    for index, column in enumerate(response_columns):
        row[f"{column}_MEAN"] = float(np.mean(response[:, index]))
        row[f"{column}_STD"] = float(np.std(response[:, index], ddof=0))
    return row


def _segment_one_split(
    frame: pd.DataFrame,
    split_name: str,
    first_segment_id: int,
    response_columns: Sequence[str],
    response_mean: np.ndarray,
    response_std: np.ndarray,
    response_threshold: float,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    output = frame.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True).copy()
    point_segment_ids = np.full(len(output), -1, dtype=np.int32)
    rows: List[Dict[str, Any]] = []

    for _, indices_raw in output.groupby("WELL", sort=False).indices.items():
        indices = np.asarray(indices_raw, dtype=np.int64)
        well = output.iloc[indices].reset_index(drop=True)
        response = well[list(response_columns)].to_numpy(dtype=np.float32)
        response = (response - response_mean) / response_std
        group = well["GROUP_ID"].to_numpy(dtype=np.int32)
        formation = well["FORMATION_ID"].to_numpy(dtype=np.int32)

        boundaries: List[Tuple[int, int]] = []
        start = 0
        for position in range(1, len(well)):
            length = position - start
            stratigraphy_changed = (
                group[position] != group[position - 1]
                or formation[position] != formation[position - 1]
            )
            response_changed = (
                length >= config.min_supernode_points
                and np.isfinite(response_threshold)
                and float(np.linalg.norm(response[position] - response[position - 1]))
                > response_threshold
            )
            if stratigraphy_changed or response_changed or length >= config.max_supernode_points:
                boundaries.append((start, position))
                start = position
        boundaries.append((start, len(well)))

        merged: List[List[int]] = []
        for start, end in boundaries:
            if not merged:
                merged.append([start, end])
                continue
            previous_start, previous_end = merged[-1]
            same_stratigraphy = (
                group[start] == group[previous_end - 1]
                and formation[start] == formation[previous_end - 1]
            )
            can_merge = (
                end - start < config.min_supernode_points
                and same_stratigraphy
                and end - previous_start <= config.max_supernode_points
            )
            if can_merge:
                merged[-1][1] = end
            else:
                merged.append([start, end])

        for start, end in merged:
            absolute_indices = indices[start:end]
            segment_id = first_segment_id + len(rows)
            segment = output.iloc[absolute_indices]
            rows.append(
                _make_supernode_row(
                    segment_id,
                    split_name,
                    start,
                    end,
                    segment,
                    response_columns,
                    response_mean,
                    response_std,
                )
            )
            point_segment_ids[absolute_indices] = segment_id

    if (point_segment_ids < 0).any():
        raise RuntimeError(f"Some {split_name} samples were not assigned to a supernode")
    output["SEGMENT_ID"] = point_segment_ids
    return cast_frame_types(output), rows


def build_supernodes_and_assign(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    response_columns: Sequence[str],
    response_mean: np.ndarray,
    response_std: np.ndarray,
    response_threshold: float,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    train_out, train_rows = _segment_one_split(
        train_df,
        "train",
        0,
        response_columns,
        response_mean,
        response_std,
        response_threshold,
        config,
    )
    val_out, val_rows = _segment_one_split(
        val_df,
        "val",
        len(train_rows),
        response_columns,
        response_mean,
        response_std,
        response_threshold,
        config,
    )
    segments = cast_frame_types(pd.DataFrame(train_rows + val_rows))
    feature_columns = (
        [f"{column}_MEAN" for column in response_columns]
        + [f"{column}_STD" for column in response_columns]
        + [
            "X_LOC_NORM",
            "Y_LOC_NORM",
            "Z_NORM",
            "DEPTH_REL_CENTER",
            "SEGMENT_THICKNESS_MD",
        ]
    )
    feature_columns = [column for column in feature_columns if column in segments.columns]
    return train_out, val_out, segments, feature_columns


def estimate_search_radius(segments: pd.DataFrame, config: PipelineConfig) -> float:
    if config.search_radius_xyz is not None:
        return float(config.search_radius_xyz)

    train_wells = (
        segments[segments["TRAIN_POINTS"] > 0][["WELL_ID", "X_LOC", "Y_LOC"]]
        .groupby("WELL_ID", as_index=False)
        .median(numeric_only=True)
    )
    xy = train_wells[["X_LOC", "Y_LOC"]].to_numpy(dtype=np.float32)
    if len(xy) <= 1:
        return float("inf")

    neighbor_count = min(config.candidate_well_k + 1, len(xy))
    estimator = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    distances, _ = estimator.fit(xy).kneighbors(xy)
    radius = float(np.quantile(distances[:, -1], config.search_radius_quantile))
    return max(radius, 1e-6)


def _nearest_positions(sorted_values: np.ndarray, target: float, count: int) -> List[int]:
    if not len(sorted_values) or count <= 0:
        return []
    position = int(np.searchsorted(sorted_values, target, side="left"))
    left, right = position - 1, position
    selected: List[int] = []
    while len(selected) < min(count, len(sorted_values)):
        if left < 0:
            selected.append(right)
            right += 1
        elif right >= len(sorted_values):
            selected.append(left)
            left -= 1
        elif abs(float(sorted_values[left]) - target) <= abs(float(sorted_values[right]) - target):
            selected.append(left)
            left -= 1
        else:
            selected.append(right)
            right += 1
    return selected


def _build_radius_index(source_ids: np.ndarray, xyz: np.ndarray, radius: float):
    estimator = NearestNeighbors(radius=radius, metric="euclidean")
    estimator.fit(xyz[source_ids])
    return source_ids, estimator


def _query_radius_index(target: int, index, xyz: np.ndarray, well_ids: np.ndarray, radius: float):
    source_ids, estimator = index
    distances, positions = estimator.radius_neighbors(
        xyz[target].reshape(1, -1),
        radius=radius,
        return_distance=True,
        sort_results=True,
    )
    candidate_ids = source_ids[positions[0]]
    candidate_distances = distances[0]
    keep = well_ids[candidate_ids] != well_ids[target]
    return candidate_ids[keep], candidate_distances[keep]


def build_candidate_graph(
    segments: pd.DataFrame,
    response_columns: Sequence[str],
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    node_count = len(segments)
    radius = estimate_search_radius(segments, config)
    mean_matrix = segments[[f"{c}_MEAN" for c in response_columns]].to_numpy(dtype=np.float32)
    std_matrix = segments[[f"{c}_STD" for c in response_columns]].to_numpy(dtype=np.float32)
    x = segments["X_LOC"].to_numpy(dtype=np.float32)
    y = segments["Y_LOC"].to_numpy(dtype=np.float32)
    z = segments["Z_LOC"].to_numpy(dtype=np.float32)
    depth_relative = segments["DEPTH_REL_CENTER"].to_numpy(dtype=np.float32)
    well_ids = segments["WELL_ID"].to_numpy(dtype=np.int32)
    group_ids = segments["GROUP_ID_MODE"].to_numpy(dtype=np.int32)
    formation_ids = segments["FORMATION_ID_MODE"].to_numpy(dtype=np.int32)
    source_ids = np.flatnonzero(segments["TRAIN_POINTS"].to_numpy() > 0).astype(np.int32)
    xyz = np.column_stack([x, y, z]).astype(np.float32)
    formation_indices = {}
    group_indices = {}
    for formation_id in np.unique(formation_ids[source_ids]):
        if formation_id != 0:
            ids = source_ids[formation_ids[source_ids] == formation_id]
            formation_indices[int(formation_id)] = _build_radius_index(ids, xyz, radius)
    for group_id in np.unique(group_ids[source_ids]):
        ids = source_ids[group_ids[source_ids] == group_id]
        group_indices[int(group_id)] = _build_radius_index(ids, xyz, radius)

    neighbors = np.zeros((node_count, config.max_candidate_nodes), dtype=np.int32)
    mask = np.zeros((node_count, config.max_candidate_nodes), dtype=np.bool_)
    pair_features = np.zeros((node_count, config.max_candidate_nodes, 5), dtype=np.float32)
    response_scale = max(float(np.sqrt(len(response_columns))), 1.0)

    for source in range(node_count):
        relations: List[Tuple[float, int, np.ndarray]] = []
        selected_ids = np.empty(0, dtype=np.int32)
        formation_index = formation_indices.get(int(formation_ids[source]))
        if formation_index is not None:
            selected_ids, _ = _query_radius_index(source, formation_index, xyz, well_ids, radius)
        if config.allow_group_fallback and len(selected_ids) < config.min_formation_candidates:
            group_index = group_indices.get(int(group_ids[source]))
            if group_index is not None:
                group_ids_found, _ = _query_radius_index(source, group_index, xyz, well_ids, radius)
                selected_ids = np.unique(np.concatenate([selected_ids, group_ids_found])).astype(np.int32)
        for target in selected_ids:
            horizontal_distance = float(np.hypot(x[source] - x[target], y[source] - y[target]))
            vertical_distance = abs(float(z[source] - z[target]))
            mean_distance = float(
                np.linalg.norm(mean_matrix[source] - mean_matrix[target]) / response_scale
            )
            std_distance = float(
                np.linalg.norm(std_matrix[source] - std_matrix[target]) / response_scale
            )
            spatial_scale = radius if np.isfinite(radius) else 1.0
            relation = np.asarray(
                [
                    horizontal_distance / spatial_scale,
                    vertical_distance / spatial_scale,
                    abs(float(depth_relative[source] - depth_relative[target])),
                    mean_distance,
                    std_distance,
                ],
                dtype=np.float32,
            )
            preliminary_score = float(relation[0] + relation[1] + 0.5 * relation[3])
            relations.append((preliminary_score, int(target), relation))

        best_by_node: Dict[int, Tuple[float, np.ndarray]] = {}
        for score, target, relation in relations:
            previous = best_by_node.get(target)
            if previous is None or score < previous[0]:
                best_by_node[target] = (score, relation)
        selected = sorted(
            (score, target, relation)
            for target, (score, relation) in best_by_node.items()
        )[: config.max_candidate_nodes]

        neighbors[source, :] = source
        for position, (_, target, relation) in enumerate(selected):
            neighbors[source, position] = target
            mask[source, position] = True
            pair_features[source, position] = relation

    LOGGER.info(
        "Candidate graph: %d nodes, radius %.2f, mean %.2f valid candidates",
        node_count,
        radius,
        float(mask.sum(axis=1).mean()),
    )
    return neighbors, mask, pair_features, radius


def build_artifacts_from_frames(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    config: Optional[PipelineConfig] = None,
) -> TrainingArtifacts:
    config = config or PipelineConfig()
    train_df, stats = preprocess_split(train_raw, "train")
    val_df, _ = preprocess_split(val_raw, "val", stats=stats)
    train_df, val_df, wells = attach_well_info(train_df, val_df)

    response_columns, response_mean, response_std, threshold = (
        fit_response_scaler_and_threshold(train_df, config)
    )
    train_df, val_df, segments, feature_columns = build_supernodes_and_assign(
        train_df,
        val_df,
        response_columns,
        response_mean,
        response_std,
        threshold,
        config,
    )
    neighbors, mask, pair_features, radius = build_candidate_graph(
        segments,
        response_columns,
        config,
    )

    stats.update(
        {
            "response_columns": response_columns,
            "response_feature_mean": response_mean.tolist(),
            "response_feature_std": response_std.tolist(),
            "response_diff_quantile": config.response_diff_quantile,
            "response_diff_threshold": threshold,
            "min_supernode_points": config.min_supernode_points,
            "max_supernode_points": config.max_supernode_points,
            "candidate_well_k": config.candidate_well_k,
            "candidate_per_well": config.candidate_per_well,
            "max_candidate_nodes": config.max_candidate_nodes,
            "min_formation_candidates": config.min_formation_candidates,
            "allow_group_fallback": config.allow_group_fallback,
            "search_radius_quantile": config.search_radius_quantile,
            "search_radius_xyz": radius,
            "segment_feature_columns": feature_columns,
        }
    )
    return TrainingArtifacts(
        train_df=train_df,
        val_df=val_df,
        wells=wells,
        segments=segments,
        segment_feature_columns=feature_columns,
        candidate_neighbors=neighbors,
        candidate_mask=mask,
        pair_features=pair_features,
        preprocessing_stats=stats,
        search_radius_xyz=radius,
    )


def build_training_artifacts(
    train_csv: PathLike,
    val_csv: PathLike,
    config: Optional[PipelineConfig] = None,
) -> TrainingArtifacts:
    config = config or PipelineConfig()
    train_raw = pd.read_csv(train_csv, sep=config.csv_separator)
    val_raw = pd.read_csv(val_csv, sep=config.csv_separator)
    return build_artifacts_from_frames(train_raw, val_raw, config)
