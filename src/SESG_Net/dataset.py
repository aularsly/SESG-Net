from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import RAW_TO_CONTINUOUS


def build_well_groups(frame: pd.DataFrame) -> Dict[object, Tuple[int, int]]:
    if frame.empty:
        return {}
    wells = frame["WELL"].to_numpy()
    groups: Dict[object, Tuple[int, int]] = {}
    start = 0
    for index in range(1, len(frame)):
        if wells[index] != wells[index - 1]:
            groups[wells[index - 1]] = (start, index)
            start = index
    groups[wells[-1]] = (start, len(frame))
    return groups


def build_diff_missing_map(
    continuous_columns: Sequence[str],
    missing_columns: Sequence[str],
    difference_columns: Sequence[str],
) -> Dict[str, int]:
    missing_set = set(missing_columns)
    mapping: Dict[str, int] = {}
    for difference_column in difference_columns:
        raw_column = next(
            (
                raw
                for raw, continuous in RAW_TO_CONTINUOUS.items()
                if continuous == difference_column
            ),
            None,
        )
        missing_column = f"{raw_column}_MISS" if raw_column is not None else None
        if missing_column in missing_set:
            mapping[difference_column] = list(missing_columns).index(missing_column)
    return mapping


class SegmentLithologySequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        well_groups: Mapping[object, Tuple[int, int]],
        indices: np.ndarray,
        continuous_columns: List[str],
        missing_columns: List[str],
        difference_columns: List[str],
        mean: np.ndarray,
        std: np.ndarray,
        window: int = 129,
    ) -> None:
        if "SEGMENT_ID" not in frame.columns:
            raise ValueError("SEGMENT_ID is missing; build training artifacts first")
        if window % 2 != 1:
            raise ValueError("The centered sequence window must have an odd length")

        self.frame = frame
        self.indices = indices.astype(np.int64)
        self.window = window
        self.half_window = window // 2
        self.continuous_columns = continuous_columns
        self.missing_columns = missing_columns
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.difference_columns = [
            column for column in difference_columns if column in continuous_columns
        ]
        difference_indices = [continuous_columns.index(c) for c in self.difference_columns]
        difference_missing_map = build_diff_missing_map(
            continuous_columns,
            missing_columns,
            self.difference_columns,
        )

        self.well_main: Dict[object, np.ndarray] = {}
        self.well_missing: Dict[object, np.ndarray] = {}
        self.well_target: Dict[object, np.ndarray] = {}
        self.well_group: Dict[object, np.ndarray] = {}
        self.well_formation: Dict[object, np.ndarray] = {}
        self.well_segment: Dict[object, np.ndarray] = {}
        self.global_to_local: Dict[int, Tuple[object, int]] = {}

        required_wells = frame.iloc[self.indices]["WELL"].unique()
        for well_name in required_wells:
            start, end = well_groups[well_name]
            well = frame.iloc[start:end]
            continuous = well[continuous_columns].to_numpy(dtype=np.float32)
            continuous = (continuous - self.mean) / self.std
            missing = (
                well[missing_columns].to_numpy(dtype=np.float32)
                if missing_columns
                else np.zeros((len(well), 0), dtype=np.float32)
            )

            if difference_indices:
                base = continuous[:, difference_indices]
                first = np.zeros_like(base, dtype=np.float32)
                second = np.zeros_like(base, dtype=np.float32)
                for feature_index, column in enumerate(self.difference_columns):
                    missing_mask = (
                        missing[:, difference_missing_map[column]]
                        if column in difference_missing_map
                        else np.zeros(len(well), dtype=np.float32)
                    )
                    valid_first = (missing_mask[1:] == 0) & (missing_mask[:-1] == 0)
                    raw_first = base[1:, feature_index] - base[:-1, feature_index]
                    first[1:, feature_index] = np.where(valid_first, raw_first, 0.0)

                    valid_second = (
                        (missing_mask[2:] == 0)
                        & (missing_mask[1:-1] == 0)
                        & (missing_mask[:-2] == 0)
                    )
                    raw_second = first[2:, feature_index] - first[1:-1, feature_index]
                    second[2:, feature_index] = np.where(valid_second, raw_second, 0.0)
            else:
                first = np.zeros((len(well), 0), dtype=np.float32)
                second = np.zeros((len(well), 0), dtype=np.float32)

            self.well_main[well_name] = np.concatenate(
                [continuous, first, second], axis=1
            ).astype(np.float32)
            self.well_missing[well_name] = missing
            self.well_target[well_name] = well["target"].to_numpy(dtype=np.int64)
            self.well_group[well_name] = well["GROUP_ID"].to_numpy(dtype=np.int64)
            self.well_formation[well_name] = well["FORMATION_ID"].to_numpy(dtype=np.int64)
            self.well_segment[well_name] = well["SEGMENT_ID"].to_numpy(dtype=np.int64)
            for global_index in range(start, end):
                self.global_to_local[global_index] = (well_name, global_index - start)

        first_well = next(iter(self.well_main))
        self.main_dim = self.well_main[first_well].shape[1]
        self.missing_dim = self.well_missing[first_well].shape[1]

    def __len__(self) -> int:
        return len(self.indices)

    def _slice_with_edge_padding(self, array: np.ndarray, position: int) -> np.ndarray:
        left = position - self.half_window
        right = position + self.half_window
        bounded_left = max(0, left)
        bounded_right = min(len(array) - 1, right)
        sequence = array[bounded_left : bounded_right + 1]
        if left < 0:
            sequence = np.vstack([np.repeat(sequence[:1], -left, axis=0), sequence])
        if right >= len(array):
            sequence = np.vstack(
                [sequence, np.repeat(sequence[-1:], right - len(array) + 1, axis=0)]
            )
        return sequence

    def __getitem__(self, item: int):
        global_index = int(self.indices[item])
        well_name, position = self.global_to_local[global_index]
        main = torch.from_numpy(
            self._slice_with_edge_padding(self.well_main[well_name], position).T.copy()
        )
        missing = torch.from_numpy(
            self._slice_with_edge_padding(self.well_missing[well_name], position).T.copy()
        )
        target = torch.tensor(self.well_target[well_name][position], dtype=torch.long)
        group = torch.tensor(self.well_group[well_name][position], dtype=torch.long)
        formation = torch.tensor(self.well_formation[well_name][position], dtype=torch.long)
        segment = torch.tensor(self.well_segment[well_name][position], dtype=torch.long)
        return (
            main,
            missing,
            target,
            group,
            formation,
            segment,
            torch.tensor(global_index, dtype=torch.long),
        )
