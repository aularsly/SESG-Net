from typing import Sequence

import torch
import torch.nn as nn


class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (9, 19, 39),
        bottleneck_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if any(kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("Temporal kernels must be odd to preserve window length")
        self.channel_mapping = nn.Conv1d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        )
        self.branches = nn.ModuleList(
            nn.Conv1d(
                bottleneck_channels,
                out_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=False,
            )
            for kernel in kernel_sizes
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        merged_channels = out_channels * (len(kernel_sizes) + 1)
        self.normalization = nn.BatchNorm1d(merged_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_channels = merged_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mapped = self.channel_mapping(inputs)
        branch_outputs = [branch(mapped) for branch in self.branches]
        branch_outputs.append(self.pool_branch(inputs))
        merged = torch.cat(branch_outputs, dim=1)
        return self.dropout(self.activation(self.normalization(merged)))


class InceptionResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        kernel_sizes: Sequence[int] = (9, 19, 39),
        bottleneck_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = []
        current_channels = in_channels
        for _ in range(depth):
            block = InceptionBlock1D(
                current_channels,
                out_channels,
                kernel_sizes,
                bottleneck_channels,
                dropout,
            )
            blocks.append(block)
            current_channels = block.out_channels
        self.blocks = nn.Sequential(*blocks)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, current_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(current_channels),
        )
        self.activation = nn.ReLU()
        self.out_channels = current_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.blocks(inputs) + self.shortcut(inputs))


class AttentionPool1D(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(inputs), dim=-1)
        return (inputs * weights).sum(dim=-1)


class MissingAuxiliaryBranch(nn.Module):
    def __init__(self, missing_dim: int, out_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.enabled = missing_dim > 0 and out_dim > 0
        self.out_dim = out_dim if self.enabled else 0
        if not self.enabled:
            return
        self.network = nn.Sequential(
            nn.Conv1d(missing_dim, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPool1D(out_dim, hidden_channels=32)

    def forward(self, missing: torch.Tensor):
        if not self.enabled:
            return None
        return self.pool(self.network(missing))


class SupernodeEncoder(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        group_embedding_dim: int,
        formation_embedding_dim: int,
        out_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = numeric_dim + group_embedding_dim + formation_embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        group_embedding: torch.Tensor,
        formation_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            torch.cat([numeric, group_embedding, formation_embedding], dim=-1)
        )


class DynamicEdgeScorer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.source_score = nn.Linear(node_dim, hidden_dim, bias=False)
        self.target_score = nn.Linear(node_dim, hidden_dim, bias=False)
        self.pair_score = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = nn.LeakyReLU(0.2)

    def forward(
        self,
        source: torch.Tensor,
        candidates: torch.Tensor,
        pair_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.source_score(source).unsqueeze(1)
            + self.target_score(candidates)
            + self.pair_score(pair_features)
        )
        scores = self.output(self.activation(hidden)).squeeze(-1).float()
        return scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)


class DynamicWeightedGraphSAGE(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.project_self = nn.Linear(feature_dim, hidden_dim)
        self.project_neighbor = nn.Linear(feature_dim, hidden_dim)
        self.join = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.GELU()
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        candidates: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        self_state = self.project_self(target)
        neighbor_state = (candidates * weights.unsqueeze(-1)).sum(dim=1)
        neighbor_state = self.project_neighbor(neighbor_state)
        joined = self.join(torch.cat([self_state, neighbor_state], dim=-1))
        return self.dropout(self.activation(self.normalization(joined))) + self_state


class SegmentSpatialTemporalStratModel(nn.Module):
    def __init__(
        self,
        main_dim: int,
        missing_dim: int,
        num_classes: int,
        segment_features: torch.Tensor,
        candidate_neighbors: torch.Tensor,
        candidate_mask: torch.Tensor,
        pair_features: torch.Tensor,
        segment_group_ids: torch.Tensor,
        segment_formation_ids: torch.Tensor,
        num_groups: int,
        num_formations: int,
        spatial_dim: int = 32,
        group_embedding_dim: int = 16,
        formation_embedding_dim: int = 16,
        base_out: int = 16,
        depth: int = 4,
        kernel_sizes: Sequence[int] = (9, 19, 39),
        bottleneck_channels: int = 32,
        dropout: float = 0.1,
        missing_branch_dim: int = 16,
        edge_hidden_dim: int = 64,
        final_neighbor_k: int = 10,
    ) -> None:
        super().__init__()
        self.sequence_encoder = InceptionResidualBlock(
            main_dim,
            base_out,
            depth,
            kernel_sizes,
            bottleneck_channels,
            dropout,
        )
        self.sequence_pool = AttentionPool1D(
            self.sequence_encoder.out_channels,
            hidden_channels=128,
        )
        self.missing_branch = MissingAuxiliaryBranch(
            missing_dim,
            missing_branch_dim,
            dropout,
        )

        self.register_buffer("segment_features", segment_features, persistent=False)
        self.register_buffer("candidate_neighbors", candidate_neighbors, persistent=False)
        self.register_buffer("candidate_mask", candidate_mask, persistent=False)
        self.register_buffer("pair_features", pair_features, persistent=False)
        self.register_buffer("segment_group_ids", segment_group_ids, persistent=False)
        self.register_buffer("segment_formation_ids", segment_formation_ids, persistent=False)

        self.group_embedding = nn.Embedding(num_groups, group_embedding_dim)
        self.formation_embedding = nn.Embedding(
            num_formations,
            formation_embedding_dim,
        )
        self.supernode_encoder = SupernodeEncoder(
            segment_features.shape[1],
            group_embedding_dim,
            formation_embedding_dim,
            spatial_dim,
            dropout,
        )
        self.edge_scorer = DynamicEdgeScorer(
            spatial_dim,
            pair_features.shape[-1],
            edge_hidden_dim,
            dropout,
        )
        self.graph_sage = DynamicWeightedGraphSAGE(spatial_dim, spatial_dim)
        self.final_neighbor_k = min(final_neighbor_k, candidate_neighbors.shape[1])

        classifier_input = self.sequence_encoder.out_channels + spatial_dim
        if self.missing_branch.enabled:
            classifier_input += self.missing_branch.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _encode_supernodes(self, node_ids: torch.Tensor) -> torch.Tensor:
        numeric = self.segment_features[node_ids]
        group = self.group_embedding(self.segment_group_ids[node_ids])
        formation = self.formation_embedding(self.segment_formation_ids[node_ids])
        return self.supernode_encoder(numeric, group, formation)

    def encode_spatial_batch(self, segment_ids: torch.Tensor) -> torch.Tensor:
        segment_ids = segment_ids.clamp(0, self.segment_features.shape[0] - 1)
        candidate_ids = self.candidate_neighbors[segment_ids]
        candidate_mask = self.candidate_mask[segment_ids]
        pair_features = self.pair_features[segment_ids]
        source = self._encode_supernodes(segment_ids)
        candidates = self._encode_supernodes(candidate_ids)

        scores = self.edge_scorer(source, candidates, pair_features, candidate_mask)
        top_scores, top_positions = torch.topk(
            scores,
            k=self.final_neighbor_k,
            dim=1,
        )
        valid = torch.gather(candidate_mask, 1, top_positions)
        gather_index = top_positions.unsqueeze(-1).expand(-1, -1, candidates.shape[-1])
        top_candidates = torch.gather(candidates, 1, gather_index)

        top_scores = top_scores.masked_fill(~valid, torch.finfo(top_scores.dtype).min)
        weights = torch.softmax(top_scores, dim=1) * valid.float()
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        return self.graph_sage(source, top_candidates, weights)

    def forward(
        self,
        sequence: torch.Tensor,
        missing: torch.Tensor,
        group_ids: torch.Tensor,
        formation_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        del group_ids, formation_ids
        sequence_features = self.sequence_pool(self.sequence_encoder(sequence))
        missing_features = self.missing_branch(missing)
        spatial_features = self.encode_spatial_batch(segment_ids)
        fused = [sequence_features, spatial_features]
        if missing_features is not None:
            fused.insert(1, missing_features)
        return self.classifier(torch.cat(fused, dim=1))
