"""Frozen neural modules used by ConcordTree.

The layer names and shapes intentionally match the historical checkpoints.
Do not change this module without a canonical-topology equivalence run.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from concordtree.sparse_attention import (
    _attention,
    compiled_mask_attention_forward,
    indexed_attention_forward,
)


class MLP(nn.Module):
    def __init__(self, isencoder: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 3)
        self.isencoder = isencoder

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.fc1(value))
        value = F.relu(self.fc2(value))
        return value if self.isencoder else self.fc3(value)


class QuartetMLP(MLP):
    """Frozen MLP used in every r4 view."""

    def __init__(self) -> None:
        super().__init__(isencoder=False)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, token_dim: int, num_heads: int) -> None:
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.o_proj = nn.Linear(token_dim, token_dim)
        self.F = _attention.apply

    def forward(
        self,
        value: torch.Tensor,
        coeff_blocks: torch.Tensor,
        quartet_matrix: torch.Tensor,
        active_indices: torch.Tensor | None = None,
        active_counts: torch.Tensor | None = None,
        pair_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, token_dim = value.size()
        q = self.q_proj(value).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(value).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(value).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        if active_indices is None and active_counts is None:
            output = self.F(q, k, v, coeff_blocks, quartet_matrix)
        elif active_indices is not None and active_counts is not None:
            if pair_masks is None:
                output = indexed_attention_forward(
                    q, k, v, active_indices, active_counts, quartet_matrix
                )
            else:
                output = compiled_mask_attention_forward(
                    q, k, v, active_indices, active_counts, pair_masks
                )
        else:
            raise ValueError("active_indices and active_counts must be provided together")
        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, token_dim
        )
        return self.o_proj(output)


class FeedForwardNetwork(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(token_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, token_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(value)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(token_dim, num_heads)
        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = FeedForwardNetwork(token_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(
        self,
        value: torch.Tensor,
        coeff_blocks: torch.Tensor,
        quartet_matrix: torch.Tensor,
        active_indices: torch.Tensor | None = None,
        active_counts: torch.Tensor | None = None,
        pair_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = self.norm1(
            value
            + self.self_attention(
                value,
                coeff_blocks,
                quartet_matrix,
                active_indices,
                active_counts,
                pair_masks,
            )
        )
        return self.norm2(value + self.ffn(value))


class QuartFormer(nn.Module):
    def __init__(self, species_num: int, num_layers: int = 3) -> None:
        super().__init__()
        token_dim = 256
        self.species_num = species_num
        self.mlp_layer = MLP(True)
        self.input_layer = nn.Sequential(
            nn.Linear(self.species_num + 128, token_dim), nn.ReLU()
        )
        self.layers = nn.ModuleList(
            TransformerEncoderBlock(token_dim, 16, 16)
            for _ in range(num_layers)
        )
        self.classifier = nn.Linear(token_dim, 3)

    def forward(
        self,
        value: torch.Tensor,
        coeff_blocks: torch.Tensor,
        quartet_matrix: torch.Tensor,
        active_indices: torch.Tensor | None = None,
        active_counts: torch.Tensor | None = None,
        pair_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, feature_dim = value.shape
        value = value.view(batch_size * sequence_length, feature_dim)
        pattern_embedding = self.mlp_layer(value[:, -256:])
        value = torch.cat(
            [value[:, : self.species_num], pattern_embedding], dim=-1
        )
        value = self.input_layer(value).view(batch_size, sequence_length, -1)
        for layer in self.layers:
            value = layer(
                value,
                coeff_blocks,
                quartet_matrix,
                active_indices,
                active_counts,
                pair_masks,
            )
        return self.classifier(value)
