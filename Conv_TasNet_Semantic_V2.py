"""V2 semantic Conv-TasNet with trainable, one-to-one semantic matching.

Differences from V1:
  * no trainable alpha suppression (semantic residual scale is fixed to 1);
  * one local verification gate only;
  * direct/swap permutation softmax gives a doubly-stochastic 2x2 assignment;
  * lightweight diagnostics expose permutation logits without attention maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Conv_TasNet_Semantic import ConvTasNetSemantic, masked_mean


class SemanticAcousticNegotiationV2(nn.Module):
    def __init__(
        self,
        acoustic_dim: int,
        semantic_dim: int = 2048,
        num_heads: int = 4,
        dropout: float = 0.0,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        if acoustic_dim % num_heads:
            raise ValueError("acoustic_dim must be divisible by num_heads")

        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, acoustic_dim),
        )
        self.acoustic_pool_proj = nn.Linear(acoustic_dim, acoustic_dim)
        self.semantic_pool_proj = nn.Linear(acoustic_dim, acoustic_dim)
        self.cross_attention = nn.MultiheadAttention(
            acoustic_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.local_gate = nn.Conv1d(4 * acoustic_dim, acoustic_dim, 1)
        self.proposal_norm = nn.LayerNorm(acoustic_dim)

        # A single conservative gate protects the acoustic path without cutting
        # gradients to all semantic modules as zero-initialized alpha did.
        nn.init.zeros_(self.local_gate.weight)
        nn.init.constant_(self.local_gate.bias, gate_bias)
        self.register_buffer("alpha", torch.ones(()), persistent=True)

    def forward(
        self,
        acoustic: torch.Tensor,
        semantic: torch.Tensor,
        semantic_mask: torch.Tensor,
        global_semantic: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        del global_semantic  # V2 intentionally uses only the local gate.
        if acoustic.dim() != 4:
            raise ValueError("acoustic must have shape [B,2,C,T]")
        if semantic.dim() != 4 or semantic_mask.dim() != 3:
            raise ValueError("semantic/mask must be [B,2,L,D] and [B,2,L]")
        batch, num_sources, _, _ = acoustic.shape
        if num_sources != 2 or semantic.shape[:2] != (batch, 2):
            raise ValueError("V2 negotiation requires exactly two sources/streams")
        if torch.any(semantic_mask.sum(dim=-1) == 0):
            raise ValueError("Every semantic stream needs at least one valid token")

        acoustic_tokens = acoustic.permute(0, 1, 3, 2)
        semantic_tokens = self.semantic_proj(semantic)
        acoustic_pooled = acoustic_tokens.mean(dim=2)
        semantic_pooled = torch.stack(
            [
                masked_mean(semantic_tokens[:, j], semantic_mask[:, j], dim=1)
                for j in range(2)
            ],
            dim=1,
        )

        acoustic_key = F.normalize(
            self.acoustic_pool_proj(acoustic_pooled), dim=-1
        )
        semantic_key = F.normalize(
            self.semantic_pool_proj(semantic_pooled), dim=-1
        )
        compatibility = torch.einsum("bic,bjc->bij", acoustic_key, semantic_key)
        permutation_logits = torch.stack(
            [
                compatibility[:, 0, 0] + compatibility[:, 1, 1],
                compatibility[:, 0, 1] + compatibility[:, 1, 0],
            ],
            dim=-1,
        )
        permutation_probability = permutation_logits.softmax(dim=-1)
        direct, swap = permutation_probability.unbind(dim=-1)
        assignment = torch.stack(
            [
                torch.stack([direct, swap], dim=-1),
                torch.stack([swap, direct], dim=-1),
            ],
            dim=1,
        )

        proposals = []
        for source_index in range(2):
            per_stream = []
            for semantic_index in range(2):
                proposal, _ = self.cross_attention(
                    query=acoustic_tokens[:, source_index],
                    key=semantic_tokens[:, semantic_index],
                    value=semantic_tokens[:, semantic_index],
                    key_padding_mask=~semantic_mask[:, semantic_index].bool(),
                    need_weights=False,
                )
                per_stream.append(proposal)
            per_stream = torch.stack(per_stream, dim=1)
            proposals.append(
                torch.einsum(
                    "bj,bjtc->btc", assignment[:, source_index], per_stream
                )
            )

        proposal = self.proposal_norm(torch.stack(proposals, dim=1))
        proposal_channels = proposal.permute(0, 1, 3, 2)
        local_input = torch.cat(
            [
                acoustic,
                proposal_channels,
                acoustic * proposal_channels,
                (acoustic - proposal_channels).abs(),
            ],
            dim=2,
        )
        local_gate = torch.stack(
            [
                torch.sigmoid(self.local_gate(local_input[:, source_index]))
                for source_index in range(2)
            ],
            dim=1,
        )
        updated = acoustic + local_gate * proposal_channels

        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "compatibility": compatibility,
                "assignment": assignment,
                "permutation_logits": permutation_logits,
                "permutation_probability": permutation_probability,
                "verification_gate": local_gate,
                "semantic_residual": local_gate * proposal_channels,
            }
        return updated, diagnostics


class ConvTasNetSemanticV2(ConvTasNetSemantic):
    """ConvTasNetSemantic topology with the V2 negotiation module."""

    def __init__(self, *args, semantic_dim=2048, semantic_heads=4,
                 semantic_dropout=0.0, **kwargs):
        super().__init__(
            *args,
            semantic_dim=semantic_dim,
            semantic_heads=semantic_heads,
            semantic_dropout=semantic_dropout,
            **kwargs,
        )
        self.negotiation = SemanticAcousticNegotiationV2(
            acoustic_dim=self.B,
            semantic_dim=semantic_dim,
            num_heads=semantic_heads,
            dropout=semantic_dropout,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ConvTasNetSemanticV2(N=64, B=32, H=64, X=3, R=3)
    audio = torch.randn(2, 3200)
    semantic = torch.randn(2, 2, 24, 2048)
    mask = torch.ones(2, 2, 24, dtype=torch.bool)
    outputs, diagnostics = model(
        audio, semantic, mask, return_diagnostics=True
    )
    (sum(x.square().mean() for x in outputs)
     + diagnostics["permutation_logits"].square().mean()).backward()
    print([tuple(x.shape) for x in outputs])
    print({key: tuple(value.shape) for key, value in diagnostics.items()})
