"""Semantic-acoustic negotiation variant of the original Conv-TasNet.

The baseline files are intentionally left untouched.  This module splits the
TCN repeats into an acoustic front half and a source-specific refinement half:

    mixture -> front TCN -> rough sources -> semantic negotiation
            -> back TCN -> final masks -> waveforms

Semantic inputs are cached LLM decoder states, not frame-aligned features.
Cross-attention performs the token-to-acoustic alignment; no interpolation of
semantic tokens onto the Conv-TasNet time axis is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from Conv_TasNet import Conv1D, Conv1D_Block, ConvTrans1D, select_norm


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean over ``dim`` with True marking valid positions."""
    weight = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


class SemanticAcousticNegotiation(nn.Module):
    """Match two rough acoustic sources to two SOT semantic streams.

    Args:
        acoustic: ``[batch, speakers, channels, frames]``
        semantic: ``[batch, streams, tokens, semantic_dim]``
        semantic_mask: ``[batch, streams, tokens]``; True means valid token
        global_semantic: optional ``[batch, semantic_dim]``
    """

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
        self.global_gate = nn.Sequential(
            nn.Linear(3 * acoustic_dim, acoustic_dim),
            nn.PReLU(),
            nn.Linear(acoustic_dim, 1),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, acoustic_dim),
        )
        self.proposal_norm = nn.LayerNorm(acoustic_dim)

        nn.init.zeros_(self.local_gate.weight)
        nn.init.constant_(self.local_gate.bias, gate_bias)
        nn.init.zeros_(self.global_gate[-1].weight)
        nn.init.constant_(self.global_gate[-1].bias, gate_bias)

        # Zero initialization makes the first forward exactly the acoustic
        # rough-source path while still allowing alpha to learn immediately.
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        acoustic: torch.Tensor,
        semantic: torch.Tensor,
        semantic_mask: torch.Tensor,
        global_semantic: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        if acoustic.dim() != 4:
            raise ValueError("acoustic must have shape [B, S, C, T]")
        if semantic.dim() != 4 or semantic_mask.dim() != 3:
            raise ValueError("semantic/mask must be [B,S,L,D] and [B,S,L]")
        batch, num_sources, channels, frames = acoustic.shape
        if semantic.shape[:2] != (batch, num_sources):
            raise ValueError("The first two acoustic/semantic dimensions must match")
        if torch.any(semantic_mask.sum(dim=-1) == 0):
            raise ValueError("Every semantic stream must contain at least one valid token")

        acoustic_tokens = acoustic.permute(0, 1, 3, 2)  # [B,S,T,C]
        semantic_tokens = self.semantic_proj(semantic)    # [B,S,L,C]
        acoustic_pooled = acoustic_tokens.mean(dim=2)
        semantic_pooled = torch.stack(
            [masked_mean(semantic_tokens[:, j], semantic_mask[:, j], dim=1)
             for j in range(num_sources)],
            dim=1,
        )

        aq = F.normalize(self.acoustic_pool_proj(acoustic_pooled), dim=-1)
        sk = F.normalize(self.semantic_pool_proj(semantic_pooled), dim=-1)
        compatibility = torch.einsum("bic,bjc->bij", aq, sk)
        assignment = compatibility.softmax(dim=-1)       # [B,S,S]

        proposals = []
        attention_maps = []
        for i in range(num_sources):
            per_stream = []
            per_stream_attn = []
            query = acoustic_tokens[:, i]
            for j in range(num_sources):
                proposal_ij, attn_ij = self.cross_attention(
                    query=query,
                    key=semantic_tokens[:, j],
                    value=semantic_tokens[:, j],
                    key_padding_mask=~semantic_mask[:, j].bool(),
                    need_weights=return_diagnostics,
                    average_attn_weights=True,
                )
                per_stream.append(proposal_ij)
                if return_diagnostics:
                    per_stream_attn.append(attn_ij)
            per_stream = torch.stack(per_stream, dim=1)  # [B,S,T,C]
            proposal_i = torch.einsum(
                "bj,bjtc->btc", assignment[:, i], per_stream
            )
            proposals.append(proposal_i)
            if return_diagnostics:
                attention_maps.append(torch.stack(per_stream_attn, dim=1))

        proposal = self.proposal_norm(torch.stack(proposals, dim=1))
        proposal_bsc_t = proposal.permute(0, 1, 3, 2)
        local_input = torch.cat(
            [
                acoustic,
                proposal_bsc_t,
                acoustic * proposal_bsc_t,
                (acoustic - proposal_bsc_t).abs(),
            ],
            dim=2,
        )
        local_gate = torch.stack(
            [torch.sigmoid(self.local_gate(local_input[:, i]))
             for i in range(num_sources)],
            dim=1,
        )

        if global_semantic is None:
            global_context = semantic_pooled.mean(dim=1)
        else:
            global_context = self.global_proj(global_semantic)
        global_context = global_context[:, None].expand(-1, num_sources, -1)
        global_gate = torch.sigmoid(
            self.global_gate(
                torch.cat(
                    [acoustic_pooled, proposal.mean(dim=2), global_context],
                    dim=-1,
                )
            )
        ).unsqueeze(-1)  # [B,S,1,1]

        verification_gate = local_gate * global_gate
        updated = acoustic + self.alpha * verification_gate * proposal_bsc_t
        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "compatibility": compatibility,
                "assignment": assignment,
                "verification_gate": verification_gate,
                "attention_maps": torch.stack(attention_maps, dim=1),
            }
        return updated, diagnostics


class ConvTasNetSemantic(nn.Module):
    """Two-speaker Conv-TasNet with one semantic negotiation point."""

    def __init__(
        self,
        N=512,
        L=16,
        B=128,
        H=512,
        P=3,
        X=8,
        R=3,
        norm="gln",
        num_spks=2,
        activate="relu",
        causal=False,
        semantic_dim=2048,
        semantic_heads=4,
        front_repeats=1,
        semantic_dropout=0.0,
        gradient_checkpointing=False,
    ):
        super().__init__()
        if num_spks != 2:
            raise ValueError("The current SOT negotiation implementation requires 2 speakers")
        if not 1 <= front_repeats < R:
            raise ValueError("front_repeats must be in [1, R-1]")

        self.N = N
        self.B = B
        self.num_spks = num_spks
        self.gradient_checkpointing = gradient_checkpointing
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        self.LayerN_S = select_norm("cln", N)
        self.BottleN_S = Conv1D(N, B, 1)
        self.front_tcn = self._make_repeats(
            front_repeats, X, B, H, P, norm, causal
        )
        self.back_tcn = self._make_repeats(
            R - front_repeats, X, B, H, P, norm, causal
        )

        self.rough_mask_head = Conv1D(B, num_spks * N, 1)
        self.source_projection = Conv1D(N, B, 1)
        self.negotiation = SemanticAcousticNegotiation(
            acoustic_dim=B,
            semantic_dim=semantic_dim,
            num_heads=semantic_heads,
            dropout=semantic_dropout,
        )
        self.final_mask_head = Conv1D(B, N, 1)
        self.decoder = ConvTrans1D(N, 1, L, stride=L // 2)

        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=1),
        }
        self.activation = activations[activate]
        self.activation_type = activate

    @staticmethod
    def _make_repeats(repeats, blocks, B, H, P, norm, causal):
        return nn.Sequential(
            *[
                nn.Sequential(
                    *[
                        Conv1D_Block(
                            in_channels=B,
                            out_channels=H,
                            kernel_size=P,
                            dilation=2 ** index,
                            norm=norm,
                            causal=causal,
                        )
                        for index in range(blocks)
                    ]
                )
                for _ in range(repeats)
            ]
        )

    def forward(
        self,
        mixture: torch.Tensor,
        semantic: torch.Tensor,
        semantic_mask: torch.Tensor,
        global_semantic: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)
        if mixture.dim() != 2:
            raise ValueError("mixture must have shape [B, samples]")

        encoded = self.encoder(mixture)                         # [B,N,T]
        bottleneck = self.BottleN_S(self.LayerN_S(encoded))     # [B,B,T]
        acoustic_mid = self._run_tcn(self.front_tcn, bottleneck)

        rough_logits = self.rough_mask_head(acoustic_mid)
        rough_logits = rough_logits.view(
            mixture.shape[0], self.num_spks, self.N, -1
        )
        if self.activation_type == "softmax":
            rough_masks = rough_logits.softmax(dim=1)
        else:
            rough_masks = self.activation(rough_logits)
        rough_encoded = encoded[:, None] * rough_masks          # [B,S,N,T]
        rough_sources = torch.stack(
            [self.source_projection(rough_encoded[:, i])
             for i in range(self.num_spks)],
            dim=1,
        )                                                       # [B,S,B,T]

        updated, diagnostics = self.negotiation(
            rough_sources,
            semantic,
            semantic_mask,
            global_semantic,
            return_diagnostics=return_diagnostics,
        )
        batch, speakers, channels, frames = updated.shape
        refined = self._run_tcn(
            self.back_tcn, updated.reshape(batch * speakers, channels, frames)
        )
        refined = refined.reshape(batch, speakers, channels, frames)
        final_logits = torch.stack(
            [self.final_mask_head(refined[:, i]) for i in range(speakers)],
            dim=1,
        )                                                       # [B,S,N,T]
        if self.activation_type == "softmax":
            final_masks = final_logits.softmax(dim=1)
        else:
            final_masks = self.activation(final_logits)

        separated_encoded = encoded[:, None] * final_masks
        estimates = [
            self.decoder(separated_encoded[:, i], squeeze=True)
            for i in range(self.num_spks)
        ]
        if return_diagnostics:
            diagnostics.update(
                {
                    "rough_masks": rough_masks,
                    "final_masks": final_masks,
                    "rough_sources": rough_sources,
                }
            )
            return estimates, diagnostics
        return estimates

    def _run_tcn(self, tcn: nn.Sequential, inputs: torch.Tensor) -> torch.Tensor:
        """Run one repeat per checkpoint segment to limit full-utterance memory."""
        if self.gradient_checkpointing and self.training and len(tcn) > 0:
            return checkpoint_sequential(
                tcn, segments=len(tcn), input=inputs, use_reentrant=False
            )
        return tcn(inputs)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ConvTasNetSemantic(N=64, B=32, H=64, X=3, R=3)
    audio = torch.randn(2, 3200)
    semantics = torch.randn(2, 2, 24, 2048)
    masks = torch.ones(2, 2, 24, dtype=torch.bool)
    outputs, info = model(audio, semantics, masks, return_diagnostics=True)
    sum(x.square().mean() for x in outputs).backward()
    print([tuple(x.shape) for x in outputs])
    print({k: tuple(v.shape) for k, v in info.items() if torch.is_tensor(v)})
