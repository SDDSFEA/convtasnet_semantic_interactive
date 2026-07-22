"""V3 semantic Conv-TasNet with a reduced local-gate input.

V3 keeps V2's semantic assignment, cross-attention proposal, single local
gate, and fixed residual scale alpha=1.  The only architectural change is
that the gate sees [acoustic, proposal] instead of V2's four-way comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from Conv_TasNet_Semantic_V2 import (
    ConvTasNetSemanticV2,
    SemanticAcousticNegotiationV2,
)


class SemanticAcousticNegotiationV3(SemanticAcousticNegotiationV2):
    def __init__(self, *args, gate_bias: float = -2.0, **kwargs):
        super().__init__(*args, gate_bias=gate_bias, **kwargs)
        acoustic_dim = self.local_gate.out_channels
        self.local_gate = nn.Conv1d(2 * acoustic_dim, acoustic_dim, 1)
        nn.init.zeros_(self.local_gate.weight)
        nn.init.constant_(self.local_gate.bias, gate_bias)

    def build_local_input(self, acoustic, proposal_channels):
        return torch.cat([acoustic, proposal_channels], dim=2)


class ConvTasNetSemanticV3(ConvTasNetSemanticV2):
    """ConvTasNetSemantic topology with the reduced-input V3 local gate."""

    def __init__(self, *args, semantic_dim=2048, semantic_heads=4,
                 semantic_dropout=0.0, **kwargs):
        super().__init__(
            *args,
            semantic_dim=semantic_dim,
            semantic_heads=semantic_heads,
            semantic_dropout=semantic_dropout,
            **kwargs,
        )
        self.negotiation = SemanticAcousticNegotiationV3(
            acoustic_dim=self.B,
            semantic_dim=semantic_dim,
            num_heads=semantic_heads,
            dropout=semantic_dropout,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ConvTasNetSemanticV3(N=64, B=32, H=64, X=3, R=3)
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
