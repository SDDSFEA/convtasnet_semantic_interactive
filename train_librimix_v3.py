#!/usr/bin/env python3
"""Train semantic Conv-TasNet V3 with the reduced local-gate input."""

from Conv_TasNet_Semantic_V3 import ConvTasNetSemanticV3
from train_librimix_v2 import main


if __name__ == "__main__":
    main(
        model_class=ConvTasNetSemanticV3,
        checkpoint_version=3,
        require_baseline_pretrained=True,
    )
