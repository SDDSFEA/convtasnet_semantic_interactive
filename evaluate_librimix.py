#!/usr/bin/env python3
"""Evaluate a train_librimix.py checkpoint on a full-utterance split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Conv_TasNet import ConvTasNet
from Conv_TasNet_Semantic import ConvTasNetSemantic
from Conv_TasNet_Semantic_V2 import ConvTasNetSemanticV2
from Conv_TasNet_Semantic_V3 import ConvTasNetSemanticV3
from SI_SNR import si_snr_loss
from train_librimix import (
    LibriMixFullUtterance,
    collate_one,
    forward_model,
    move_batch,
)


LOG = logging.getLogger("evaluate_librimix")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--semantic-root",
        help=(
            "Override the semantic-feature root saved in arguments.json; "
            "the split name is appended automatically"
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    checkpoint_path = Path(args.checkpoint)
    saved_args_path = checkpoint_path.parent / "arguments.json"
    saved_args = json.loads(saved_args_path.read_text())
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_kind = saved_args.get(
        "model", "semantic" if checkpoint.get("version") else "baseline"
    )

    common = {
        name: saved_args.get(name, default)
        for name, default in (
            ("N", 512), ("L", 16), ("B", 128), ("H", 512),
            ("P", 3), ("X", 8), ("R", 3),
        )
    }
    common["num_spks"] = 2
    if model_kind == "baseline":
        model = ConvTasNet(**common, gradient_checkpointing=False)
        semantic_root = None
    else:
        version = checkpoint.get("version")
        model_class = {
            2: ConvTasNetSemanticV2,
            3: ConvTasNetSemanticV3,
        }.get(version, ConvTasNetSemantic)
        model = model_class(**common, gradient_checkpointing=False)
        semantic_base = args.semantic_root or saved_args["semantic_root"]
        semantic_root = str(Path(semantic_base) / args.split)

    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    dataset = LibriMixFullUtterance(
        args.data_root, args.split, semantic_root, args.limit
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_one,
    )
    separated_total = 0.0
    mixture_total = 0.0
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            batch = move_batch(batch, device)
            estimates = forward_model(model, batch, model_kind)
            separated_sisnr = -si_snr_loss(estimates, batch).item()
            mixture = batch["mix"][..., : estimates[0].shape[-1]]
            mixture_sisnr = -si_snr_loss([mixture, mixture], batch).item()
            separated_total += separated_sisnr
            mixture_total += mixture_sisnr
            if step == 1 or step % 100 == 0:
                LOG.info(
                    "step=%d/%d id=%s SI-SNR=%.4f SI-SNRi=%.4f",
                    step, len(loader), batch["id"], separated_sisnr,
                    separated_sisnr - mixture_sisnr,
                )

    count = len(loader)
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "data_root": str(Path(args.data_root).resolve()),
        "split": args.split,
        "utterances": count,
        "si_snr_db": separated_total / count,
        "mixture_si_snr_db": mixture_total / count,
        "si_snri_db": (separated_total - mixture_total) / count,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
