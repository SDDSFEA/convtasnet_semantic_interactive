#!/usr/bin/env python3
"""Diagnose semantic direct/swap predictions against waveform PIT labels.

This script is deliberately read-only with respect to the model and trainer. It
uses a forward pre-hook to observe the inputs of the negotiation block, then
recomputes the model's existing pooled-cosine compatibility scores.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Conv_TasNet_Semantic import ConvTasNetSemantic, masked_mean
from Conv_TasNet_Semantic_V2 import ConvTasNetSemanticV2
from Conv_TasNet_Semantic_V3 import ConvTasNetSemanticV3
from SI_SNR import sisnr
from train_librimix import (
    LibriMixFullUtterance,
    align_for_sisnr,
    collate_one,
    ensure_batched,
    move_batch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Semantic last.pt or best.pt")
    parser.add_argument(
        "--data-root",
        default="/home/zt/Desktop/STL/espnet/egs2/librimix/sot_asr1/data",
    )
    parser.add_argument(
        "--semantic-root",
        default=(
            "/home/zt/Desktop/STL/Multi-talker-ASR-with-LLMs/"
            "semantic_features/libri2mix_clean_offset"
        ),
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args_path = checkpoint_path.parent / "arguments.json"
    saved_args = json.loads(saved_args_path.read_text()) if saved_args_path.exists() else {}
    model_class = {
        2: ConvTasNetSemanticV2,
        3: ConvTasNetSemanticV3,
    }.get(checkpoint.get("version"), ConvTasNetSemantic)
    model = model_class(
        N=saved_args.get("N", 512),
        L=saved_args.get("L", 16),
        B=saved_args.get("B", 128),
        H=saved_args.get("H", 512),
        P=saved_args.get("P", 3),
        X=saved_args.get("X", 8),
        R=saved_args.get("R", 3),
        num_spks=2,
        gradient_checkpointing=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


def compatibility_from_inputs(negotiation, acoustic, semantic, semantic_mask):
    acoustic_tokens = acoustic.permute(0, 1, 3, 2)
    semantic_tokens = negotiation.semantic_proj(semantic)
    acoustic_pooled = acoustic_tokens.mean(dim=2)
    semantic_pooled = torch.stack(
        [
            masked_mean(semantic_tokens[:, j], semantic_mask[:, j], dim=1)
            for j in range(semantic.shape[1])
        ],
        dim=1,
    )
    acoustic_key = F.normalize(
        negotiation.acoustic_pool_proj(acoustic_pooled), dim=-1
    )
    semantic_key = F.normalize(
        negotiation.semantic_pool_proj(semantic_pooled), dim=-1
    )
    return torch.einsum("bic,bjc->bij", acoustic_key, semantic_key)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu if intended")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    dataset = LibriMixFullUtterance(
        args.data_root,
        args.split,
        f"{args.semantic_root}/{args.split}",
        args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_one,
    )

    captured = {}

    def capture_negotiation_inputs(_module, inputs):
        captured["acoustic"] = inputs[0]
        captured["semantic"] = inputs[1]
        captured["semantic_mask"] = inputs[2]

    hook = model.negotiation.register_forward_pre_hook(capture_negotiation_inputs)
    rows = []
    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader, 1):
                batch = move_batch(batch, device)
                estimates = model(
                    batch["mix"],
                    batch["semantic"],
                    batch["semantic_mask"],
                    batch["global_semantic"],
                )
                estimates = align_for_sisnr(ensure_batched(estimates), batch)
                compatibility = compatibility_from_inputs(
                    model.negotiation,
                    captured.pop("acoustic"),
                    captured.pop("semantic"),
                    captured.pop("semantic_mask"),
                )
                permutation_logits = torch.stack(
                    [
                        compatibility[:, 0, 0] + compatibility[:, 1, 1],
                        compatibility[:, 0, 1] + compatibility[:, 1, 0],
                    ],
                    dim=-1,
                )
                probability = permutation_logits.softmax(dim=-1)
                predicted = permutation_logits.argmax(dim=-1)

                direct = (sisnr(estimates[0], batch["ref"][0]) +
                          sisnr(estimates[1], batch["ref"][1])) / 2
                swapped = (sisnr(estimates[0], batch["ref"][1]) +
                           sisnr(estimates[1], batch["ref"][0])) / 2
                pit_scores = torch.stack([direct, swapped], dim=-1)
                pit_target = pit_scores.argmax(dim=-1)
                pit_margin = (direct - swapped).abs()
                rows.append(
                    {
                        "id": batch["id"],
                        "predicted": int(predicted.item()),
                        "pit_target": int(pit_target.item()),
                        "correct": bool(predicted.eq(pit_target).item()),
                        "direct_probability": float(probability[0, 0].item()),
                        "pit_margin_db": float(pit_margin.item()),
                    }
                )
                if index % 100 == 0:
                    print(f"processed {index}/{len(loader)}", flush=True)
    finally:
        hook.remove()

    def accuracy(selected):
        return (sum(row["correct"] for row in selected) / len(selected)
                if selected else math.nan)

    confident_1 = [row for row in rows if row["pit_margin_db"] > 1.0]
    confident_3 = [row for row in rows if row["pit_margin_db"] > 3.0]
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "count": len(rows),
        "accuracy": accuracy(rows),
        "accuracy_margin_gt_1db": accuracy(confident_1),
        "count_margin_gt_1db": len(confident_1),
        "accuracy_margin_gt_3db": accuracy(confident_3),
        "count_margin_gt_3db": len(confident_3),
        "mean_pit_margin_db": (
            sum(row["pit_margin_db"] for row in rows) / len(rows)
            if rows else math.nan
        ),
        "mean_direct_probability": (
            sum(row["direct_probability"] for row in rows) / len(rows)
            if rows else math.nan
        ),
        "label_convention": "0=direct, 1=swap",
        "mapping_assumption": "ref[0]/ref[1] correspond to semantic Z1/Z2",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"summary": summary, "utterances": rows}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
