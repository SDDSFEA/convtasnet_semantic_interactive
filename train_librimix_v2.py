#!/usr/bin/env python3
"""Train semantic Conv-TasNet V2 with PIT-supervised permutation matching."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Conv_TasNet_Semantic_V2 import ConvTasNetSemanticV2
from SI_SNR import sisnr
from train_librimix import (
    LibriMixFullUtterance,
    align_for_sisnr,
    collate_one,
    ensure_batched,
    move_batch,
)


LOG = logging.getLogger("train_librimix_v2")


def load_acoustic_pretrained(model, checkpoint_path, require_baseline=False):
    """Load every non-negotiation tensor and keep V2 negotiation fresh.

    This is initialization, not resume: optimizer state, epoch, global step,
    and all ``negotiation.*`` tensors in the checkpoint are intentionally
    ignored.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source_state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    if not isinstance(source_state, dict):
        raise TypeError(
            "Acoustic pretrained checkpoint must be a state_dict or contain "
            "a 'model' state_dict"
        )
    if require_baseline and any(
        key.startswith("negotiation.") for key in source_state
    ):
        raise ValueError(
            "V3 initialization requires a baseline checkpoint without "
            "negotiation.* tensors; a semantic checkpoint was provided"
        )

    target_state = model.state_dict()
    if require_baseline:
        mapped_state = {}
        fresh_keys = []
        derived_keys = []
        front_repeats = len(model.front_tcn)
        for target_key, target_value in target_state.items():
            if target_key.startswith("negotiation."):
                fresh_keys.append(target_key)
                continue
            source_key = target_key
            if target_key.startswith("front_tcn."):
                source_key = "separation." + target_key[len("front_tcn."):]
            elif target_key.startswith("back_tcn."):
                suffix = target_key[len("back_tcn."):]
                repeat, remainder = suffix.split(".", 1)
                source_key = f"separation.{int(repeat) + front_repeats}.{remainder}"
            elif target_key.startswith("rough_mask_head."):
                source_key = "gen_masks." + target_key.split(".", 1)[1]
            elif target_key.startswith("final_mask_head."):
                baseline_key = "gen_masks." + target_key.split(".", 1)[1]
                if baseline_key not in source_state:
                    raise RuntimeError(
                        f"Baseline checkpoint is missing {baseline_key}"
                    )
                baseline_value = source_state[baseline_key]
                if baseline_value.shape[0] != 2 * target_value.shape[0]:
                    raise RuntimeError(
                        f"Cannot derive {target_key} from {baseline_key}: "
                        f"{tuple(target_value.shape)} vs "
                        f"{tuple(baseline_value.shape)}"
                    )
                mapped_state[target_key] = baseline_value.reshape(
                    2, target_value.shape[0], *target_value.shape[1:]
                ).mean(dim=0)
                derived_keys.append(target_key)
                continue
            elif target_key.startswith("source_projection."):
                fresh_keys.append(target_key)
                continue

            if source_key not in source_state:
                raise RuntimeError(
                    f"Baseline checkpoint is missing {source_key} needed for "
                    f"{target_key}"
                )
            if source_state[source_key].shape != target_value.shape:
                raise RuntimeError(
                    f"Baseline shape mismatch for {target_key} <- {source_key}: "
                    f"{tuple(target_value.shape)} vs "
                    f"{tuple(source_state[source_key].shape)}"
                )
            mapped_state[target_key] = source_state[source_key]

        incompatible = model.load_state_dict(mapped_state, strict=False)
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != set(fresh_keys):
            raise RuntimeError(
                "Unexpected baseline mapping result: "
                f"unexpected={list(incompatible.unexpected_keys)}, "
                f"missing={list(incompatible.missing_keys)}"
            )
        LOG.info(
            "initialized V3 from baseline %s: exact=%d derived=%d fresh=%d",
            checkpoint_path, len(mapped_state) - len(derived_keys),
            len(derived_keys), len(fresh_keys),
        )
        return {
            "path": str(Path(checkpoint_path).resolve()),
            "source_model": "baseline",
            "exact_mapped_tensors": len(mapped_state) - len(derived_keys),
            "derived_mask_tensors": len(derived_keys),
            "fresh_tensors": len(fresh_keys),
            "fresh_tensor_prefixes": ["source_projection.", "negotiation."],
        }

    acoustic_keys = [
        key for key in target_state if not key.startswith("negotiation.")
    ]
    missing = [key for key in acoustic_keys if key not in source_state]
    mismatched = [
        key for key in acoustic_keys
        if key in source_state
        and source_state[key].shape != target_state[key].shape
    ]
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if mismatched:
            details.append(
                "shape_mismatch="
                + repr([
                    (
                        key,
                        tuple(source_state[key].shape),
                        tuple(target_state[key].shape),
                    )
                    for key in mismatched[:5]
                ])
            )
        raise RuntimeError(
            "Acoustic checkpoint is incompatible with the requested V2 "
            "architecture: " + "; ".join(details)
        )

    acoustic_state = {key: source_state[key] for key in acoustic_keys}
    incompatible = model.load_state_dict(acoustic_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_negotiation_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("negotiation.")
    ]
    if unexpected or non_negotiation_missing:
        raise RuntimeError(
            "Unexpected partial-load result: "
            f"unexpected={unexpected}, missing={non_negotiation_missing}"
        )
    LOG.info(
        "initialized %d acoustic tensors from %s; kept %d negotiation "
        "tensors freshly initialized",
        len(acoustic_state), checkpoint_path,
        sum(key.startswith("negotiation.") for key in target_state),
    )
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "loaded_acoustic_tensors": len(acoustic_state),
        "reinitialized_negotiation_tensors": sum(
            key.startswith("negotiation.") for key in target_state
        ),
    }


def pit_loss_and_target(estimates, batch):
    """Return PIT SI-SNR loss, direct/swap target, and score margin."""
    references = batch["ref"]
    direct = (
        sisnr(estimates[0], references[0])
        + sisnr(estimates[1], references[1])
    ) / 2
    swapped = (
        sisnr(estimates[0], references[1])
        + sisnr(estimates[1], references[0])
    ) / 2
    scores = torch.stack([direct, swapped], dim=-1)
    best_score, target = scores.max(dim=-1)
    return -best_score.mean(), target.detach(), (direct - swapped).abs().detach()


def module_l2_norm(module, gradients=False):
    total = None
    for parameter in module.parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            continue
        squared = value.detach().float().square().sum()
        total = squared if total is None else total + squared
    return math.sqrt(total.item()) if total is not None else 0.0


def make_optimizer(model, lr, weight_decay):
    negotiation_ids = {id(parameter) for parameter in model.negotiation.parameters()}
    acoustic_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in negotiation_ids
    ]
    negotiation_parameters = list(model.negotiation.parameters())
    optimizer = torch.optim.Adam(
        [
            {
                "params": acoustic_parameters,
                "weight_decay": weight_decay,
                "name": "acoustic",
            },
            {
                "params": negotiation_parameters,
                "weight_decay": 0.0,
                "name": "semantic_negotiation",
            },
        ],
        lr=lr,
    )
    return optimizer


def run_epoch(
    model,
    loader,
    device,
    lambda_match,
    optimizer=None,
    writer=None,
    split="train",
    global_step=0,
    gradient_accumulation_steps=1,
    clip_norm=5.0,
    log_interval=100,
    step_callback=None,
):
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    totals = {
        "loss": 0.0,
        "separation_loss": 0.0,
        "matching_loss": 0.0,
        "correct": 0,
        "count": 0,
        "pit_margin_db": 0.0,
        "entropy": 0.0,
    }
    for step, batch in enumerate(loader, 1):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            estimates, diagnostics = model(
                batch["mix"],
                batch["semantic"],
                batch["semantic_mask"],
                batch["global_semantic"],
                return_diagnostics=True,
            )
            estimates = align_for_sisnr(ensure_batched(estimates), batch)
            separation_loss, pit_target, pit_margin = pit_loss_and_target(
                estimates, batch
            )
            permutation_logits = diagnostics["permutation_logits"]
            matching_loss = F.cross_entropy(permutation_logits, pit_target)
            loss = separation_loss + lambda_match * matching_loss

        update_now = False
        if training:
            (loss / gradient_accumulation_steps).backward()
            update_now = (
                step % gradient_accumulation_steps == 0 or step == len(loader)
            )
            if update_now:
                remainder = step % gradient_accumulation_steps
                if step == len(loader) and remainder:
                    correction = gradient_accumulation_steps / remainder
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)

                semantic_grad_norm = module_l2_norm(
                    model.negotiation, gradients=True
                )
                matching_grad_norm = math.sqrt(
                    sum(
                        parameter.grad.detach().float().square().sum().item()
                        for module in (
                            model.negotiation.acoustic_pool_proj,
                            model.negotiation.semantic_pool_proj,
                        )
                        for parameter in module.parameters()
                        if parameter.grad is not None
                    )
                )
                total_grad_norm = float(clip_grad_norm_(model.parameters(), clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if writer is not None:
                    writer.add_scalar(
                        "grad_norm/semantic_negotiation", semantic_grad_norm,
                        global_step,
                    )
                    writer.add_scalar(
                        "grad_norm/matching_projections", matching_grad_norm,
                        global_step,
                    )
                    writer.add_scalar(
                        "grad_norm/model_before_clip", total_grad_norm,
                        global_step,
                    )
                    writer.add_scalar(
                        "parameter_norm/semantic_negotiation",
                        module_l2_norm(model.negotiation),
                        global_step,
                    )
                    for name, module in (
                        ("semantic_projection", model.negotiation.semantic_proj),
                        ("acoustic_pool_projection", model.negotiation.acoustic_pool_proj),
                        ("semantic_pool_projection", model.negotiation.semantic_pool_proj),
                        ("cross_attention", model.negotiation.cross_attention),
                        ("local_gate", model.negotiation.local_gate),
                    ):
                        writer.add_scalar(
                            f"parameter_norm/{name}", module_l2_norm(module),
                            global_step,
                        )

        with torch.no_grad():
            probability = permutation_logits.softmax(dim=-1)
            predicted = permutation_logits.argmax(dim=-1)
            correct = predicted.eq(pit_target)
            entropy = -(
                probability * probability.clamp_min(1e-8).log()
            ).sum(dim=-1)
            count = pit_target.numel()
            totals["loss"] += loss.item() * count
            totals["separation_loss"] += separation_loss.item() * count
            totals["matching_loss"] += matching_loss.item() * count
            totals["correct"] += correct.sum().item()
            totals["count"] += count
            totals["pit_margin_db"] += pit_margin.sum().item()
            totals["entropy"] += entropy.sum().item()

        if writer is not None and (not training or update_now):
            writer.add_scalar(f"loss/{split}_total", loss.item(), global_step)
            writer.add_scalar(
                f"loss/{split}_separation", separation_loss.item(), global_step
            )
            writer.add_scalar(
                f"loss/{split}_matching", matching_loss.item(), global_step
            )
            writer.add_scalar(
                f"matching/{split}_accuracy_step",
                correct.float().mean().item(),
                global_step,
            )
            writer.add_scalar(
                f"matching/{split}_entropy_step", entropy.mean().item(), global_step
            )
            writer.add_scalar(
                f"matching/{split}_pit_margin_db_step",
                pit_margin.mean().item(),
                global_step,
            )
            writer.add_scalar(
                f"gate/{split}_mean",
                diagnostics["verification_gate"].mean().item(),
                global_step,
            )

        if step == 1 or step % log_interval == 0:
            LOG.info(
                "%s step=%d/%d total=%.4f sep=%.4f match=%.4f acc=%.3f",
                split, step, len(loader), loss.item(), separation_loss.item(),
                matching_loss.item(), correct.float().mean().item(),
            )

        if training and update_now and step_callback is not None:
            denominator = max(totals["count"], 1)
            running_metrics = {
                "loss": totals["loss"] / denominator,
                "separation_loss": totals["separation_loss"] / denominator,
                "matching_loss": totals["matching_loss"] / denominator,
                "matching_accuracy": totals["correct"] / denominator,
                "mean_pit_margin_db": totals["pit_margin_db"] / denominator,
                "mean_permutation_entropy": totals["entropy"] / denominator,
            }
            step_callback(step, global_step, running_metrics)
            model.train(True)

    denominator = max(totals["count"], 1)
    metrics = {
        "loss": totals["loss"] / denominator,
        "separation_loss": totals["separation_loss"] / denominator,
        "matching_loss": totals["matching_loss"] / denominator,
        "matching_accuracy": totals["correct"] / denominator,
        "mean_pit_margin_db": totals["pit_margin_db"] / denominator,
        "mean_permutation_entropy": totals["entropy"] / denominator,
    }
    LOG.info("%s metrics=%s", split, json.dumps(metrics, sort_keys=True))
    return metrics, global_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--resume-lr", type=float,
        help=(
            "Override the checkpoint learning rate when resuming. By default, "
            "resume preserves the optimizer and scheduler learning rate."
        ),
    )
    parser.add_argument(
        "--lr-patience", type=int, default=3,
        help=(
            "Number of completed epochs without dev-loss improvement to "
            "tolerate before reducing LR."
        ),
    )
    parser.add_argument(
        "--lr-factor", type=float, default=0.5,
        help="Multiplier applied when the validation loss plateaus.",
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6,
        help="Lower bound for the automatically reduced learning rate.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-match", type=float, default=0.02)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument(
        "--val-interval-steps",
        type=int,
        default=10000,
        help=(
            "Validate and save checkpoints every N optimizer updates. "
            "Set to 0 to validate only at epoch boundaries."
        ),
    )
    parser.add_argument("--device", default="cuda")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--resume",
        help="Resume a V2 run including model, optimizer, epoch, and step.",
    )
    initialization.add_argument(
        "--acoustic-pretrained-checkpoint",
        "--acoustic-pretrained",
        dest="acoustic_pretrained_checkpoint",
        help=(
            "Initialize only non-negotiation acoustic tensors from a V1/V2 "
            "checkpoint. The V2 negotiation module remains freshly initialized."
        ),
    )
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--P", type=int, default=3)
    parser.add_argument("--X", type=int, default=8)
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main(model_class=ConvTasNetSemanticV2, checkpoint_version=2,
         require_baseline_pretrained=False):
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation must be at least 1")
    if args.lambda_match < 0:
        raise ValueError("lambda-match must be non-negative")
    if args.val_interval_steps < 0:
        raise ValueError("val-interval-steps must be non-negative")
    if args.lr_patience < 0:
        raise ValueError("lr-patience must be non-negative")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("lr-factor must be between 0 and 1")
    if args.min_lr < 0.0 or args.min_lr > args.lr:
        raise ValueError("min-lr must be non-negative and no greater than lr")
    if args.resume_lr is not None and args.resume_lr <= 0.0:
        raise ValueError("resume-lr must be positive")
    if (
        require_baseline_pretrained
        and not args.acoustic_pretrained_checkpoint
        and not args.resume
    ):
        raise ValueError(
            "V3 training requires --acoustic-pretrained-checkpoint pointing "
            "to a trained baseline checkpoint"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_arguments = vars(args).copy()
    run_arguments.update(model="semantic", version=checkpoint_version)
    (output_dir / "arguments.json").write_text(
        json.dumps(run_arguments, indent=2), encoding="utf-8"
    )
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    train_dataset = LibriMixFullUtterance(
        args.data_root, "train", f"{args.semantic_root}/train", args.train_limit
    )
    dev_dataset = LibriMixFullUtterance(
        args.data_root, "dev", f"{args.semantic_root}/dev", args.dev_limit
    )
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_one,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_one,
    )

    model = model_class(
        N=args.N, L=args.L, B=args.B, H=args.H, P=args.P, X=args.X, R=args.R,
        num_spks=2, gradient_checkpointing=args.gradient_checkpointing,
    )
    initialization_info = None
    if args.acoustic_pretrained_checkpoint:
        initialization_info = load_acoustic_pretrained(
            model, args.acoustic_pretrained_checkpoint,
            require_baseline=require_baseline_pretrained,
        )
    model.to(device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    start_epoch, global_step, best_dev = 1, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.resume_lr is not None:
            restored_lrs = [group["lr"] for group in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = args.resume_lr
            LOG.info(
                "resume optimizer learning rates overridden: %s -> %s",
                restored_lrs,
                [group["lr"] for group in optimizer.param_groups],
            )
        start_epoch = checkpoint["epoch"] + (
            1 if checkpoint.get("epoch_complete", True) else 0
        )
        global_step = checkpoint.get("global_step", 0)
        best_dev = checkpoint.get("best_dev", best_dev)
        initialization_info = checkpoint.get("initialization")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    if args.resume and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    LOG.info(
        "params=%.2fM train=%d dev=%d lambda_match=%g semantic_weight_decay=0 "
        "initialization=%s",
        sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        len(train_dataset), len(dev_dataset), args.lambda_match,
        "resume" if args.resume else (
            "acoustic_pretrained" if args.acoustic_pretrained_checkpoint
            else "random"
        ),
    )
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            def validate_and_save(train_metrics, current_global_step,
                                  epoch_complete):
                nonlocal best_dev
                with torch.no_grad():
                    dev_metrics, _ = run_epoch(
                        model, dev_loader, device, args.lambda_match,
                        writer=None, split="dev",
                        global_step=current_global_step,
                    )
                for split, metrics in (
                    ("train", train_metrics), ("dev", dev_metrics)
                ):
                    for name, value in metrics.items():
                        writer.add_scalar(
                            f"validation/{split}_{name}",
                            value,
                            current_global_step,
                        )
                old_lrs = [group["lr"] for group in optimizer.param_groups]
                if epoch_complete:
                    scheduler.step(dev_metrics["loss"])
                new_lrs = [group["lr"] for group in optimizer.param_groups]
                for group_index, lr in enumerate(new_lrs):
                    writer.add_scalar(
                        f"optimizer/learning_rate_group_{group_index}",
                        lr,
                        current_global_step,
                    )
                if new_lrs != old_lrs:
                    LOG.info(
                        "automatic learning-rate reduction: %s -> %s",
                        old_lrs, new_lrs,
                    )
                writer.flush()

                state = {
                    "version": checkpoint_version,
                    "epoch": epoch,
                    "epoch_complete": epoch_complete,
                    "global_step": current_global_step,
                    "best_dev": min(best_dev, dev_metrics["loss"]),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "dev_metrics": dev_metrics,
                    "initialization": initialization_info,
                }
                torch.save(state, output_dir / "last.pt")
                if dev_metrics["loss"] < best_dev:
                    best_dev = dev_metrics["loss"]
                    state["best_dev"] = best_dev
                    torch.save(state, output_dir / "best.pt")
                LOG.info(
                    "validation epoch=%d global_step=%d dev_loss=%.4f "
                    "best=%.4f epoch_complete=%s",
                    epoch, current_global_step, dev_metrics["loss"],
                    best_dev, epoch_complete,
                )
                return dev_metrics

            def maybe_validate(_epoch_step, current_global_step,
                               running_train_metrics):
                if (
                    args.val_interval_steps > 0
                    and current_global_step % args.val_interval_steps == 0
                ):
                    validate_and_save(
                        running_train_metrics,
                        current_global_step,
                        epoch_complete=False,
                    )

            train_metrics, global_step = run_epoch(
                model, train_loader, device, args.lambda_match,
                optimizer=optimizer, writer=writer, split="train",
                global_step=global_step,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                clip_norm=args.clip_norm,
                step_callback=maybe_validate,
            )
            # Always validate/save at the epoch boundary, even if an interval
            # validation happened on the final update, so resume advances.
            dev_metrics = validate_and_save(
                train_metrics, global_step, epoch_complete=True
            )
            for split, metrics in (
                ("train", train_metrics), ("dev", dev_metrics)
            ):
                for name, value in metrics.items():
                    writer.add_scalar(f"epoch/{split}_{name}", value, epoch)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
