#!/usr/bin/env python3
"""Train baseline or semantic Conv-TasNet on prepared ESPnet LibriMix.

The first reproducible version intentionally uses full utterances with batch
size one.  This keeps baseline and semantic experiments on identical audio and
prevents complete-utterance ASR semantics from leaking into random 4 s chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import soundfile as sf
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from Conv_TasNet import ConvTasNet
from Conv_TasNet_Semantic import ConvTasNetSemantic
from SI_SNR import si_snr_loss
from semantic_data import SemanticFeatureStore, collate_semantic_records


LOG = logging.getLogger("train_librimix")


def read_scp(path):
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            key, value = line.rstrip().split(maxsplit=1)
            result[key] = value
    return result


class LibriMixFullUtterance(Dataset):
    def __init__(self, data_root, split, semantic_root=None, limit=None):
        split_root = Path(data_root) / split
        mix = read_scp(split_root / "wav_clean.scp")
        source1 = read_scp(split_root / "spk1.scp")
        source2 = read_scp(split_root / "spk2.scp")
        if not (mix.keys() == source1.keys() == source2.keys()):
            raise RuntimeError(f"SCP ID mismatch in {split_root}")
        self.ids = list(mix)
        if limit is not None:
            self.ids = self.ids[:limit]
        self.mix, self.source1, self.source2 = mix, source1, source2
        self.semantic_store = (
            SemanticFeatureStore(semantic_root) if semantic_root else None
        )
        if self.semantic_store:
            missing = [key for key in self.ids if key not in self.semantic_store]
            if missing:
                raise RuntimeError(f"Missing {len(missing)} semantic records; first={missing[0]}")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def load_audio(path):
        audio, sample_rate = sf.read(path, dtype="float32")
        if sample_rate != 16000 or audio.ndim != 1:
            raise ValueError(f"Expected mono 16 kHz audio: {path}")
        return torch.from_numpy(audio)

    def __getitem__(self, index):
        key = self.ids[index]
        item = {
            "id": key,
            "mix": self.load_audio(self.mix[key]),
            "ref": [
                self.load_audio(self.source1[key]),
                self.load_audio(self.source2[key]),
            ],
        }
        if self.semantic_store:
            item["semantic_record"] = self.semantic_store.load(key)
        return item


def collate_one(items):
    if len(items) != 1:
        raise ValueError("This leakage-safe trainer currently requires batch_size=1")
    item = items[0]
    batch = {
        "id": item["id"],
        "mix": item["mix"].unsqueeze(0),
        "ref": [source.unsqueeze(0) for source in item["ref"]],
    }
    if "semantic_record" in item:
        batch.update(collate_semantic_records([item["semantic_record"]]))
    return batch


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            moved[key] = [tensor.to(device) for tensor in value]
        else:
            moved[key] = value
    return moved


def ensure_batched(estimates):
    return [estimate.unsqueeze(0) if estimate.dim() == 1 else estimate for estimate in estimates]


def align_for_sisnr(estimates, batch):
    """Crop sub-stride reconstruction differences before waveform loss.

    Conv1d/ConvTranspose1d without input padding cannot reconstruct the final
    incomplete encoder stride of an arbitrary-length utterance.  The original
    4 s chunks hid this because their lengths were divisible by the stride.
    """
    common_length = min(
        [estimate.shape[-1] for estimate in estimates]
        + [reference.shape[-1] for reference in batch["ref"]]
    )
    estimates = [estimate[..., :common_length] for estimate in estimates]
    batch["ref"] = [reference[..., :common_length] for reference in batch["ref"]]
    return estimates


def forward_model(model, batch, model_kind):
    if model_kind == "baseline":
        estimates = model(batch["mix"])
    else:
        estimates = model(
            batch["mix"],
            batch["semantic"],
            batch["semantic_mask"],
            batch["global_semantic"],
        )
    return align_for_sisnr(ensure_batched(estimates), batch)


def run_epoch(
    model,
    loader,
    device,
    model_kind,
    optimizer=None,
    clip_norm=5.0,
    writer=None,
    epoch=1,
    global_step_offset=0,
    step_callback=None,
    gradient_accumulation_steps=1,
):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            estimates = forward_model(model, batch, model_kind)
            loss = si_snr_loss(estimates, batch)
        if training:
            (loss / gradient_accumulation_steps).backward()
            update_now = (
                step % gradient_accumulation_steps == 0 or step == len(loader)
            )
            if update_now:
                # If the final accumulation group is shorter than requested,
                # undo the extra division so it remains an average of that group.
                remainder = step % gradient_accumulation_steps
                if step == len(loader) and remainder:
                    correction = gradient_accumulation_steps / remainder
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total += loss.item()
        global_step = global_step_offset + step
        if writer is not None:
            split = "train" if training else "dev"
            writer.add_scalar(f"loss/{split}_step", loss.item(), global_step)
        if step == 1 or step % 100 == 0:
            LOG.info("%s step=%d/%d loss=%.4f id=%s", "train" if training else "dev", step, len(loader), loss.item(), batch["id"])
        if training and step_callback is not None and update_now:
            step_callback(step, global_step, total / step)
            model.train(True)
    return total / max(len(loader), 1), global_step_offset + len(loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "semantic"), required=True)
    parser.add_argument("--data-root", default="/home/zt/Desktop/STL/espnet/egs2/librimix/sot_asr1/data")
    parser.add_argument("--semantic-root", default="/home/zt/Desktop/STL/Multi-talker-ASR-with-LLMs/semantic_features/libri2mix_clean_offset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Accumulate gradients over this many utterances before each "
            "optimizer update. The DataLoader batch size remains 1."
        ),
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument(
        "--val-interval-steps",
        type=int,
        default=10000,
        help=(
            "Run validation and update last.pt/best.pt every N training steps. "
            "Set to 0 to validate only at epoch boundaries."
        ),
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "Logical torch device, e.g. cuda or cuda:0. Physical GPU selection "
            "should normally be done with CUDA_VISIBLE_DEVICES."
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
        help="Recompute semantic TCN activations in backward to reduce GPU memory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if (
        args.val_interval_steps > 0
        and args.val_interval_steps % args.gradient_accumulation_steps != 0
    ):
        raise ValueError(
            "--val-interval-steps must be divisible by "
            "--gradient-accumulation-steps so checkpoints are saved only "
            "after complete optimizer updates"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full Conv-TasNet training")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "arguments.json").write_text(json.dumps(vars(args), indent=2))
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    semantic_train = f"{args.semantic_root}/train" if args.model == "semantic" else None
    semantic_dev = f"{args.semantic_root}/dev" if args.model == "semantic" else None
    train_set = LibriMixFullUtterance(args.data_root, "train", semantic_train, args.train_limit)
    dev_set = LibriMixFullUtterance(args.data_root, "dev", semantic_dev, args.dev_limit)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_one)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_one)

    common = dict(N=args.N, L=args.L, B=args.B, H=args.H, P=args.P, X=args.X, R=args.R, num_spks=2)
    model = (
        ConvTasNet(
            **common, gradient_checkpointing=args.gradient_checkpointing
        )
        if args.model == "baseline"
        else ConvTasNetSemantic(
            **common, gradient_checkpointing=args.gradient_checkpointing
        )
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_dev, global_step = 1, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + (1 if checkpoint.get("epoch_complete", True) else 0)
        best_dev = checkpoint.get("best_dev", best_dev)
        # Older epoch-boundary checkpoints predate global_step. Infer the
        # completed sample steps so resumed TensorBoard curves continue rather
        # than restarting at step zero and overlapping the existing run.
        global_step = checkpoint.get(
            "global_step", checkpoint["epoch"] * len(train_loader)
        )

    LOG.info("model=%s params=%.2fM train=%d dev=%d", args.model, sum(p.numel() for p in model.parameters()) / 1e6, len(train_set), len(dev_set))
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            def validate_and_save(train_loss, current_global_step, epoch_complete):
                nonlocal best_dev
                with torch.no_grad():
                    dev_loss, _ = run_epoch(
                        model, dev_loader, device, args.model,
                        writer=None, epoch=epoch,
                    )
                writer.add_scalar("loss/dev_validation", dev_loss, current_global_step)
                writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], current_global_step)
                writer.flush()
                state = {
                    "epoch": epoch,
                    "epoch_complete": epoch_complete,
                    "global_step": current_global_step,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_dev": min(best_dev, dev_loss),
                    "train_loss": train_loss,
                    "dev_loss": dev_loss,
                }
                torch.save(state, output_dir / "last.pt")
                if dev_loss < best_dev:
                    best_dev = dev_loss
                    state["best_dev"] = best_dev
                    torch.save(state, output_dir / "best.pt")
                LOG.info(
                    "validation epoch=%d global_step=%d train_loss=%.4f "
                    "dev_loss=%.4f best=%.4f epoch_complete=%s",
                    epoch, current_global_step, train_loss, dev_loss, best_dev,
                    epoch_complete,
                )
                return dev_loss

            def maybe_validate(_epoch_step, current_global_step, running_train_loss):
                if (
                    args.val_interval_steps > 0
                    and current_global_step % args.val_interval_steps == 0
                ):
                    validate_and_save(
                        running_train_loss, current_global_step, epoch_complete=False
                    )

            train_loss, global_step = run_epoch(
                model, train_loader, device, args.model, optimizer,
                writer=writer, epoch=epoch, global_step_offset=global_step,
                step_callback=maybe_validate,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            # Always mark the epoch boundary as complete. If an interval validation
            # happened on the final step, repeat validation so last.pt records the
            # completed epoch and resume advances to the following epoch.
            dev_loss = validate_and_save(
                train_loss, global_step, epoch_complete=True
            )
            writer.add_scalars(
                "loss/epoch",
                {"train": train_loss, "dev": dev_loss},
                epoch,
            )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
