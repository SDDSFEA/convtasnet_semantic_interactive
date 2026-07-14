"""Utilities for reading the cached SOT semantic features by utterance ID."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence


class SemanticFeatureStore:
    """Index one split or a directory containing train shard manifests."""

    def __init__(self, root):
        self.root = Path(root)
        manifests = sorted(self.root.glob("shard_*/manifest.jsonl"))
        if not manifests:
            manifests = [self.root / "manifest.jsonl"]
        self.paths = {}
        for manifest in manifests:
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                path = Path(row["path"])
                if not path.is_absolute():
                    # Existing manifests are relative to the ASR repository.
                    candidates = [Path.cwd() / path]
                    candidates.extend(parent / path for parent in manifest.parents)
                    path = next((p for p in candidates if p.exists()), path)
                self.paths[row["id"]] = path

    def __len__(self):
        return len(self.paths)

    def __contains__(self, utterance_id):
        return utterance_id in self.paths

    def load(self, utterance_id):
        record = torch.load(
            self.paths[utterance_id], map_location="cpu", weights_only=False
        )
        return {
            "id": utterance_id,
            "speaker1_hidden": record["speaker1_hidden"].float(),
            "speaker2_hidden": record["speaker2_hidden"].float(),
            "global_pooled": record["global_pooled"].float(),
            "sc_count": int(record.get("sc_count", 1)),
        }


def collate_semantic_records(records):
    """Pad variable-length speaker streams for ``ConvTasNetSemantic``.

    Returns:
        semantic: ``[B,2,Lmax,D]``
        semantic_mask: ``[B,2,Lmax]``
        global_semantic: ``[B,D]``
    """
    streams = []
    lengths = []
    for record in records:
        for key in ("speaker1_hidden", "speaker2_hidden"):
            stream = record[key]
            streams.append(stream)
            lengths.append(stream.shape[0])
    padded = pad_sequence(streams, batch_first=True)
    max_length = padded.shape[1]
    mask = torch.arange(max_length)[None, :] < torch.tensor(lengths)[:, None]
    batch = len(records)
    return {
        "semantic": padded.reshape(batch, 2, max_length, -1),
        "semantic_mask": mask.reshape(batch, 2, max_length),
        "global_semantic": torch.stack([r["global_pooled"] for r in records]),
        "sc_count": torch.tensor([r["sc_count"] for r in records]),
        "ids": [r["id"] for r in records],
    }
