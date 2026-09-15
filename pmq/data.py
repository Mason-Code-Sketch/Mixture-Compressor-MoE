"""Standalone calibration and PPL data preparation for PMQ."""

from __future__ import annotations

import random
from collections.abc import Mapping

import torch
from datasets import load_from_disk


def _tokenize_corpus(tokenizer, dataset, *, add_special_tokens: bool) -> torch.Tensor:
    return tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    ).input_ids[0]


def calibration_inputs(tokenizer, dataset_path, settings: Mapping[str, object], *, samples: int, seq_len: int, seed: int) -> torch.Tensor:
    """Create calibration windows according to a PMQ-owned dataset protocol."""
    dataset = load_from_disk(str(dataset_path))
    split = dataset[str(settings["split"])]
    protocol = str(settings.get("protocol", "corpus_windows"))
    if protocol == "corpus_windows":
        tokens = _tokenize_corpus(
            tokenizer,
            split,
            add_special_tokens=bool(settings.get("add_special_tokens", True)),
        )
        available = tokens.numel() - int(seq_len) + 1
        if samples <= 0 or available < samples:
            raise ValueError("calibration corpus cannot provide the requested unique windows")
        starts = random.Random(seed).sample(range(available), int(samples))
        return torch.stack([tokens[start : start + seq_len] for start in starts])
    if protocol == "random_document_windows":
        generator = random.Random(seed)
        windows = []
        while len(windows) < int(samples):
            text = split[generator.randrange(len(split))]["text"]
            encoded = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=bool(settings.get("add_special_tokens", True)),
            ).input_ids[0]
            if encoded.numel() < int(seq_len):
                continue
            start = generator.randint(0, encoded.numel() - int(seq_len))
            windows.append(encoded[start : start + seq_len])
        return torch.stack(windows)
    raise ValueError(f"unsupported PMQ calibration protocol: {protocol}")


def evaluation_blocks(tokenizer, dataset_path, settings: Mapping[str, object]) -> list[torch.Tensor]:
    """Build deterministic, non-overlapping language-model evaluation blocks."""
    dataset = load_from_disk(str(dataset_path))
    split = dataset[str(settings["split"])]
    protocol = str(settings.get("protocol", "corpus_windows"))
    if protocol == "corpus_windows":
        tokens = _tokenize_corpus(
            tokenizer,
            split,
            add_special_tokens=bool(settings.get("add_special_tokens", True)),
        )
    elif protocol == "gptq_c4_new":
        text_count = int(settings["text_count"])
        if len(split) < text_count:
            raise ValueError("C4 split contains fewer texts than text_count")
        tokens = tokenizer(
            " ".join(split[:text_count]["text"]), return_tensors="pt"
        ).input_ids[0]
    else:
        raise ValueError(f"unsupported PMQ evaluation protocol: {protocol}")
    seq_len = int(settings["seq_len"])
    max_blocks = settings.get("max_blocks")
    complete_tokens = (tokens.numel() // seq_len) * seq_len
    blocks = [
        tokens[start : start + seq_len].clone()
        for start in range(0, complete_tokens, seq_len)
    ]
    return blocks if max_blocks is None else blocks[: int(max_blocks)]
