"""Standalone PMQ allocation materialization and perplexity evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

import torch

from pmq.adapters import PmqMoeAdapter
from pmq.config import ProtocolConfig
from pmq.data import evaluation_blocks
from pmq.quantize import apply_allocation
from pmq.runtime import input_device, load_model, load_tokenizer


def _parse_plan(raw_plan: dict) -> dict[tuple[int, int], int]:
    allocation: dict[tuple[int, int], int] = {}
    for key, projections in raw_plan.items():
        layer, expert = (int(part) for part in key.split(","))
        bits = {int(bit) for bit in projections.values()}
        if len(bits) != 1:
            raise ValueError(f"PMQ plan must tie all projections: {key}")
        allocation[(layer, expert)] = bits.pop()
    return allocation


@torch.inference_mode()
def _ppl(model, blocks: list[torch.Tensor]) -> float:
    device = input_device(model)
    total_nll = 0.0
    total_tokens = 0
    for block in blocks:
        input_ids = block.unsqueeze(0).to(device)
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        loss = float(output.loss)
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite PMQ perplexity loss")
        token_count = input_ids.size(1) - 1
        total_nll += loss * token_count
        total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("PMQ evaluation has no predicted tokens")
    return math.exp(total_nll / total_tokens)


def evaluate_pmq_plan(
    protocol: ProtocolConfig,
    *,
    allocation_path: str | Path,
    seed: int,
    output_dir: str | Path,
) -> Path:
    """Apply one PMQ allocation and compute every configured PPL metric."""
    payload = json.loads(Path(allocation_path).read_text())
    allocation = _parse_plan(payload["allocation"])
    tokenizer = load_tokenizer(protocol.model_path)
    model = load_model(protocol)
    adapter = PmqMoeAdapter(protocol.architecture)
    start = perf_counter()
    apply_allocation(
        model,
        adapter,
        allocation,
        protocol=protocol,
        seed=seed,
    )
    quantize_seconds = perf_counter() - start
    metrics = {}
    for name, settings in protocol.data["dataset"]["evaluations"].items():
        blocks = evaluation_blocks(tokenizer, protocol.evaluation_paths[name], settings)
        evaluation_start = perf_counter()
        metrics[name] = {
            "ppl": _ppl(model, blocks),
            "blocks": len(blocks),
            "seconds": perf_counter() - evaluation_start,
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation.json"
    path.write_text(
        json.dumps(
            {
                "method": "pmq",
                "allocation": str(Path(allocation_path).resolve()),
                "seed": int(seed),
                "quantize_seconds": quantize_seconds,
                "ppl": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    return path
