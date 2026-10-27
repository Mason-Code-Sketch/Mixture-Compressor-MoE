"""Native PMQ quantization, checkpoint saving and independent PPL evaluation."""

from __future__ import annotations

import json
import gc
import math
import pickle
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from pmq.adapters import PmqMoeAdapter
from pmq.artifacts import provenance, sha256_file, write_json, validate_provenance
from pmq.config import ProtocolConfig
from pmq.data import evaluation_blocks
from pmq.monitor import LOGGER, progress
from pmq.packed import save_packed_model, load_packed_model
from pmq.quantize import apply_allocation
from pmq.runtime import input_device, load_model, load_tokenizer


def _parse_plan(raw_plan: dict) -> dict[tuple[int, int], int]:
    allocation = {}
    for key, projections in raw_plan.items():
        layer, expert = (int(part) for part in key.split(","))
        bits = {int(bit) for bit in projections.values()}
        if len(bits) != 1:
            raise ValueError(f"PMQ plan must tie all projections: {key}")
        allocation[(layer, expert)] = bits.pop()
    return allocation


def load_allocation(path):
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return _parse_plan(payload["allocation"]), payload
    with path.open("rb") as stream:
        native = pickle.load(stream)
    allocation = {(int(layer), int(expert)): int(bit)
                  for layer, experts in native.items() for expert, bit in experts.items()}
    metadata = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
    if metadata.get("allocation_sha256") and sha256_file(path) != metadata["allocation_sha256"]:
        raise ValueError("allocation PKL checksum mismatch")
    return allocation, metadata


def validate_checkpoint(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "checkpoint_complete.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("PMQ checkpoint is incomplete")
    for name, digest in metadata["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"PMQ checkpoint checksum mismatch: {name}")
    return metadata


@torch.inference_mode()
def _ppl(model, blocks: list[torch.Tensor]) -> float:
    device = input_device(model)
    total_nll, total_tokens = 0.0, 0
    for position, block in enumerate(blocks, 1):
        ids = block.unsqueeze(0).to(device)
        logits = model(input_ids=ids, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1), reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite PMQ perplexity loss")
        total_nll += float(loss)
        total_tokens += ids.numel() - ids.size(0)
        if position == 1 or position % 16 == 0 or position == len(blocks):
            progress("PPL_BLOCK", position, len(blocks))
    if not total_tokens:
        raise ValueError("PMQ evaluation has no predicted tokens")
    return math.exp(total_nll / total_tokens)


def quantize_pmq_plan(protocol, *, allocation_path, seed, output_dir):
    output = Path(output_dir)
    if any((output / name).exists() for name in ("qmodel.pt", "checkpoint_complete.json")):
        raise FileExistsError(f"checkpoint artifacts already exist: {output}")
    allocation, allocation_metadata = load_allocation(allocation_path)
    if any(bit not in protocol.candidate_bits for bit in allocation.values()):
        raise ValueError("allocation contains an unsupported bit")
    if allocation_metadata.get("provenance"):
        validate_provenance(protocol, allocation_metadata["provenance"], seed=seed)
    tokenizer = load_tokenizer(protocol.model_path)
    model = load_model(protocol)
    adapter = PmqMoeAdapter(protocol.architecture)
    expected = {(layer, expert) for layer, module in adapter.collect_moe_modules(model).items()
                for expert in range(adapter.num_experts(module))}
    if expected != set(allocation):
        raise ValueError("allocation does not match the model's complete expert set")
    LOGGER.info("QUANTIZE_START checkpoint=%s expert_count=%s", output, len(allocation))
    start = perf_counter()
    apply_allocation(model, adapter, allocation, protocol=protocol, seed=seed)
    seconds = perf_counter() - start
    save_packed_model(model, tokenizer, output)
    files = ["qmodel.pt", "config.json", "tokenizer_config.json"]
    metadata = {"status": "complete", "method": "pmq", "quantize_seconds": seconds,
                "model_implementation": type(model).__module__ + "." + type(model).__name__,
                "provenance": provenance(protocol, seed=seed),
                "allocation": str(Path(allocation_path).resolve()),
                "allocation_sha256": sha256_file(allocation_path),
                "allocation_metadata": allocation_metadata,
                "files": {name: sha256_file(output / name) for name in files}}
    write_json(output / "checkpoint_complete.json", metadata)
    LOGGER.info("QUANTIZE_COMPLETE seconds=%.3f checkpoint=%s", seconds, output)
    del model
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return output


def evaluate_saved_ppl(protocol, *, checkpoint_dir, seed):
    output = Path(checkpoint_dir)
    if (output / "evaluation.json").exists():
        raise FileExistsError(f"PPL result already exists: {output}")
    metadata = validate_checkpoint(output)
    validate_provenance(protocol, metadata["provenance"], seed=seed)
    tokenizer = load_tokenizer(output)
    model = load_packed_model(output)
    metrics = {}
    for name, settings in protocol.data["dataset"]["evaluations"].items():
        blocks = evaluation_blocks(tokenizer, protocol.evaluation_paths[name], settings)
        start = perf_counter()
        LOGGER.info("PPL_START dataset=%s blocks=%s", name, len(blocks))
        metrics[name] = {"ppl": _ppl(model, blocks), "blocks": len(blocks),
                         "predicted_tokens": sum(block.numel() - 1 for block in blocks),
                         "seconds": perf_counter() - start}
        LOGGER.info("PPL_COMPLETE dataset=%s ppl=%.8f", name, metrics[name]["ppl"])
    result = {"method": "pmq", "status": "complete", "seed": seed, "checkpoint": str(output.resolve()),
              "checkpoint_sha256": metadata["files"]["qmodel.pt"], "loss_dtype": "float32", "ppl": metrics}
    path = output / "evaluation.json"
    write_json(path, result)
    return path


def evaluate_pmq_plan(protocol: ProtocolConfig, *, allocation_path, seed, output_dir) -> Path:
    directory = quantize_pmq_plan(protocol, allocation_path=allocation_path, seed=seed, output_dir=output_dir)
    # Release the original quantization-time model before reloading the saved checkpoint.
    return evaluate_saved_ppl(protocol, checkpoint_dir=directory, seed=seed)
