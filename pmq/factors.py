"""Standalone PMQ routing factors and candidate reconstruction losses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pmq.adapters import PmqMoeAdapter
from pmq.config import ProtocolConfig
from pmq.data import calibration_inputs
from pmq.quantize import quantized_weights
from pmq.routing import NativeRoutingCollector, RoutingStatistics
from pmq.runtime import input_device, load_model, load_tokenizer


@dataclass(frozen=True)
class PmqFactors:
    """Native routing factors and single-expert PMQ candidate losses."""

    routing: RoutingStatistics
    candidate_loss: dict[int, dict[int, dict[int, float]]]


def _module_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unsupported MoE output type: {type(output)!r}")


def _module_device(module) -> torch.device:
    return next(module.parameters()).device


def _capture_layer_io(model, module, inputs, device):
    """Capture native MoE inputs and outputs without replacing its forward."""
    captured_inputs: list[torch.Tensor] = []
    captured_outputs: list[torch.Tensor] = []

    def capture(_module, args, output):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("native MoE module did not receive hidden states")
        captured_inputs.append(args[0].detach().cpu())
        captured_outputs.append(_module_output(output).detach().cpu())

    handle = module.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for ids in inputs:
                model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        handle.remove()
    if len(captured_inputs) != len(inputs):
        raise RuntimeError("incomplete PMQ native MoE capture")
    return captured_inputs, captured_outputs


@torch.inference_mode()
def _candidate_loss_for_layer(
    *,
    module,
    adapter: PmqMoeAdapter,
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    bits: tuple[int, ...],
    group_size: int,
) -> dict[int, dict[int, float]]:
    """Measure original PMQ single-expert output perturbations per bit."""
    device = _module_device(module)
    losses: dict[int, dict[int, float]] = {}
    for expert in range(adapter.num_experts(module)):
        original = {
            name: weight.clone()
            for name, weight in adapter.expert_weights(module, expert).items()
        }
        by_bit: dict[int, float] = {}
        try:
            for bit in bits:
                adapter.set_expert_weights(
                    module,
                    expert,
                    quantized_weights(original, bit=int(bit), group_size=group_size),
                )
                loss = 0.0
                for hidden, reference in zip(inputs, outputs):
                    actual = _module_output(module(hidden.to(device)))
                    delta = reference.to(device, dtype=torch.float64) - actual.to(torch.float64)
                    loss += torch.linalg.vector_norm(delta).item()
                by_bit[int(bit)] = float(loss)
        finally:
            adapter.set_expert_weights(module, expert, original)
        losses[expert] = by_bit
    return losses


def collect_pmq_factors(protocol: ProtocolConfig, *, seed: int) -> tuple[PmqFactors, dict]:
    """Collect PMQ factors through the model's unmodified native forwards."""
    tokenizer = load_tokenizer(protocol.model_path)
    inputs = calibration_inputs(
        tokenizer,
        protocol.factor_path,
        protocol.data["dataset"]["factors"],
        samples=int(protocol.data["calibration"]["factors"]["samples"]),
        seq_len=int(protocol.data["calibration"]["factors"]["seq_len"]),
        seed=seed,
    )
    model = load_model(protocol)
    adapter = PmqMoeAdapter(protocol.architecture)
    modules = adapter.collect_moe_modules(model)
    collector = NativeRoutingCollector(modules, adapter, model.config)
    device = input_device(model)
    try:
        with torch.inference_mode():
            for ids in inputs:
                model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
        routing = collector.close()
    except Exception:
        collector.close()
        raise

    candidate_loss: dict[int, dict[int, dict[int, float]]] = {}
    for layer, module in sorted(modules.items()):
        layer_inputs, layer_outputs = _capture_layer_io(model, module, inputs, device)
        candidate_loss[layer] = _candidate_loss_for_layer(
            module=module,
            adapter=adapter,
            inputs=layer_inputs,
            outputs=layer_outputs,
            bits=protocol.candidate_bits,
            group_size=int(protocol.data["quantization"]["group_size"]),
        )
        del layer_inputs, layer_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "model": protocol.data["model"]["name"],
        "architecture": protocol.architecture,
        "seed": int(seed),
        "factor_dataset": str(protocol.factor_path),
        "factor_protocol": protocol.data["dataset"]["factors"]["protocol"],
        "calibration_samples": int(inputs.shape[0]),
        "calibration_seq_len": int(inputs.shape[1]),
        "candidate_bits": list(protocol.candidate_bits),
        "candidate_quantizer": protocol.data["quantization"]["candidate_quantizer"],
        "candidate_loss": "single-expert native-MoE output L2 perturbation",
    }
    return PmqFactors(routing=routing, candidate_loss=candidate_loss), metadata


def save_pmq_factors(output_dir: str | Path, factors: PmqFactors, metadata: dict) -> Path:
    """Persist PMQ factors under the PMQ result directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pmq_factors.pt"
    torch.save(
        {
            "selected_count": factors.routing.selected_count,
            "selected_weight": factors.routing.selected_weight,
            "candidate_loss": factors.candidate_loss,
            "metadata": metadata,
        },
        path,
    )
    return path
