"""PMQ factor collection using native model forwards and GPTQ candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pmq.bridge import enable_current_project_imports
from pmq.config import ProtocolConfig
from pmq.routing import NativeRoutingCollector, RoutingStatistics


@dataclass(frozen=True)
class PmqFactors:
    """Native routing factors and single-expert candidate reconstruction losses."""

    routing: RoutingStatistics
    candidate_loss: dict[int, dict[int, dict[int, float]]]


def _module_output(output) -> torch.Tensor:
    """Extract the hidden-state tensor from standard MoE module return values."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unsupported MoE output type: {type(output)!r}")


def _load_context(protocol: ProtocolConfig, seed: int):
    enable_current_project_imports()
    from datasets import load_from_disk
    from src.experiment.data_preparation import prepare_calibration_inputs
    from src.models import get_moe_adapter
    from src.models.loading import (
        get_input_device,
        load_model,
        load_model_config,
        load_tokenizer,
    )

    model_cfg = protocol.data["model"]
    calibration_cfg = protocol.data["dataset"]["calibration"]
    sample_cfg = protocol.data["calibration"]
    adapter = get_moe_adapter(protocol.architecture)
    model_config = load_model_config(str(protocol.model_path))
    tokenizer = load_tokenizer(str(protocol.model_path))
    dataset = load_from_disk(str(protocol.calibration_path))
    input_ids, attention_mask = prepare_calibration_inputs(
        tokenizer,
        dataset,
        protocol=str(calibration_cfg["protocol"]),
        split=str(calibration_cfg["split"]),
        num_samples=int(sample_cfg["samples"]),
        seq_len=int(sample_cfg["seq_len"]),
        seed=int(seed),
        add_special_tokens=bool(calibration_cfg.get("add_special_tokens", True)),
    )
    model = load_model(
        str(protocol.model_path),
        device_map="auto",
        attn_implementation="eager",
        torch_dtype="auto",
    )
    return model, model_cfg, model_config, adapter, input_ids, attention_mask, get_input_device(model)


def _collect_layer_io(model, module, input_ids, attention_mask, input_device):
    """Capture one native MoE layer's unmodified inputs and outputs on CPU."""
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []

    def capture(_module, args, output):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("native MoE module did not receive hidden states")
        inputs.append(args[0].detach().to("cpu"))
        outputs.append(_module_output(output).detach().to("cpu"))

    handle = module.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for ids, mask in zip(input_ids, attention_mask):
                model(
                    input_ids=ids.unsqueeze(0).to(input_device, non_blocking=True),
                    attention_mask=mask.unsqueeze(0).to(input_device, non_blocking=True),
                    use_cache=False,
                )
    finally:
        handle.remove()
    if len(inputs) != len(input_ids) or len(outputs) != len(input_ids):
        raise RuntimeError("incomplete native MoE capture")
    return inputs, outputs


@torch.inference_mode()
def _candidate_loss_for_layer(
    *,
    layer: int,
    module,
    adapter,
    cache,
    bits: tuple[int, ...],
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
) -> dict[int, dict[int, float]]:
    """Measure one-expert output perturbation for cached GPTQ candidates.

    The model's normal forward path is used only to obtain reference inputs and
    outputs. Each candidate then replaces one expert temporarily while the
    native MoE module evaluates the same captured inputs.
    """
    device = next(module.parameters()).device
    losses: dict[int, dict[int, float]] = {}
    for expert in sorted(cache.experts_for_layer(layer)):
        original = {
            projection: weight.detach().clone()
            for projection, weight in adapter.expert_weights(module, expert).items()
        }
        bit_losses: dict[int, float] = {}
        try:
            for bit in bits:
                candidate = {
                    projection: cache.get(layer, expert, projection, bit, device)
                    for projection in adapter.weight_types
                }
                adapter.set_expert_weights(module, expert, candidate)
                loss = 0.0
                with torch.inference_mode():
                    for hidden, reference_output in zip(inputs, outputs):
                        actual = _module_output(module(hidden.to(device, non_blocking=True)))
                        delta = reference_output.to(device, dtype=torch.float64) - actual.to(torch.float64)
                        loss += torch.linalg.vector_norm(delta).item()
                bit_losses[int(bit)] = float(loss)
        finally:
            adapter.set_expert_weights(module, expert, original)
        losses[int(expert)] = bit_losses
    return losses


def collect_pmq_factors(
    protocol: ProtocolConfig,
    *,
    candidate_cache_dir: str | Path,
    seed: int,
) -> tuple[PmqFactors, dict]:
    """Collect complete PMQ factors without replacing any native MoE forward."""
    enable_current_project_imports()
    from src.scoring.candidate_weights import CandidateWeightCache

    cache = CandidateWeightCache(cache_dir=Path(candidate_cache_dir))
    model, model_cfg, model_config, adapter, input_ids, attention_mask, input_device = _load_context(protocol, seed)
    modules = adapter.collect_moe_modules(model)
    expected_keys = {
        (int(layer), expert)
        for layer in modules
        for expert in range(adapter.num_experts(model_config))
    }
    if not cache.covers(expected_keys):
        missing = len(expected_keys - cache.expert_keys)
        raise ValueError(f"candidate cache is missing {missing} routed experts")

    collector = NativeRoutingCollector(modules, adapter, model_config)
    try:
        with torch.inference_mode():
            for ids, mask in zip(input_ids, attention_mask):
                model(
                    input_ids=ids.unsqueeze(0).to(input_device, non_blocking=True),
                    attention_mask=mask.unsqueeze(0).to(input_device, non_blocking=True),
                    use_cache=False,
                )
        routing = collector.close()
    except Exception:
        collector.close()
        raise

    candidate_loss: dict[int, dict[int, dict[int, float]]] = {}
    for layer, module in sorted(modules.items()):
        layer_inputs, layer_outputs = _collect_layer_io(
            model,
            module,
            input_ids,
            attention_mask,
            input_device,
        )
        candidate_loss[int(layer)] = _candidate_loss_for_layer(
            layer=int(layer),
            module=module,
            adapter=adapter,
            cache=cache,
            bits=protocol.candidate_bits,
            inputs=layer_inputs,
            outputs=layer_outputs,
        )
        del layer_inputs, layer_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "model": model_cfg["name"],
        "architecture": adapter.name,
        "seed": int(seed),
        "calibration_path": str(protocol.calibration_path),
        "calibration_protocol": protocol.data["dataset"]["calibration"]["protocol"],
        "calibration_samples": int(input_ids.shape[0]),
        "calibration_seq_len": int(input_ids.shape[1]),
        "candidate_cache_dir": str(Path(candidate_cache_dir).resolve()),
        "candidate_bits": list(protocol.candidate_bits),
        "candidate_loss": "single-expert native-MoE output L2 perturbation",
    }
    del model
    return PmqFactors(routing=routing, candidate_loss=candidate_loss), metadata


def save_pmq_factors(output_dir: str | Path, factors: PmqFactors, metadata: dict) -> Path:
    """Persist all PMQ factors under the caller-owned output directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "pmq_factors.pt"
    torch.save(
        {
            "selected_count": factors.routing.selected_count,
            "selected_weight": factors.routing.selected_weight,
            "candidate_loss": factors.candidate_loss,
            "metadata": metadata,
        },
        output,
    )
    return output
