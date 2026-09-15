"""PMQ candidate construction and standalone final GPTQ materialization."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional

from pmq.adapters import PmqMoeAdapter
from pmq.data import calibration_inputs
from pmq.gptq import TensorGPTQ
from pmq.runtime import input_device, load_tokenizer
from utils.normal_quantizer import normal_quantize


def quantized_weights(
    weights: Mapping[str, torch.Tensor], *, bit: int, group_size: int
) -> dict[str, torch.Tensor]:
    """Create PMQ's per-projection normal-quantization candidate tensors."""
    return {
        name: normal_quantize(weight, blocksize=group_size, wbit=int(bit))
        for name, weight in weights.items()
    }


def _decoder_layers(model) -> list[torch.nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("PMQ GPTQ requires a model.model.layers decoder stack")
    return list(layers)


def _capture_inputs(model, module, inputs, device) -> list[torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture(_module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("PMQ GPTQ module did not receive tensor inputs")
        captured.append(args[0].detach().cpu())

    handle = module.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for ids in inputs:
                model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        handle.remove()
    if len(captured) != len(inputs):
        raise RuntimeError("incomplete PMQ GPTQ calibration capture")
    return captured


def _capture_many_inputs(model, modules, inputs, device) -> dict[torch.nn.Module, list[torch.Tensor]]:
    """Capture every independent standard-linear input in one calibration replay."""
    captured = {module: [] for module in modules}
    handles = []
    for module in modules:
        def capture(_module, args, *, target=module):
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("PMQ GPTQ module did not receive tensor inputs")
            captured[target].append(args[0].detach().cpu())

        handles.append(module.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            for ids in inputs:
                model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    incomplete = [
        module for module, batches in captured.items() if len(batches) != len(inputs)
    ]
    if incomplete:
        raise RuntimeError("incomplete PMQ GPTQ multi-module calibration capture")
    return captured


def _quantize_matrix(
    weight: torch.Tensor,
    input_batches: list[torch.Tensor],
    *,
    bit: int,
    group_size: int,
    percdamp: float,
) -> torch.Tensor:
    if int(bit) >= 16:
        return weight.detach().clone()
    gptq = TensorGPTQ(weight, bits=int(bit))
    for inputs in input_batches:
        gptq.add_batch(inputs.to(weight.device, dtype=weight.dtype))
    return gptq.quantize(group_size=group_size, percdamp=percdamp)


def _standard_linears(layer, moe_module: torch.nn.Module | None) -> list[tuple[str, torch.nn.Linear]]:
    excluded_roots = []
    if moe_module is not None:
        moe_name = next(name for name, child in layer.named_modules() if child is moe_module)
        excluded_roots.append(f"{moe_name}.experts")
        for shared_name in ("shared_experts", "shared_expert"):
            shared = getattr(moe_module, shared_name, None)
            if shared is None:
                continue
            qualified_name = next(
                name for name, child in layer.named_modules() if child is shared
            )
            excluded_roots.append(qualified_name)
    return [
        (name, child)
        for name, child in layer.named_modules()
        if isinstance(child, torch.nn.Linear)
        and not any(name == root or name.startswith(f"{root}.") for root in excluded_roots)
    ]


def _routed_inputs(
    module,
    adapter: PmqMoeAdapter,
    hidden_batches: list[torch.Tensor],
    model_config,
    expert: int,
) -> list[torch.Tensor]:
    routed = []
    device = next(module.parameters()).device
    for hidden_cpu in hidden_batches:
        hidden = hidden_cpu.to(device)
        indices, _weights = adapter.route_state(
            module, adapter.router_module(module)(hidden), model_config
        )
        selected = (indices == int(expert)).any(dim=-1)
        values = hidden[selected]
        if values.numel():
            routed.append(values.unsqueeze(0).detach().cpu())
    return routed


def _expert_down_inputs(
    weights: Mapping[str, torch.Tensor],
    input_batches: list[torch.Tensor],
    *,
    mixtral: bool,
    activation,
) -> list[torch.Tensor]:
    activations = []
    device = next(iter(weights.values())).device
    for values_cpu in input_batches:
        values = values_cpu.to(device)
        if mixtral:
            gate = activation(functional.linear(values, weights["w1"]))
            up = functional.linear(values, weights["w3"])
        else:
            gate = activation(functional.linear(values, weights["gate_proj"]))
            up = functional.linear(values, weights["up_proj"])
        activations.append((gate * up).detach().cpu())
    return activations


def _quantize_expert(
    module,
    adapter: PmqMoeAdapter,
    hidden_batches: list[torch.Tensor],
    model_config,
    *,
    expert: int,
    bit: int,
    group_size: int,
    percdamp: float,
) -> None:
    inputs = _routed_inputs(module, adapter, hidden_batches, model_config, expert)
    mixtral = adapter.architecture == "mixtral"
    if mixtral:
        original = {
            name: value.detach().clone()
            for name, value in adapter.expert_weights(module, expert).items()
        }
        quantized = dict(original)
        quantized["w1"] = _quantize_matrix(
            original["w1"], inputs, bit=bit, group_size=group_size, percdamp=percdamp
        )
        quantized["w3"] = _quantize_matrix(
            original["w3"], inputs, bit=bit, group_size=group_size, percdamp=percdamp
        )
        down_inputs = _expert_down_inputs(
            quantized,
            inputs,
            mixtral=True,
            activation=module.experts[expert].act_fn,
        )
        quantized["w2"] = _quantize_matrix(
            original["w2"], down_inputs, bit=bit, group_size=group_size, percdamp=percdamp
        )
        adapter.set_expert_weights(module, expert, quantized)
        return

    original_gate_up = module.experts.gate_up_proj[expert].detach().clone()
    original_down = module.experts.down_proj[expert].detach().clone()
    quantized_gate_up = _quantize_matrix(
        original_gate_up, inputs, bit=bit, group_size=group_size, percdamp=percdamp
    )
    gate, up = quantized_gate_up.chunk(2, dim=0)
    down_inputs = _expert_down_inputs(
        {"gate_proj": gate, "up_proj": up},
        inputs,
        mixtral=False,
        activation=module.experts.act_fn,
    )
    quantized_down = _quantize_matrix(
        original_down, down_inputs, bit=bit, group_size=group_size, percdamp=percdamp
    )
    adapter.set_expert_weights(
        module,
        expert,
        {"gate_proj": gate, "up_proj": up, "down_proj": quantized_down},
    )


def _quantize_shared_expert(
    module,
    adapter: PmqMoeAdapter,
    hidden_batches: list[torch.Tensor],
    *,
    bit: int,
    group_size: int,
    percdamp: float,
) -> None:
    original = adapter.shared_weights(module)
    if not original:
        return
    quantized = dict(original)
    quantized["gate_proj"] = _quantize_matrix(
        original["gate_proj"], hidden_batches, bit=bit, group_size=group_size, percdamp=percdamp
    )
    quantized["up_proj"] = _quantize_matrix(
        original["up_proj"], hidden_batches, bit=bit, group_size=group_size, percdamp=percdamp
    )
    shared = getattr(module, "shared_experts", None) or getattr(module, "shared_expert", None)
    down_inputs = _expert_down_inputs(
        quantized,
        hidden_batches,
        mixtral=False,
        activation=getattr(shared, "act_fn", functional.silu),
    )
    quantized["down_proj"] = _quantize_matrix(
        original["down_proj"], down_inputs, bit=bit, group_size=group_size, percdamp=percdamp
    )
    adapter.set_shared_weights(module, quantized)


@torch.inference_mode()
def apply_allocation(
    model,
    adapter: PmqMoeAdapter,
    allocation: Mapping[tuple[int, int], int],
    *,
    protocol,
    seed: int,
) -> None:
    """Materialize one PMQ plan using PMQ's layer-sequential GPTQ pipeline."""
    tokenizer = load_tokenizer(protocol.model_path)
    gptq_inputs = calibration_inputs(
        tokenizer,
        protocol.gptq_calibration_path,
        protocol.data["dataset"]["gptq_calibration"],
        samples=int(protocol.data["calibration"]["gptq"]["samples"]),
        seq_len=int(protocol.data["calibration"]["gptq"]["seq_len"]),
        seed=seed,
    )
    quantization = protocol.data["quantization"]
    group_size = int(quantization["group_size"])
    percdamp = float(quantization["percdamp"])
    standard_bit = int(quantization["standard_linear_bit"])
    router_bit = int(quantization["router_bit"])
    shared_bit = int(protocol.data["pmq"]["shared_expert_bit"])
    device = input_device(model)
    modules = adapter.collect_moe_modules(model)
    for layer_index, layer in enumerate(_decoder_layers(model)):
        module = modules.get(layer_index)
        router = adapter.router_module(module) if module is not None else None
        standard_named_linears = _standard_linears(layer, module)
        standard_linears = [linear for _name, linear in standard_named_linears]
        standard_inputs = _capture_many_inputs(model, standard_linears, gptq_inputs, device)
        for name, linear in standard_named_linears:
            bit = router_bit if linear is router or name.endswith("shared_expert_gate") else standard_bit
            if bit >= 16:
                continue
            linear.weight.copy_(
                _quantize_matrix(
                    linear.weight,
                    standard_inputs[linear],
                    bit=bit,
                    group_size=group_size,
                    percdamp=percdamp,
                )
            )
        del standard_inputs
        if module is None:
            continue
        hidden_batches = _capture_inputs(model, module, gptq_inputs, device)
        for expert in range(adapter.num_experts(module)):
            key = (int(layer_index), expert)
            if key not in allocation:
                raise KeyError(f"PMQ allocation omits routed expert {key}")
            _quantize_expert(
                module,
                adapter,
                hidden_batches,
                model.config,
                expert=expert,
                bit=int(allocation[key]),
                group_size=group_size,
                percdamp=percdamp,
            )
        _quantize_shared_expert(
            module,
            adapter,
            hidden_batches,
            bit=shared_bit,
            group_size=group_size,
            percdamp=percdamp,
        )
        del hidden_batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
