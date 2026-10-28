"""Original QLinear-style W_q/meta/bias storage for native PMQ models."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from quant.bitpack import BitPack


PACKING = {1: "1bit_u8", 2: "2bit_u8", 3: "3bit_32", 4: "4bit_u8", 8: "8bit_u8"}
PACK_ROWS = {1: 8, 2: 4, 3: 10, 4: 2, 8: 1}


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def pack_matrix(codes, scale, zero, *, bits, group_size, dtype):
    """Pack solved GPTQ integers without solving a second quantization grid."""
    if bits not in PACKING or group_size <= 0:
        raise ValueError("unsupported PMQ packing parameters")
    rows, columns = codes.shape
    groups = (columns + group_size - 1) // group_size
    if tuple(scale.shape) != (rows, groups) or zero.shape != scale.shape:
        raise ValueError("GPTQ grid shape does not match matrix groups")
    padded = F.pad(codes, (0, groups * group_size - columns)).reshape(-1, group_size)
    count = padded.shape[0]
    padded = F.pad(padded, (0, 0, 0, (-count) % PACK_ROWS[bits]))
    packing = PACKING[bits]
    packed = getattr(BitPack, "pack_" + packing)(padded)
    return {
        "W_q": packed.detach().cpu(),
        "meta": {"nbits": bits, "group_size": group_size, "shape": (rows, columns),
                 "scale": scale.reshape(-1, 1).float().cpu(),
                 "zero": zero.reshape(-1, 1).float().cpu(),
                 "axis": 1, "packing": packing, "compute_dtype": dtype,
                 "view_as_float": False, "quant_scale": False, "quant_zero": False,
                 "pmq_groups": count, "pmq_format_version": 1},
        "bias": None,
    }


class PackedLinear(nn.Module):
    """Dequantize the original bit layout at the requested compute dtype."""

    def __init__(self, state, *, device="cpu", dtype=None):
        super().__init__()
        self._meta = dict(state["meta"])
        self.compute_dtype = dtype or self._meta["compute_dtype"]
        for key in ("scale", "zero"):
            self.register_buffer(key, self._meta.pop(key).to(device=device, dtype=torch.float32))
        self.register_buffer("W_q", state["W_q"].to(device=device))
        self.register_buffer("bias", None if state.get("bias") is None else state["bias"].to(device=device, dtype=self.compute_dtype))
        self.out_features, self.in_features = self.meta["shape"]

    @property
    def meta(self):
        return dict(self._meta, scale=self.scale, zero=self.zero, compute_dtype=self.compute_dtype)

    def _apply(self, fn, recurse=True):
        probe = fn(torch.empty(0, dtype=self.compute_dtype))
        grids = {key: self._buffers.pop(key) for key in ("scale", "zero")}
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            for key, value in grids.items():
                self._buffers[key] = value.to(device=probe.device)
        self.compute_dtype = probe.dtype
        return self

    def dequantize(self):
        meta = self.meta
        values = getattr(BitPack, "unpack_" + meta["packing"])(self.W_q, dtype=torch.float32)
        groups = meta["pmq_groups"]
        values = (values[:groups] - meta["zero"]) * meta["scale"]
        rows, columns = meta["shape"]
        return values.reshape(rows, -1)[:, :columns].to(self.compute_dtype)

    def forward(self, inputs):
        return F.linear(inputs, self.dequantize(), self.bias)

    def state_dict(self, *args, **kwargs):
        return _cpu({"W_q": self.W_q, "meta": self.meta, "bias": self.bias})


class PackedExperts(nn.Module):
    """Fused-HF expert call contract, retaining a fused gate/up matrix per expert."""

    def __init__(self, states, activation, *, device, dtype):
        super().__init__()
        self.act_fn = activation
        self.experts = nn.ModuleList()
        for gate_up, down in states:
            expert = nn.Module()
            expert.gate_up_proj = PackedLinear(gate_up, device=device, dtype=dtype)
            expert.down_proj = PackedLinear(down, device=device, dtype=dtype)
            self.experts.append(expert)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        for index in torch.unique(top_k_index).tolist():
            rank, token = torch.where(top_k_index.T == index)
            expert = self.experts[index]
            gate, up = expert.gate_up_proj(hidden_states[token]).chunk(2, dim=-1)
            values = expert.down_proj(self.act_fn(gate) * up)
            values = values * top_k_weights[token, rank, None]
            output.index_add_(0, token, values.to(output.dtype))
        return output


def _replace(model, name, module):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, module)


def install_packed_model(model, adapter, states):
    """Replace quantized tensors after all calibration passes have finished."""
    fused = {}
    named = dict(model.named_modules())
    for layer, moe in adapter.collect_moe_modules(model).items():
        if not hasattr(moe.experts, "gate_up_proj"):
            continue
        name = next(key for key, value in named.items() if value is moe.experts)
        entries = []
        for expert in range(adapter.num_experts(moe)):
            entries.append((states.pop(f"{name}.{expert}.gate_up_proj"),
                            states.pop(f"{name}.{expert}.down_proj")))
        device, dtype = moe.experts.gate_up_proj.device, moe.experts.gate_up_proj.dtype
        fused[name] = {"experts": len(entries), "dtype": str(dtype).removeprefix("torch.")}
        moe.experts = PackedExperts(entries, moe.experts.act_fn, device=device, dtype=dtype)
    for name, state in states.items():
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"packed projection is not a Linear: {name}")
        state["bias"] = None if module.bias is None else module.bias.detach().cpu()
        _replace(model, name, PackedLinear(state, device=module.weight.device, dtype=module.weight.dtype))
    model.config.pmq_quantization = {
        "format_version": 1, "fused_experts": fused,
        "trust_remote_code": not type(model).__module__.startswith("transformers.models."),
        "attn_implementation": getattr(model.config, "_attn_implementation", "eager"),
        "experts_implementation": getattr(model.config, "_experts_implementation", None),
    }
    return model


def save_packed_model(model, tokenizer, directory):
    """Preserve the upstream qmodel.pt/config.json/tokenizer file convention."""
    from utils.pack import save_quantized

    directory = Path(directory)
    if (directory / "qmodel.pt").exists():
        raise FileExistsError(f"packed checkpoint already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    save_quantized(model, str(directory))
    tokenizer.save_pretrained(str(directory))


def load_packed_model(directory, *, device_map="auto", dtype=None, attn_implementation=None):
    """Rebuild the native skeleton and load the upstream per-module state mapping."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM
    from pmq.runtime import load_model_config, align_deepseek_softmax_scale

    directory = Path(directory)
    config = load_model_config(directory)
    metadata = getattr(config, "pmq_quantization", None)
    if not metadata or metadata.get("format_version") != 1:
        raise ValueError("checkpoint is not a PMQ native packed model")
    kwargs = {"trust_remote_code": metadata.get("trust_remote_code", False),
              "attn_implementation": (attn_implementation if attn_implementation is not None
                                      else metadata["attn_implementation"])}
    if metadata.get("experts_implementation"):
        kwargs["experts_implementation"] = metadata["experts_implementation"]
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, **kwargs)
    # Keep tensors in CPU RAM until leaf modules have been reconstructed.
    weights = torch.load(directory / "qmodel.pt", map_location="cpu", weights_only=False)
    for name, details in metadata["fused_experts"].items():
        original = model.get_submodule(name)
        entries = [
            (weights[f"{name}.experts.{index}.gate_up_proj"],
             weights[f"{name}.experts.{index}.down_proj"])
            for index in range(details["experts"])
        ]
        _replace(model, name, PackedExperts(entries, original.act_fn, device="cpu",
                 dtype=dtype or getattr(torch, details["dtype"])))
    for name, state in weights.items():
        module = model.get_submodule(name)
        if "W_q" in state:
            _replace(model, name, PackedLinear(state, dtype=dtype))
        else:
            for key, value in state.items():
                target = value.to(dtype=dtype) if dtype and value.is_floating_point() else value
                if key in module._parameters:
                    module._parameters[key] = nn.Parameter(target, requires_grad=False)
                elif key in module._buffers:
                    module._buffers[key] = target
                else:
                    raise ValueError(f"unknown checkpoint tensor {name}.{key}")
    if any(parameter.device.type == "meta" for parameter in model.parameters()):
        raise ValueError("packed checkpoint did not populate every model parameter")
    align_deepseek_softmax_scale(model)
    if device_map == "auto":
        # A layer map works for packed buffers as well as floating-point parameters.
        if torch.cuda.is_available():
            layers = model.model.layers
            count = torch.cuda.device_count()
            mapping = {"model.embed_tokens": 0, "model.norm": count - 1, "lm_head": count - 1}
            mapping.update({f"model.layers.{index}": min(count - 1, index * count // len(layers))
                            for index in range(len(layers))})
            from accelerate import dispatch_model
            model = dispatch_model(model, device_map=mapping, force_hooks=True)
        else:
            model = model.to("cpu")
    elif isinstance(device_map, (str, torch.device)):
        model = model.to(device_map)
    else:
        from accelerate import dispatch_model
        model = dispatch_model(model, device_map=device_map, force_hooks=True)
    model.eval()
    return model
