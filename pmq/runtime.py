"""Model loading and device helpers owned by the PMQ project."""

from __future__ import annotations

import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


def load_model_config(model_path):
    return AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)


def load_model(protocol):
    settings = protocol.data["model"]
    dtype_name = str(settings.get("dtype", "float16"))
    dtype = getattr(torch, dtype_name)
    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    attn_implementation = settings.get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = str(attn_implementation)
    return AutoModelForCausalLM.from_pretrained(str(protocol.model_path), **kwargs)


def input_device(model) -> torch.device:
    device_map = getattr(model, "hf_device_map", {})
    for device in device_map.values():
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        if isinstance(device, str) and device.startswith("cuda"):
            return torch.device(device)
    return next(model.parameters()).device


def align_deepseek_softmax_scale(model) -> int:
    """Apply the original YaRN logit scale only when the native path omits it."""
    config = model.config
    if getattr(config, "model_type", None) != "deepseek_v2":
        return 0
    rope = getattr(config, "rope_scaling", None) or {}
    factor, mscale = float(rope.get("factor", 1)), float(rope.get("mscale_all_dim", 0))
    if factor <= 1 or not mscale:
        return 0
    correction = (1 + 0.1 * mscale * math.log(factor)) ** 2
    patched = 0
    for module in model.modules():
        if hasattr(module, "qk_head_dim") and hasattr(module, "scaling"):
            baseline = module.qk_head_dim ** -0.5
            if math.isclose(float(module.scaling), baseline, rel_tol=1e-6):
                module.scaling *= correction
                patched += 1
    return patched
