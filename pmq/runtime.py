"""Model loading and device helpers owned by the PMQ project."""

from __future__ import annotations

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
