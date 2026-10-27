"""Model loading and device helpers owned by the PMQ project."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _native_deepseek(model_path) -> bool:
    path = Path(model_path) / "config.json"
    return path.is_file() and json.loads(path.read_text()).get("model_type") == "deepseek_v2"


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_config(model_path):
    # Use the same native implementation for scoring, quantization and reload.
    return AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=not _native_deepseek(model_path)
    )


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


def load_model(protocol):
    settings = protocol.data["model"]
    dtype = getattr(torch, str(settings.get("dtype", "float16")))
    native_deepseek = _native_deepseek(protocol.model_path)
    kwargs = {
        "torch_dtype": dtype,
        "device_map": settings.get("device_map", "auto"),
        "trust_remote_code": not native_deepseek,
    }
    if settings.get("attn_implementation"):
        kwargs["attn_implementation"] = str(settings["attn_implementation"])
    if native_deepseek:
        kwargs["experts_implementation"] = "eager"
    if settings.get("max_memory"):
        kwargs["max_memory"] = {
            int(key) if str(key).isdigit() else key: value
            for key, value in settings["max_memory"].items()
        }
    model = AutoModelForCausalLM.from_pretrained(str(protocol.model_path), **kwargs)
    patched = align_deepseek_softmax_scale(model)
    if patched:
        print(f"[pmq] applied native DeepSeek YaRN logit scaling to {patched} layers", flush=True)
    model.eval()
    return model


def input_device(model) -> torch.device:
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embeddings is not None and embeddings.weight.device.type != "meta":
        return embeddings.weight.device
    device_map = getattr(model, "hf_device_map", {})
    for device in device_map.values():
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        if isinstance(device, str) and device.startswith("cuda"):
            return torch.device(device)
    return next(model.parameters()).device
