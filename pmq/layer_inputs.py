"""Bounded CPU decoder-input reuse for native Qwen3 sequential GPTQ."""
from __future__ import annotations

import torch


def can_replay_layers(model, inputs, device):
    """CUDA captures own CPU copies; sharing Q/K/V pays for one input buffer."""
    layers = getattr(getattr(model, "model", None), "layers", ())
    return (torch.device(device).type == "cuda" and not model.training
            and getattr(model.config, "model_type", None) == "qwen3_moe"
            and getattr(model.config, "_attn_implementation", None) == "eager"
            and len(inputs) > 0 and len(layers) > 0
            and all(ids.ndim == 1 and ids.shape == inputs[0].shape for ids in inputs)
            and not any(hasattr(m, "_hf_hook") for m in model.modules())
            and all(type(layer).__name__ == "Qwen3MoeDecoderLayer"
                    and all(hasattr(layer.self_attn, name) for name in ("q_proj", "k_proj", "v_proj"))
                    for layer in layers))


class LayerInputs:
    """Keep one CPU input per sample and overwrite it after each quantized layer."""

    def __init__(self, model, inputs, device):
        self.device = device
        self.buffers = []
        self.kwargs = None
        first = model.model.layers[0]

        class Captured(Exception):
            pass

        def capture(_module, args, kwargs):
            if len(args) != 1 or not torch.is_tensor(args[0]):
                raise RuntimeError("Unsupported decoder input interface")
            if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
                raise RuntimeError("Decoder replay requires cache-free calibration")
            self.buffers.append(args[0].detach().to("cpu", copy=True))
            if self.kwargs is None:
                # Native model supplies its original causal mask, RoPE and positions.
                # Equal-length, unpadded samples use the same immutable metadata.
                self.kwargs = dict(kwargs)
            raise Captured

        handle = first.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.inference_mode():
                for ids in inputs:
                    try:
                        model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
                    except Captured:
                        pass
        finally:
            handle.remove()
        if len(self.buffers) != len(inputs):
            raise RuntimeError("Incomplete decoder input capture")

    def forward(self, layer, sample):
        return layer(self.buffers[sample].to(self.device), **self.kwargs)

    @torch.inference_mode()
    def advance(self, layer):
        for sample, buffer in enumerate(self.buffers):
            output = self.forward(layer, sample)
            if not torch.is_tensor(output) or output.shape != buffer.shape:
                raise RuntimeError("Unsupported decoder output interface")
            # No second sample-sized CPU buffer or GPU-resident activation cache.
            buffer.copy_(output.detach())
            del output
