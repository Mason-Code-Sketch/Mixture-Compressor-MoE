"""Bounded CPU decoder-input reuse for native MoE sequential GPTQ."""
from __future__ import annotations

import torch


_DECODER_TYPES = {
    "deepseek_v2": "DeepseekV2DecoderLayer",
    "qwen2_moe": "Qwen2MoeDecoderLayer",
    "qwen3_moe": "Qwen3MoeDecoderLayer",
    "mixtral": "MixtralDecoderLayer",
}


def can_capture_native_layers(model):
    """Restrict early capture and replay to inspected, cache-free decoder layouts."""
    layers = getattr(getattr(model, "model", None), "layers", ())
    expected = _DECODER_TYPES.get(getattr(model.config, "model_type", None))
    return (not model.training and expected is not None and len(layers) > 0
            and not any(hasattr(m, "_hf_hook") for m in model.modules())
            and all(type(layer).__name__ == expected
                    and all(hasattr(layer, name) for name in
                            ("self_attn", "input_layernorm", "post_attention_layernorm", "mlp"))
                    for layer in layers))


def can_replay_layers(model, inputs, device):
    """CUDA captures own CPU copies; sharing Q/K/V pays for one input buffer."""
    # A single metadata object is valid only when every layer uses the same mask.
    layer_types = getattr(model.config, "layer_types", None)
    return (torch.device(device).type == "cuda" and can_capture_native_layers(model)
            and getattr(model.config, "_attn_implementation", None) == "eager"
            and len(inputs) > 0
            and all(ids.ndim == 1 and ids.shape == inputs[0].shape for ids in inputs)
            and (not layer_types or len(set(layer_types)) == 1))


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
            # Checkpoint-owned older decoders return (hidden_states, ...).
            if isinstance(output, (tuple, list)) and output:
                output = output[0]
            if (not torch.is_tensor(output) or output.shape != buffer.shape
                    or output.dtype != buffer.dtype):
                raise RuntimeError("Unsupported decoder output interface")
            # No second sample-sized CPU buffer or GPU-resident activation cache.
            buffer.copy_(output.detach())
            del output
