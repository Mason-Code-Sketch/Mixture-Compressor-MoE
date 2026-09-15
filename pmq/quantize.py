"""PMQ-owned candidate construction and allocation materialization."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from pmq.adapters import PmqMoeAdapter
from utils.normal_quantizer import normal_quantize


def quantized_weights(
    weights: Mapping[str, torch.Tensor], *, bit: int, group_size: int
) -> dict[str, torch.Tensor]:
    """Create PMQ's per-projection normal-quantization candidate tensors."""
    return {
        name: normal_quantize(weight, blocksize=group_size, wbit=int(bit))
        for name, weight in weights.items()
    }


@torch.inference_mode()
def apply_allocation(
    model,
    adapter: PmqMoeAdapter,
    allocation: Mapping[tuple[int, int], int],
    *,
    shared_bit: int,
    group_size: int,
) -> None:
    """Quantize every allocated routed expert and fixed shared expert in place."""
    for layer, module in adapter.collect_moe_modules(model).items():
        for expert in range(adapter.num_experts(module)):
            key = (int(layer), expert)
            if key not in allocation:
                raise KeyError(f"PMQ allocation omits routed expert {key}")
            adapter.set_expert_weights(
                module,
                expert,
                quantized_weights(
                    adapter.expert_weights(module, expert),
                    bit=int(allocation[key]),
                    group_size=group_size,
                ),
            )
        shared = adapter.shared_weights(module)
        if shared:
            adapter.set_shared_weights(
                module,
                quantized_weights(shared, bit=int(shared_bit), group_size=group_size),
            )
