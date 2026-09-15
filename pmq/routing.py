"""Native-router statistics for PMQ importance estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutingStatistics:
    """PMQ's routed-token counts and unnormalised selected-weight sums."""

    selected_count: dict[int, torch.Tensor]
    selected_weight: dict[int, torch.Tensor]


class NativeRoutingCollector:
    """Collect PMQ router factors without replacing the MoE forward path."""

    def __init__(self, moe_modules: dict[int, torch.nn.Module], adapter, config):
        self._adapter = adapter
        self._counts: dict[int, torch.Tensor] = {}
        self._weights: dict[int, torch.Tensor] = {}
        self._handles = []
        experts_per_token = adapter.num_experts_per_tok(config)
        for layer, moe_module in moe_modules.items():
            experts = adapter.num_experts(config)
            self._counts[int(layer)] = torch.zeros(experts, dtype=torch.long)
            self._weights[int(layer)] = torch.zeros(experts, dtype=torch.float64)
            hook_module = adapter.router_hook_module(moe_module)
            self._handles.append(
                hook_module.register_forward_hook(
                    self._hook(int(layer), moe_module, experts_per_token)
                )
            )

    def _hook(self, layer: int, moe_module: torch.nn.Module, experts_per_token: int):
        def record(_module, inputs, output):
            if isinstance(output, torch.Tensor):
                # Mixtral exposes logits directly from its native gate Linear.
                # Reconstruct the same normalized Top-k weights used by its MoE block.
                scores = torch.softmax(output, dtype=torch.float32, dim=-1)
                weights, indices = torch.topk(
                    scores,
                    k=experts_per_token,
                    dim=-1,
                )
                weights = weights / weights.sum(dim=-1, keepdim=True)
                state = indices, weights.to(output.dtype)
            else:
                state = self._adapter.scoring_route_state(moe_module, inputs, output)
            if state is None:
                raise RuntimeError(
                    f"{self._adapter.name} did not expose native routing state "
                    f"for layer {layer}"
                )
            indices, weights = state
            if indices.shape != weights.shape or indices.shape[-1] != experts_per_token:
                raise RuntimeError(
                    f"invalid native router state at layer {layer}: "
                    f"indices={tuple(indices.shape)}, weights={tuple(weights.shape)}"
                )
            flat_indices = indices.detach().reshape(-1).to("cpu", dtype=torch.long)
            flat_weights = weights.detach().reshape(-1).to("cpu", dtype=torch.float64)
            self._counts[layer].scatter_add_(
                0, flat_indices, torch.ones_like(flat_indices, dtype=torch.long)
            )
            self._weights[layer].scatter_add_(0, flat_indices, flat_weights)

        return record

    def close(self) -> RoutingStatistics:
        """Remove hooks and return detached CPU statistics exactly once."""
        while self._handles:
            self._handles.pop().remove()
        return RoutingStatistics(
            selected_count={layer: value.clone() for layer, value in self._counts.items()},
            selected_weight={layer: value.clone() for layer, value in self._weights.items()},
        )


def pmq_significance(
    statistics: RoutingStatistics,
    *,
    alpha: float = 1.0,
    beta: float = 1.5,
) -> dict[int, torch.Tensor]:
    """Compute the original PMQ count/weight importance factor by layer."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("PMQ alpha and beta must be positive")
    return {
        layer: count.to(torch.float64).pow(alpha) * weight.pow(beta)
        for layer, (count, weight) in (
            (layer, (statistics.selected_count[layer], statistics.selected_weight[layer]))
            for layer in statistics.selected_count
        )
    }
