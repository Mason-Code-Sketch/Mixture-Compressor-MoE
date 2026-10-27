"""PMQ per-layer allocation and conversion to the shared plan format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from time import perf_counter

from pmq.monitor import LOGGER, progress

import torch


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def tied_projection_plan(bits_by_expert: Mapping[tuple[int, int], int]) -> dict:
    """Expand PMQ's one-bit-per-expert assignment into the shared plan schema."""
    return {
        f"{layer},{expert}": {projection: int(bit) for projection in PROJECTIONS}
        for (layer, expert), bit in sorted(bits_by_expert.items())
    }


def validate_layer_budget(
    *,
    num_routed_experts: int,
    average_bit: float,
    shared_units: int,
    shared_bit: int,
) -> int:
    """Return PMQ's routed-expert integer bit target for one MoE layer.

    Shared units are fixed at the current protocol's high bit and consume
    their parameter-equivalent part of the layer budget before PMQ solves the
    routed-expert ILP.
    """
    total_units = int(num_routed_experts) + int(shared_units)
    total_budget = total_units * float(average_bit)
    rounded_budget = round(total_budget)
    if abs(total_budget - rounded_budget) > 1e-8:
        raise ValueError(
            "average bit is not representable for this layer's effective "
            f"expert count: units={total_units}, average_bit={average_bit}"
        )
    routed_budget = rounded_budget - int(shared_units) * int(shared_bit)
    if routed_budget < num_routed_experts:
        raise ValueError(
            "PMQ routed budget is below the one-bit minimum after fixed "
            f"shared-expert cost: {routed_budget}"
        )
    if routed_budget > 3 * num_routed_experts:
        raise ValueError(
            "PMQ routed budget exceeds the three-bit maximum after fixed "
            f"shared-expert cost: {routed_budget}"
        )
    return routed_budget


def build_loss_by_layer(
    *,
    selected_count: Mapping[int, torch.Tensor],
    selected_weight: Mapping[int, torch.Tensor],
    candidate_loss: Mapping[int, Mapping[int, Mapping[int, float]]],
    alpha: float = 1.0,
    beta: float = 1.5,
    normalize_factors: bool = True,
    scale_factor: float = 1000.0,
) -> dict[int, dict[int, dict[int, float]]]:
    """Build PMQ's original significance-weighted candidate-loss objective.

    The original solver normalizes activation count and selected routing-weight
    sums independently inside each MoE layer, then multiplies their powers by
    the single-expert reconstruction loss for every candidate bit.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("PMQ alpha and beta must be positive")
    if scale_factor <= 0:
        raise ValueError("PMQ scale_factor must be positive")
    losses: dict[int, dict[int, dict[int, float]]] = {}
    for layer, expert_losses in candidate_loss.items():
        counts = selected_count[int(layer)].to(torch.float64)
        weights = selected_weight[int(layer)].to(torch.float64)
        if normalize_factors:
            count_total = counts.sum()
            weight_total = weights.sum()
            counts = counts / count_total if count_total > 0 else torch.zeros_like(counts)
            weights = weights / weight_total if weight_total > 0 else torch.zeros_like(weights)
        layer_losses: dict[int, dict[int, float]] = {}
        for expert, bit_losses in expert_losses.items():
            significance = float(counts[int(expert)].pow(alpha) * weights[int(expert)].pow(beta))
            layer_losses[int(expert)] = {
                int(bit): significance * float(loss) ** alpha * scale_factor
                for bit, loss in bit_losses.items()
            }
        losses[int(layer)] = layer_losses
    return losses


def solve_pmq_layers(
    loss_by_layer: Mapping[int, Mapping[int, Mapping[int, float]]],
    *,
    average_bit: float,
    shared_units: int,
    shared_bit: int,
    candidate_bits: Sequence[int] = (1, 2, 3),
    backend: str = "highs",
    budget_policy: str = "strict",
) -> dict[tuple[int, int], int]:
    """Solve PMQ's independent layer ILPs and flatten assignments by expert."""
    from pmq.solver import solve_layer_ilp

    if budget_policy not in {"strict", "floor", "balanced_exact"}:
        raise ValueError(f"unknown PMQ budget policy: {budget_policy}")
    start = perf_counter()
    solutions = {}
    upgrades = []
    totals = {layer: (len(losses) + shared_units) * average_bit
              for layer, losses in sorted(loss_by_layer.items())}
    target = round(sum(totals.values()))
    if budget_policy == "balanced_exact" and not math.isclose(sum(totals.values()), target, abs_tol=1e-8):
        raise ValueError("the requested average bit is not representable across all MoE layers")
    def solve(layer, total_budget, exact):
        losses = loss_by_layer[layer]
        routed = int(total_budget) - shared_units * shared_bit
        result = solve_layer_ilp(losses, bit_budget=routed, candidate_bits=candidate_bits,
                                 backend=backend, exact_budget=exact)
        if exact and sum(result.values()) != routed:
            raise RuntimeError("solver did not fill the exact PMQ layer budget")
        return result, sum(float(losses[expert][bit]) for expert, bit in result.items())
    lower_total = 0
    for position, (layer, total) in enumerate(totals.items(), 1):
        if budget_policy == "strict":
            routed = validate_layer_budget(num_routed_experts=len(loss_by_layer[layer]),
                      average_bit=average_bit, shared_units=shared_units, shared_bit=shared_bit)
            result = solve_layer_ilp(loss_by_layer[layer], bit_budget=routed,
                                     candidate_bits=candidate_bits, backend=backend)
            solutions[layer] = result
        else:
            lower = math.floor(total + 1e-8)
            lower_total += lower
            result, cost = solve(layer, lower, budget_policy == "balanced_exact")
            solutions[layer] = result
            if budget_policy == "balanced_exact" and not math.isclose(total, lower, abs_tol=1e-8):
                high, high_cost = solve(layer, lower + 1, True)
                upgrades.append((cost - high_cost, layer, high))
        progress("ALLOCATE_LAYER", position, len(totals), layer=layer, policy=budget_policy)
    if budget_policy == "balanced_exact":
        extra = target - lower_total
        if not 0 <= extra <= len(upgrades):
            raise ValueError("invalid balanced PMQ layer budget")
        # Equal gains have deterministic layer-order tie breaking.
        for _, layer, result in sorted(upgrades, key=lambda item: (-item[0], item[1]))[:extra]:
            solutions[layer] = result
    assignments = {(int(layer), int(expert)): int(bit)
                   for layer, solved in solutions.items() for expert, bit in solved.items()}
    actual = sum(assignments.values()) + len(solutions) * shared_units * shared_bit
    if budget_policy == "balanced_exact" and actual != target:
        raise RuntimeError("PMQ actual total budget does not equal the requested budget")
    LOGGER.info("ALLOCATE_COMPLETE policy=%s seconds=%.4f total_budget=%s", budget_policy, perf_counter() - start, actual)
    return assignments


def describe_budget(assignments, *, shared_units, shared_bit, requested_average_bit, budget_policy):
    layers = sorted({layer for layer, expert in assignments})
    counts = {layer: sum(key[0] == layer for key in assignments) for layer in layers}
    units = sum(counts.values()) + len(layers) * shared_units
    per_layer = {str(layer): sum(bit for (index, _), bit in assignments.items() if index == layer)
                 + shared_units * shared_bit for layer in layers}
    total = sum(per_layer.values())
    return {"policy": budget_policy, "requested_average_bit": requested_average_bit,
            "actual_average_bit": total / units, "actual_total_budget": total,
            "parameter_equivalent_units": units, "layer_total_budgets": per_layer,
            "shared_units_per_layer": shared_units, "shared_expert_bit": shared_bit}
