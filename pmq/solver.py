"""General PMQ per-layer ILP for arbitrary routed-expert counts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def solve_layer_ilp(
    loss_by_expert: Mapping[int, Mapping[int, float]],
    *,
    bit_budget: int,
    candidate_bits: Sequence[int] = (1, 2, 3),
    require_mid_and_high: bool = True,
    backend: str = "highs",
) -> dict[int, int]:
    """Solve PMQ's original one-bit-per-expert layer objective.

    The original PMQ LP constrains the sum of assigned expert bits with
    ``<= bit_budget`` and requires at least one middle and high precision
    expert.  This function retains those semantics while removing the fixed
    Mixtral eight-expert assumption.
    """
    bits = tuple(int(bit) for bit in candidate_bits)
    if bits != tuple(sorted(set(bits))) or bits[0] < 1:
        raise ValueError(f"candidate_bits must be sorted distinct positives, got {bits}")
    experts = tuple(sorted(int(expert) for expert in loss_by_expert))
    if not experts:
        return {}
    if any(set(loss_by_expert[expert]) != set(bits) for expert in experts):
        raise ValueError("every PMQ expert must provide a loss for every candidate bit")
    if bit_budget < len(experts) * bits[0]:
        raise ValueError("PMQ bit budget is below the minimum candidate cost")

    if backend == "gurobi":
        return _solve_with_gurobi(
            loss_by_expert, experts, bits, bit_budget, require_mid_and_high
        )
    if backend == "highs":
        return _solve_with_highs(
            loss_by_expert, experts, bits, bit_budget, require_mid_and_high
        )
    raise ValueError("PMQ backend must be 'gurobi' or 'highs'")


def _solve_with_gurobi(loss_by_expert, experts, bits, bit_budget, require_mid_and_high):
    try:
        import gurobipy as gp
    except ImportError as error:
        raise RuntimeError("PMQ Gurobi backend requires gurobipy") from error
    model = gp.Model("pmq_layer_allocation")
    model.Params.OutputFlag = 0
    choices = model.addVars(experts, bits, vtype=gp.GRB.BINARY, name="bit")
    model.setObjective(
        gp.quicksum(
            float(loss_by_expert[expert][bit]) * choices[expert, bit]
            for expert in experts for bit in bits
        ),
        gp.GRB.MINIMIZE,
    )
    model.addConstr(
        gp.quicksum(bit * choices[expert, bit] for expert in experts for bit in bits)
        <= int(bit_budget),
    )
    for expert in experts:
        model.addConstr(gp.quicksum(choices[expert, bit] for bit in bits) == 1)
    if require_mid_and_high and len(bits) >= 3:
        model.addConstr(gp.quicksum(choices[expert, bits[1]] for expert in experts) >= 1)
        model.addConstr(gp.quicksum(choices[expert, bits[-1]] for expert in experts) >= 1)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"PMQ layer ILP did not solve optimally: status={model.Status}")
    return {
        expert: next(bit for bit in bits if choices[expert, bit].X > 0.5)
        for expert in experts
    }


def _solve_with_highs(loss_by_expert, experts, bits, bit_budget, require_mid_and_high):
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as error:
        raise RuntimeError("PMQ HiGHS backend requires scipy.optimize.milp") from error
    variables = [(expert, bit) for expert in experts for bit in bits]
    index = {key: position for position, key in enumerate(variables)}
    objective = np.asarray(
        [float(loss_by_expert[expert][bit]) for expert, bit in variables],
        dtype=np.float64,
    )
    rows, lower, upper = [], [], []
    budget = np.asarray([bit for _expert, bit in variables], dtype=np.float64)
    rows.append(budget)
    lower.append(-np.inf)
    upper.append(float(bit_budget))
    for expert in experts:
        row = np.zeros(len(variables), dtype=np.float64)
        for bit in bits:
            row[index[(expert, bit)]] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    if require_mid_and_high and len(bits) >= 3:
        for bit in (bits[1], bits[-1]):
            row = np.zeros(len(variables), dtype=np.float64)
            for expert in experts:
                row[index[(expert, bit)]] = 1.0
            rows.append(row)
            lower.append(1.0)
            upper.append(np.inf)
    solution = milp(
        c=objective,
        integrality=np.ones(len(variables), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            np.stack(rows), np.asarray(lower), np.asarray(upper)
        ),
        options={"disp": False},
    )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"PMQ HiGHS ILP did not solve optimally: {solution.message}")
    return {
        expert: max(bits, key=lambda bit: solution.x[index[(expert, bit)]])
        for expert in experts
    }
