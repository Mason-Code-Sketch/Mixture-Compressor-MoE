"""Executable PMQ protocol runner for current-project MoE model layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pmq.allocation import build_loss_by_layer, solve_pmq_layers, tied_projection_plan
from pmq.collect import collect_native_routing_statistics, save_routing_statistics
from pmq.config import load_protocol_config
from pmq.evaluate import evaluate_pmq_plan
from pmq.factors import collect_pmq_factors, save_pmq_factors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PMQ protocol runner")
    parser.add_argument("--config", required=True, help="PMQ model YAML")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--phase",
        choices=("routing-statistics", "factors", "allocate", "evaluate"),
        required=True,
        help="Run one explicit PMQ phase.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--candidate-cache-dir",
        default=None,
        help="Complete current-project GPTQ candidate cache required by factors/evaluate.",
    )
    parser.add_argument("--factors-path", default=None)
    parser.add_argument("--allocation-path", default=None)
    parser.add_argument("--average-bit", type=float, default=None)
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Optional current-project evaluation YAML, for example WikiText2.",
    )
    return parser.parse_args(argv)


def _output_dir(protocol, explicit: str | None) -> Path:
    return Path(explicit if explicit is not None else protocol.data["pmq"]["output_dir"])


def _factors_path(output_dir: Path, explicit: str | None) -> Path:
    return Path(explicit) if explicit is not None else output_dir / "pmq_factors.pt"


def _allocation_path(output_dir: Path, average_bit: float, explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return output_dir / "allocations" / f"avg{average_bit:g}" / "allocation.json"


def _load_factor_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing PMQ factors: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"selected_count", "selected_weight", "candidate_loss", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"invalid PMQ factors missing={sorted(missing)}")
    return payload


def _allocate(protocol, *, factors_path: Path, output_dir: Path, average_bit: float) -> Path:
    from pmq.bridge import enable_current_project_imports

    enable_current_project_imports()
    from src.models import get_moe_adapter
    from src.models.loading import load_model_config

    payload = _load_factor_payload(factors_path)
    pmq_cfg = protocol.data["pmq"]
    candidate_bits = protocol.candidate_bits
    if not candidate_bits[0] <= average_bit <= candidate_bits[-1]:
        raise ValueError(
            f"average bit must lie in [{candidate_bits[0]}, {candidate_bits[-1]}]"
        )
    loss_by_layer = build_loss_by_layer(
        selected_count=payload["selected_count"],
        selected_weight=payload["selected_weight"],
        candidate_loss=payload["candidate_loss"],
        alpha=float(pmq_cfg.get("alpha", 1.0)),
        beta=float(pmq_cfg.get("beta", 1.5)),
        normalize_factors=bool(pmq_cfg.get("normalize_factors", True)),
        scale_factor=float(pmq_cfg.get("scale_factor", 1000.0)),
    )
    adapter = get_moe_adapter(protocol.architecture)
    model_config = load_model_config(str(protocol.model_path))
    assignments = solve_pmq_layers(
        loss_by_layer,
        average_bit=float(average_bit),
        shared_units=adapter.shared_expert_budget_units(model_config),
        shared_bit=int(pmq_cfg["shared_expert_bit"]),
        candidate_bits=candidate_bits,
        backend=str(pmq_cfg.get("ilp_backend", "highs")),
    )
    path = _allocation_path(output_dir, average_bit, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": "pmq",
                "average_bit": float(average_bit),
                "candidate_bits": list(candidate_bits),
                "shared_expert_bit": int(pmq_cfg["shared_expert_bit"]),
                "allocation": tied_projection_plan(assignments),
                "factor_source": str(factors_path.resolve()),
                "objective": {
                    "alpha": float(pmq_cfg.get("alpha", 1.0)),
                    "beta": float(pmq_cfg.get("beta", 1.5)),
                    "normalize_factors": bool(pmq_cfg.get("normalize_factors", True)),
                    "scale_factor": float(pmq_cfg.get("scale_factor", 1000.0)),
                    "ilp_backend": str(pmq_cfg.get("ilp_backend", "highs")),
                },
            },
            indent=2,
        )
        + "\n"
    )
    return path


def main(argv=None):
    args = parse_args(argv)
    protocol = load_protocol_config(args.config)
    output_dir = _output_dir(protocol, args.out_dir)
    if args.phase == "routing-statistics":
        statistics, metadata = collect_native_routing_statistics(protocol, seed=args.seed)
        output = save_routing_statistics(output_dir, statistics, metadata)
        print(f"[pmq] native routing statistics saved to {output}", flush=True)
        return
    if args.phase == "factors":
        if args.candidate_cache_dir is None:
            raise ValueError("factors requires --candidate-cache-dir")
        factors, metadata = collect_pmq_factors(
            protocol,
            candidate_cache_dir=args.candidate_cache_dir,
            seed=args.seed,
        )
        output = save_pmq_factors(output_dir, factors, metadata)
        print(f"[pmq] complete factors saved to {output}", flush=True)
        return
    factors_path = _factors_path(output_dir, args.factors_path)
    if args.phase == "allocate":
        if args.average_bit is None:
            raise ValueError("allocate requires --average-bit")
        output = _allocate(
            protocol,
            factors_path=factors_path,
            output_dir=output_dir,
            average_bit=args.average_bit,
        )
        print(f"[pmq] allocation saved to {output}", flush=True)
        return
    if args.candidate_cache_dir is None:
        raise ValueError("evaluate requires --candidate-cache-dir")
    if args.allocation_path is None:
        if args.average_bit is None:
            raise ValueError("evaluate requires --allocation-path or --average-bit")
        allocation_path = _allocation_path(output_dir, args.average_bit, None)
    else:
        allocation_path = Path(args.allocation_path)
    output = evaluate_pmq_plan(
        protocol,
        allocation_path=allocation_path,
        candidate_cache_dir=args.candidate_cache_dir,
        seed=args.seed,
        eval_config=args.eval_config,
        output_dir=allocation_path.parent,
    )
    print(f"[pmq] evaluation saved to {output}", flush=True)


if __name__ == "__main__":
    main()
