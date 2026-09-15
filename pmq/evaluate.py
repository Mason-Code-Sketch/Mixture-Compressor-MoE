"""PPL evaluation for PMQ allocations using the current project's evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from pmq.bridge import current_project_root, enable_current_project_imports
from pmq.config import ProtocolConfig


def _parse_plan(raw_plan: dict) -> dict[tuple[int, int], dict[str, int]]:
    plan: dict[tuple[int, int], dict[str, int]] = {}
    for key, projections in raw_plan.items():
        layer, expert = (int(part) for part in key.split(","))
        plan[(layer, expert)] = {
            str(projection): int(bit) for projection, bit in projections.items()
        }
    return plan


def evaluate_pmq_plan(
    protocol: ProtocolConfig,
    *,
    allocation_path: str | Path,
    candidate_cache_dir: str | Path,
    seed: int,
    eval_config: str | Path | None = None,
    output_dir: str | Path,
) -> Path:
    """Evaluate a PMQ allocation through the same candidate-cache PPL path."""
    enable_current_project_imports()
    project_root = current_project_root()
    os.chdir(project_root)
    from src.experiment.evaluation import evaluate_quantization_plans
    from src.experiment.setup import (
        apply_evaluation_config,
        initialize_experiment,
        load_experiment_data,
        model_load_kwargs_for_phase,
    )

    allocation_payload = json.loads(Path(allocation_path).read_text())
    plan = _parse_plan(allocation_payload["allocation"])
    args = SimpleNamespace(
        config=str(protocol.current_project_config),
        seed=int(seed),
        visible_devices=None,
        pipeline_parallel_size=None,
        max_gpu_memory=None,
    )
    setup = initialize_experiment(
        args,
        method="pmq",
        bits=list(protocol.candidate_bits),
        average_bit=float(allocation_payload["average_bit"]),
    )
    apply_evaluation_config(
        setup.config,
        str(eval_config) if eval_config is not None else None,
    )
    timings: dict[str, float] = {}
    prepared = load_experiment_data(
        setup,
        args,
        timings,
        include_calibration=False,
    )
    start = perf_counter()
    results, strategy_timings, elapsed = evaluate_quantization_plans(
        model_path=setup.model_path,
        model_load_kwargs=model_load_kwargs_for_phase(setup, "evaluation"),
        plans={"PMQ": plan},
        evaluation_blocks=prepared.evaluation_blocks,
        statistics=None,
        adapter=setup.adapter,
        gptq_config=setup.config.get("gptq", {}),
        candidate_cache_dir=Path(candidate_cache_dir),
        num_gpus=setup.num_gpus,
        pipeline_parallel_size=setup.runtime_policy.pipeline_parallel_size,
        attention_plan={layer: 4 for layer in range(prepared.info.num_layers)},
        shared_bit=int(protocol.data["pmq"]["shared_expert_bit"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation.json"
    path.write_text(
        json.dumps(
            {
                "method": "pmq",
                "allocation": str(Path(allocation_path).resolve()),
                "candidate_cache_dir": str(Path(candidate_cache_dir).resolve()),
                "evaluation_config": str(eval_config) if eval_config else None,
                "seed": int(seed),
                "ppl": results,
                "load_data_seconds": timings.get("load_and_prepare_data"),
                "evaluate_seconds": elapsed,
                "wall_seconds": perf_counter() - start,
                "strategy_timings": strategy_timings,
            },
            indent=2,
        )
        + "\n"
    )
    return path
