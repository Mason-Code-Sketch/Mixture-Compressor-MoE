"""Standalone native PMQ phases with original artifact formats."""

from __future__ import annotations
from pmq.batch_profiles import resolve_cli_batch_cap

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import torch

from pmq.adapters import PmqMoeAdapter
from pmq.allocation import (
    allocation_requires_mid_and_high, build_loss_by_layer,
    solve_pmq_layers, tied_projection_plan,
)
from pmq.artifacts import provenance, sha256_file, write_json, write_pickle, validate_provenance
from pmq.config import load_protocol_config
from pmq.evaluate import evaluate_pmq_plan, quantize_pmq_plan, evaluate_saved_ppl, validate_checkpoint
from pmq.factors import collect_pmq_factors, save_pmq_factors
from pmq.monitor import RunMonitor, LOGGER, component
from pmq.runtime import load_model_config, load_tokenizer


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def batch_size(value):
    return "auto" if value == "auto" else positive_int(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Standalone PMQ protocol runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", choices=("factors", "allocate", "quantize", "ppl", "zeroshot", "evaluate", "mmlu_5shot", "gsm8k_5shot"), required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--factors-path", default=None, help="Directory containing the original three PKLs, or a legacy factors PT")
    parser.add_argument("--allocation-path", default=None)
    parser.add_argument("--average-bit", type=float, default=None)
    parser.add_argument("--saving-path", "--saving_path", default=None, help="Original PMQ packed-model directory")
    parser.add_argument("--log-dir", type=Path, help="Supplemental task batch log directory")
    parser.add_argument("--batch-size", type=batch_size, default=None,
                        help="Evaluation batch size (default: auto)")
    parser.add_argument("--max-batch-size", type=positive_int, default=None,
                        help="Auto batch cap by model (seven/MMLU5shot/GSM5shot): DeepSeek-V2=32/24/20; Qwen1.5-MoE=32/24/56; Mixtral=34/10/13; Qwen3-MoE=51/59/84; others=51/17/96")
    parser.add_argument("--limit", type=int, help="Supplemental smoke only; no formal result written")
    parser.add_argument("--monitor-interval", type=float, default=5)
    args = parser.parse_args(argv)
    args._max_batch_size_explicit = args.max_batch_size is not None
    if (args.batch_size is not None or args.max_batch_size is not None) and args.phase not in ("zeroshot", "mmlu_5shot", "gsm8k_5shot"):
        parser.error("--batch-size and --max-batch-size require a zeroshot or supplemental phase")
    if args.batch_size is None:
        args.batch_size = "auto"
    if args.max_batch_size is None:
        args.max_batch_size = {"mmlu_5shot": 17, "gsm8k_5shot": 96}.get(args.phase, 51)
    if args.limit is not None and (args.limit < 1 or args.phase not in ("mmlu_5shot", "gsm8k_5shot")):
        parser.error("--limit requires a supplemental phase and a positive count")
    if args.log_dir is not None and args.phase not in ("mmlu_5shot", "gsm8k_5shot"):
        parser.error("--log-dir applies only to supplemental phases")
    return args


def _output_dir(protocol, explicit):
    return Path(explicit if explicit is not None else protocol.data["pmq"]["output_dir"]).expanduser()


def _factors_path(output_dir, explicit):
    return Path(explicit) if explicit else output_dir


def _allocation_path(output_dir, average_bit, explicit=None):
    if explicit:
        return Path(explicit)
    if average_bit is None:
        raise ValueError("allocation requires --average-bit or --allocation-path")
    return output_dir / "experts_mixture_bit_selection" / f"experts_mixture_bitwidth_combination_avg{average_bit:g}bit.pkl"


def _load_factor_payload(path):
    path = Path(path)
    if path.is_file() and path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
    else:
        filenames = {"selected_count": "experts_act_frequency.pkl", "selected_weight": "experts_act_weight.pkl",
                     "candidate_loss": "experts_quant_loss.pkl"}
        metadata_path = path / "factors_complete.json"
        if not metadata_path.exists():
            raise ValueError("PMQ factor completion record is missing")
        manifest = json.loads(metadata_path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError("PMQ factors are incomplete")
        payload = {"metadata": manifest["metadata"]}
        for key, name in filenames.items():
            if sha256_file(path / name) != manifest["files"][name]:
                raise ValueError(f"PMQ factor checksum mismatch: {name}")
            with (path / name).open("rb") as stream:
                payload[key] = pickle.load(stream)
    required = {"selected_count", "selected_weight", "candidate_loss", "metadata"}
    if not required <= set(payload):
        raise ValueError("invalid PMQ factor payload")
    return payload


def _allocate(protocol, *, factors_path, output_dir, average_bit, allocation_path=None, seed=0):
    started = perf_counter()
    path = _allocation_path(output_dir, average_bit, allocation_path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"allocation already exists: {path}")
    payload = _load_factor_payload(factors_path)
    if payload["metadata"].get("seed") != seed:
        raise ValueError("factor seed does not match allocation seed")
    if payload["metadata"].get("provenance"):
        validate_provenance(protocol, payload["metadata"]["provenance"], seed=seed)
    if payload["metadata"].get("architecture") != protocol.architecture:
        raise ValueError("factor architecture does not match the allocation model")
    pmq_cfg = protocol.data["pmq"]
    bits = protocol.candidate_bits
    if not bits[0] <= average_bit <= bits[-1]:
        raise ValueError(f"average bit must lie in [{bits[0]}, {bits[-1]}]")
    require_mid_and_high = allocation_requires_mid_and_high(
        protocol.architecture, average_bit,
        relax_mixtral_1p25_c2c3=pmq_cfg.get("relax_mixtral_1p25_c2c3", False),
    )
    losses = build_loss_by_layer(
        selected_count=payload["selected_count"], selected_weight=payload["selected_weight"],
        candidate_loss=payload["candidate_loss"], alpha=float(pmq_cfg.get("alpha", 1.0)),
        beta=float(pmq_cfg.get("beta", 1.5)), normalize_factors=bool(pmq_cfg.get("normalize_factors", True)),
        scale_factor=float(pmq_cfg.get("scale_factor", 1000.0)),
    )
    adapter = PmqMoeAdapter(protocol.architecture)
    shared_units = adapter.shared_budget_units(load_model_config(protocol.model_path))
    with component("solve_lp"):
        assignments = solve_pmq_layers(losses, average_bit=average_bit, shared_units=shared_units,
            shared_bit=int(pmq_cfg["shared_expert_bit"]), candidate_bits=bits,
            backend=pmq_cfg.get("ilp_backend", "highs"),
            require_mid_and_high=require_mid_and_high)
    native = {}
    for (layer, expert), bit in sorted(assignments.items()):
        native.setdefault(layer, {})[expert] = bit
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pickle(path, native)
    metadata = {"status": "complete", "method": "pmq", "average_bit": average_bit,
                "allocation_constraints": {"require_mid_and_high": require_mid_and_high},
                "candidate_bits": list(bits), "shared_expert_bit": pmq_cfg["shared_expert_bit"],
                "allocation": tied_projection_plan(assignments), "allocation_sha256": sha256_file(path),
                "allocation_seconds": perf_counter() - started,
                "factor_source": str(factors_path.resolve()), "provenance": provenance(protocol, seed=payload["metadata"]["seed"]),
                "objective": {key: pmq_cfg[key] for key in ("alpha", "beta", "normalize_factors", "scale_factor", "ilp_backend")}}
    write_json(path.with_suffix(".json"), metadata)
    return path


def main(argv=None):
    args = parse_args(argv)
    protocol = load_protocol_config(args.config)
    output_dir = _output_dir(protocol, args.out_dir)
    phase_name = args.phase + (f"_avg{args.average_bit:g}" if args.average_bit is not None else "")
    if args.phase in ("mmlu_5shot", "gsm8k_5shot"):
        from pmq.supplemental_entry import run
        return run(args, protocol, output_dir)
    with RunMonitor(output_dir / "logs", phase_name, interval=args.monitor_interval):
        if args.phase == "factors":
            if any((output_dir / name).exists() for name in ("experts_act_frequency.pkl", "factors_complete.json")):
                raise FileExistsError("factor files already exist; reuse them for allocation")
            factors, metadata = collect_pmq_factors(protocol, seed=args.seed)
            with component("save_factors"):
                output = save_pmq_factors(output_dir, factors, metadata)
        elif args.phase == "allocate":
            if args.average_bit is None:
                raise ValueError("allocate requires --average-bit")
            output = _allocate(protocol, factors_path=_factors_path(output_dir, args.factors_path),
                output_dir=output_dir, average_bit=args.average_bit,
                allocation_path=args.allocation_path, seed=args.seed)
        else:
            if args.saving_path:
                checkpoint = Path(args.saving_path)
            elif args.average_bit is not None:
                checkpoint = output_dir / f"avg{args.average_bit:g}"
            else:
                raise ValueError("provide --saving-path or --average-bit")
            if args.phase in {"quantize", "evaluate"}:
                allocation = _allocation_path(output_dir, args.average_bit, args.allocation_path)
                function = quantize_pmq_plan if args.phase == "quantize" else evaluate_pmq_plan
                output = function(protocol, allocation_path=allocation, seed=args.seed, output_dir=checkpoint)
            elif args.phase == "ppl":
                output = evaluate_saved_ppl(protocol, checkpoint_dir=checkpoint, seed=args.seed)
            else:
                from pmq.zeroshot import evaluate_seven, load_zeroshot_model
                identity = validate_checkpoint(checkpoint)
                validate_provenance(protocol, identity["provenance"], seed=args.seed)
                if not (checkpoint / "evaluation.json").exists():
                    raise ValueError("PPL result is required before setting completion")
                if (checkpoint / "complete.json").exists():
                    raise FileExistsError("setting completion already exists")
                with component("load_model"):
                    model = load_zeroshot_model(checkpoint)
                    tokenizer = load_tokenizer(checkpoint)
                resolve_cli_batch_cap(args, model, args.phase)
                with component("evaluate_zeroshot"):
                    output = evaluate_seven(model, tokenizer, checkpoint, seed=args.seed,
                                            batch_size=args.batch_size, max_batch_size=args.max_batch_size)
                write_json(checkpoint / "complete.json", {"status": "complete", "method": "pmq",
                    "ppl_sha256": sha256_file(checkpoint / "evaluation.json"),
                    "zeroshot_sha256": sha256_file(output),
                    "checkpoint_sha256": sha256_file(checkpoint / "qmodel.pt")})
        LOGGER.info("OUTPUT %s", output)
        return output


if __name__ == "__main__":
    main()
