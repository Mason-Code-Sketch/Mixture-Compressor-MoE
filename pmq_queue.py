"""Explicit PMQ queue: factors once, all PPL first, then seven tasks per model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pmq.artifacts import sha256_file, write_json, validate_provenance
from pmq.config import load_protocol_config
from pmq.evaluate import validate_checkpoint
from pmq_runner import _allocation_path, _load_factor_payload


def stage_complete(stage, root, bit):
    """Reuse complete artifacts only; partial paths are left for diagnosis."""
    checkpoint = root / f"avg{bit:g}" if bit is not None else None
    if stage == "factors":
        if not (root / "factors_complete.json").exists():
            return False
        _load_factor_payload(root)
    elif stage == "allocate":
        allocation = _allocation_path(root, bit)
        if not allocation.with_suffix(".json").exists():
            return False
        data = json.loads(allocation.with_suffix(".json").read_text())
        if data.get("status") != "complete" or sha256_file(allocation) != data["allocation_sha256"]:
            raise ValueError("allocation identity check failed")
    elif stage == "quantize":
        if not (checkpoint / "checkpoint_complete.json").exists():
            return False
        validate_checkpoint(checkpoint)
    elif stage == "ppl":
        path = checkpoint / "evaluation.json"
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        identity = validate_checkpoint(checkpoint)
        if data.get("status") != "complete" or data["checkpoint_sha256"] != identity["files"]["qmodel.pt"]:
            raise ValueError("PPL checkpoint identity check failed")
    elif stage == "zeroshot":
        path = checkpoint / "complete.json"
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        if data.get("status") != "complete":
            raise ValueError("setting completion is invalid")
        for file, key in [("zeroshot.json", "zeroshot_sha256"), ("evaluation.json", "ppl_sha256"), ("qmodel.pt", "checkpoint_sha256")]:
            if sha256_file(checkpoint / file) != data[key]:
                raise ValueError(f"setting identity check failed: {file}")
    return True


def validate_reuse(stage, root, bit, protocol, seed, policy):
    if stage == "factors":
        metadata = _load_factor_payload(root)["metadata"]["provenance"]
    elif stage == "allocate":
        data = json.loads(_allocation_path(root, bit).with_suffix(".json").read_text())
        metadata = data["provenance"]
        if data["budget"]["policy"] != policy or data["budget"]["requested_average_bit"] != bit:
            raise ValueError("PMQ allocation budget policy changed")
    else:
        metadata = json.loads((root / f"avg{bit:g}" / "checkpoint_complete.json").read_text())["provenance"]
    validate_provenance(protocol, metadata, seed=seed)


def build_steps(config, output, bits, seed, policy, monitor_interval):
    base = [sys.executable, "-B", "pmq_runner.py", "--config", str(config),
            "--out-dir", str(output), "--seed", str(seed),
            "--monitor-interval", str(monitor_interval)]
    if policy:
        base += ["--budget-policy", policy]
    yield "factors", None, base + ["--phase", "factors"]
    for bit in bits:
        for stage in ("allocate", "quantize", "ppl"):
            yield stage, bit, base + ["--phase", stage, "--average-bit", str(bit)]
    for bit in bits:
        yield "zeroshot", bit, base + ["--phase", "zeroshot", "--average-bit", str(bit)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--bits", nargs="+", type=float, default=[1.25, 1.5, 1.75, 2., 2.25, 2.5])
    parser.add_argument("--out-root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-policy", choices=["strict", "floor", "balanced_exact"])
    parser.add_argument("--monitor-interval", type=float, default=5.)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.bits) != len(set(args.bits)):
        raise ValueError("duplicate bit settings")
    configurations = [load_protocol_config(path) for path in args.configs]
    # Resolve metadata and budget representability before any GPU command starts.
    from pmq.adapters import PmqMoeAdapter
    from pmq.runtime import load_model_config
    from pmq.allocation import validate_layer_budget
    for protocol in configurations:
        config = load_model_config(protocol.model_path)
        routed = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
        shared = PmqMoeAdapter(protocol.architecture).shared_budget_units(config)
        policy = args.budget_policy or protocol.data["pmq"].get("budget_policy", "strict")
        if policy == "strict":
            for bit in args.bits:
                validate_layer_budget(num_routed_experts=routed, average_bit=bit,
                                      shared_units=shared, shared_bit=protocol.data["pmq"]["shared_expert_bit"])
    for protocol in configurations:
        output = Path(args.out_root) / protocol.source_path.stem if args.out_root else Path(protocol.data["pmq"]["output_dir"])
        for stage, bit, command in build_steps(protocol.source_path, output, args.bits, args.seed,
                                               args.budget_policy, args.monitor_interval):
            if args.dry_run:
                print(json.dumps({"stage": stage, "bit": bit, "command": command}))
                continue
            if stage_complete(stage, output, bit):
                validate_reuse(stage, output, bit, protocol, args.seed,
                    args.budget_policy or protocol.data["pmq"].get("budget_policy", "strict"))
                print(f"REUSE stage={stage} bit={bit}", flush=True)
                continue
            subprocess.run(command, check=True, cwd=str(protocol.repository_root))
            if not stage_complete(stage, output, bit):
                raise RuntimeError(f"native PMQ exited without valid completion: {stage} {bit}")
        if not args.dry_run and not (output / "all_complete.json").exists():
            write_json(output / "all_complete.json", {"status": "complete", "settings": len(args.bits),
                "ppl_values": 2 * len(args.bits), "seven_groups": len(args.bits),
                "bits": args.bits, "seed": args.seed,
                "further_settings_authorized": False, "action": "await_user_decision"})


if __name__ == "__main__":
    main()
