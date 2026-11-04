"""Supplemental PMQ evaluation, preserving completed seven-task receipts."""
import json
from pathlib import Path

from pmq.artifacts import validate_provenance
from pmq.evaluate import validate_checkpoint
from pmq.monitor import RunMonitor, component, LOGGER
from pmq.runtime import load_tokenizer
from pmq.zeroshot import load_zeroshot_model
from pmq.supplemental import evaluate


def run(args, protocol, output_dir):
    if args.saving_path:
        checkpoint = Path(args.saving_path).expanduser().resolve()
    elif args.average_bit is not None:
        checkpoint = (output_dir / f"avg{args.average_bit:g}").resolve()
    else:
        raise ValueError("provide --saving-path or --average-bit")
    output = checkpoint / (args.phase + ".json")
    if args.limit is None and output.exists():
        raise FileExistsError(output)
    log_dir = args.log_dir or output_dir / "logs"
    # Explicit checkpoint basename prevents logs for different bits from colliding.
    stage = f"{args.phase}_{checkpoint.name}" + ("_smoke" if args.limit is not None else "")
    # RunMonitor otherwise archives previous attempts. Supplemental runs never do so.
    if any(log_dir.glob(stage + "*")):
        raise FileExistsError(f"supplemental logs already exist: {log_dir / stage}")
    identity = validate_checkpoint(checkpoint)
    validate_provenance(protocol, identity["provenance"], seed=args.seed)
    with RunMonitor(log_dir, stage, interval=args.monitor_interval):
        with component("load_model"):
            model = load_zeroshot_model(checkpoint)
            tokenizer = load_tokenizer(checkpoint)
            from pmq.evaluation_backend import prepare_evaluation_model
            backend = prepare_evaluation_model(model)
        with component(args.phase):
            payload = evaluate(model, tokenizer, args.phase, seed=args.seed, batch_size=1, limit=args.limit)
        payload["experiment"] = {"method": "pmq", "checkpoint": str(checkpoint),
                                 "checkpoint_sha256": identity["files"]["qmodel.pt"],
                                 "provenance": identity["provenance"],
                                 "evaluation_backend": backend,
                                 "log_path": str((log_dir / (stage + ".log")).resolve())}
        LOGGER.info("SUPPLEMENTAL_RESULT %s", json.dumps(payload["summary"]))
        if args.limit is not None:
            return None
        from lm_eval.utils import handle_non_serializable
        # Exclusive creation also guards against a competing writer.
        with output.open("x") as stream:
            json.dump(payload, stream, indent=2, default=handle_non_serializable)
            stream.write("\n")
        return output
