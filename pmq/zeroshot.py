"""Seven-task zero-shot evaluation of the saved native PMQ checkpoint."""

from __future__ import annotations

from pathlib import Path
from importlib.metadata import version

from pmq.artifacts import sha256_file, write_json
from pmq.monitor import LOGGER


SEVEN_TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "mathqa", "mmlu")


def load_zeroshot_model(checkpoint_dir, *, device_map="auto", dtype=None):
    """Choose seven-task attention when constructing the saved model skeleton."""
    from pmq.runtime import load_model_config
    from pmq.packed import load_packed_model

    config = load_model_config(checkpoint_dir)
    attention = {"deepseek_v2": "eager", "qwen2_moe": "sdpa",
                 "qwen3_moe": "sdpa", "mixtral": "sdpa"}.get(config.model_type)
    model = load_packed_model(checkpoint_dir, device_map=device_map, dtype=dtype,
                              attn_implementation=attention)
    if attention is not None and model.config._attn_implementation != attention:
        raise RuntimeError(f"seven-task attention must be {attention}")
    return model


def summarize_seven(raw):
    """Prefer acc_norm and include the MMLU group exactly once."""
    selected = {}
    for task in SEVEN_TASKS:
        metrics = raw["results"][task]
        key = next((name for name in ("acc_norm,none", "acc,none", "acc_norm", "acc") if name in metrics), None)
        if key is None:
            raise ValueError(f"seven-task result lacks accuracy for {task}")
        value = float(metrics[key])
        if not 0 <= value <= 1:
            raise ValueError(f"invalid accuracy for {task}: {value}")
        selected[task] = {"metric": key, "value": value}
    return {"tasks": selected, "mean": sum(value["value"] for value in selected.values()) / len(SEVEN_TASKS)}


def evaluate_seven(model, tokenizer, checkpoint_dir, *, seed=0):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    if version("lm_eval") != "0.4.13":
        raise RuntimeError("PMQ seven-task evaluation requires lm_eval==0.4.13")
    checkpoint_dir = Path(checkpoint_dir)
    output = checkpoint_dir / "zeroshot.json"
    if output.exists():
        raise FileExistsError(f"seven-task result already exists: {output}")
    model.eval()
    from pmq.evaluation_backend import prepare_evaluation_model
    backend = prepare_evaluation_model(model)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto", max_batch_size=8,
              add_bos_token=False, parallelize=True)
    attention = getattr(getattr(model, "config", None), "_attn_implementation", None)
    LOGGER.info("SEVEN_START tasks=%s num_fewshot=0 add_bos_token=False attention=%s",
                SEVEN_TASKS, attention)
    raw = evaluator.simple_evaluate(
        model=lm, tasks=list(SEVEN_TASKS), num_fewshot=0, limit=None,
        apply_chat_template=False, log_samples=False,
        random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed,
    )
    if raw is None:
        raise RuntimeError("seven-task evaluator returned no result")
    if any(value != 0 for value in raw.get("n-shot", {}).values()):
        raise ValueError("seven-task evaluator did not use zero-shot everywhere")
    summary = summarize_seven(raw)
    write_json(output, {
        "experiment": {"method": "pmq", "checkpoint": str(checkpoint_dir.resolve()),
                       "checkpoint_sha256": sha256_file(checkpoint_dir / "qmodel.pt"),
                       "seed": seed, "tasks": list(SEVEN_TASKS), "num_fewshot": 0,
                       "add_bos_token": False, "apply_chat_template": False, "limit": None,
                       "batch_size": "auto", "max_batch_size": 8, "evaluation_backend": backend,
                       "attn_implementation": attention},
        "summary": summary, "harness": raw,
    })
    LOGGER.info("SEVEN_COMPLETE mean=%.8f", summary["mean"])
    return output
