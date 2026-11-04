# Supplemental MMLU and GSM8K evaluation

The two independent profiles are `mmlu_5shot` and `gsm8k_0shot`. The existing seven-task command and outputs keep their zero-shot protocol. Supplemental scores never enter its seven-task average.

Protocol: lm-eval 0.4.13; seed 0 by default; no chat template or added BOS. MMLU uses the harness dev-set first-five examples and sample-weighted accuracy over 57 subjects / 14,042 test questions. GSM8K uses zero demonstrations, the harness Question/Answer prompt and answer filters, deterministic generation, and an explicit 256-token generation budget (the HFLM default). Both strict-match and flexible-extract exact match are stored for all 1,319 test questions. The resolved task configs, including stop strings and extraction filters, are retained in `harness.configs` with a digest in `protocol`.

Each task is a separate invocation and result. Batch size defaults to 1. A smoke `--limit 1` means one question per leaf task (57 questions for MMLU), never a full benchmark result. It does not write a formal result JSON. Full runs check actual shot counts, effective sample counts and metrics before writing. Existing results/logs are refused. This interface does not schedule experiments.

Use the existing native runner:

```bash
python pmq_runner.py --config /absolute/model.yaml --saving-path /absolute/checkpoint/path --phase mmlu_5shot --log-dir /absolute/batch/logs
python pmq_runner.py --config /absolute/model.yaml --saving-path /absolute/checkpoint/path --phase gsm8k_0shot --log-dir /absolute/batch/logs
```

Results are `mmlu_5shot.json` and `gsm8k_0shot.json` beside `evaluation.json`, `zeroshot.json` and `qmodel.pt`. Logs and resource records use `<profile>_<checkpoint-directory-name>` in `--log-dir` (default: existing output directory / logs). Use model-specific log directories when checkpoint basenames coincide. `--out-dir` plus `--average-bit` can resolve the existing `avg<bit>` checkpoint instead. Existing `complete.json` is accepted and preserved; supplemental evaluation neither re-quantizes nor rewrites seven-task completion. The config/checkpoint provenance is validated by the existing PMQ loader.
