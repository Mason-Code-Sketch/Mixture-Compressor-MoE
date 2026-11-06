# Supplemental MMLU and GSM8K evaluation

The two independent profiles are `mmlu_5shot` and `gsm8k_0shot`. The existing seven-task command and outputs keep their zero-shot protocol. Supplemental scores never enter its seven-task average.

Protocol: lm-eval 0.4.13; seed 0 by default; no chat template or added BOS. MMLU uses the harness dev-set first-five examples and sample-weighted accuracy over 57 subjects / 14,042 test questions. GSM8K uses zero demonstrations, the harness Question/Answer prompt and answer filters, deterministic generation, and an explicit 256-token generation budget (the HFLM default). Both strict-match and flexible-extract exact match are stored for all 1,319 test questions. The resolved task configs, including stop strings and extraction filters, are retained in `harness.configs` with a digest in `protocol`.

Each task is a separate invocation and result. Batch size defaults to `auto`, with model-specific caps listed below. Use `--batch-size 1` (or another positive integer) for a fixed batch, and `--max-batch-size 4` to lower the automatic cap. The cap applies to automatic sizing; an explicit fixed batch is used as requested. Actual batch sizes depend on available memory and input lengths. A smoke `--limit 1` means one question per leaf task (57 questions for MMLU), never a full benchmark result. It does not write a formal result JSON. Full runs check actual shot counts, effective sample counts and metrics before writing. Existing results/logs are refused. This interface does not schedule experiments.

Use the existing native runner:

```bash
python pmq_runner.py --config /absolute/model.yaml --saving-path /absolute/checkpoint/path --phase mmlu_5shot --log-dir /absolute/batch/logs
python pmq_runner.py --config /absolute/model.yaml --saving-path /absolute/checkpoint/path --phase gsm8k_0shot --log-dir /absolute/batch/logs
```

Results are `mmlu_5shot.json` and `gsm8k_0shot.json` beside `evaluation.json`, `zeroshot.json` and `qmodel.pt`. Logs and resource records use `<profile>_<checkpoint-directory-name>` in `--log-dir` (default: existing output directory / logs). Use model-specific log directories when checkpoint basenames coincide. `--out-dir` plus `--average-bit` can resolve the existing `avg<bit>` checkpoint instead. Existing `complete.json` is accepted and preserved; supplemental evaluation neither re-quantizes nor rewrites seven-task completion. The config/checkpoint provenance is validated by the existing PMQ loader.

The existing seven-task zeroshot entry also defaults to `auto` with the model-specific cap listed below; its batch options accept the same manual overrides.

For GSM8K auto batching, the probe length is the longest effective request input plus its generation budget (256 tokens), capped by the unchanged model context limit. The native five-forward-pass probe and OOM batch reduction are retained; this is not a full generation benchmark. Formal input truncation, generation and scoring are unchanged. `GSM8K_BATCH_PROBE` logs and `protocol.generation_batch_probes` record the lengths and selected batch. Fixed batch settings bypass this probe.

### Model-specific auto batch defaults

The loaded model config selects the auto cap. An explicit maximum batch argument overrides these defaults; a fixed batch remains explicit.

| Loaded `model_type` | Seven tasks | MMLU 5-shot | GSM8K 0-shot |
|---|---:|---:|---:|
| `deepseek_v2` | 8 | 2 | 16 |
| `qwen2_moe` | 12 | 2 | 24 |
| `mixtral` | 34 | 8 | 48 |
| `qwen3_moe` and other models | 51 | 17 | 96 |

DeepSeek-V2-Lite and Qwen1.5-MoE-A2.7B defaults are initial estimates for four 24 GB RTX 4090 cards with decoder layers resident across the cards. Selection uses `model_type`, without hardware detection. Historical GEMQ `grouped_cached` seven-task runs completed at batch 8 for both models; Qwen1.5 batch 12 and both models' MMLU/GSM caps remain estimates pending GPU validation. Different backends or placement may need an explicit cap. Mixtral 34 passed complete KDMoQ seven-task evaluation; its 8 and 48 caps remain estimates. Tasks, few-shot settings and generation budgets are unchanged.
