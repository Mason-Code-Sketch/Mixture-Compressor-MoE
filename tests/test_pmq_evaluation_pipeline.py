"""Independent PPL and seven-task phases reuse one saved PMQ checkpoint."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
import yaml


def test_saved_checkpoint_phases_use_resident_grouped_backend(tmp_path):
    from datasets import Dataset, DatasetDict
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, Qwen2MoeConfig, PreTrainedTokenizerFast
    from pmq.artifacts import sha256_file
    from pmq.evaluation_backend import GroupedExperts
    from pmq.zeroshot import SEVEN_TASKS
    from pmq_runner import main

    workspace = tmp_path / "workspace"
    repository = workspace / "worktrees" / "PMQ"
    config_path = repository / "configs" / "tiny.yaml"
    config_path.parent.mkdir(parents=True)
    model_path = workspace / "models" / "tiny"
    cfg = Qwen2MoeConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    AutoModelForCausalLM.from_config(cfg).to(torch.float16).save_pretrained(model_path)
    raw_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5}, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(model_path)
    text = "a b c d " * 12
    for name in ("wiki", "c4"):
        DatasetDict({split: Dataset.from_dict({"text": [text, text]})
                     for split in ("train", "test", "validation")}).save_to_disk(workspace / "datasets" / name)
    output = repository / "results" / "tiny"
    config = {
        "model": {"name": "tiny", "id": "tiny", "architecture": "qwen2_moe", "dtype": "float16", "device_map": "cpu", "attn_implementation": "eager"},
        "dataset": {"factors": {"name": "c4", "split": "train", "protocol": "random_document_windows"},
            "gptq_calibration": {"name": "wiki", "split": "train", "protocol": "corpus_windows"},
            "evaluations": {"wikitext2": {"name": "wiki", "split": "test", "protocol": "corpus_windows", "seq_len": 8, "max_blocks": 2},
                "c4": {"name": "c4", "split": "validation", "protocol": "gptq_c4_new", "seq_len": 8, "text_count": 2, "max_blocks": 2}}},
        "calibration": {"factors": {"samples": 2, "seq_len": 8}, "gptq": {"samples": 2, "seq_len": 8}},
        "quantization": {"group_size": 4, "percdamp": .01, "standard_linear_bit": 4, "router_bit": 16},
        "pmq": {"output_dir": str(output), "candidate_bits": [1, 2, 3], "shared_expert_bit": 3},
    }
    config_path.write_text(yaml.safe_dump(config))
    allocation = repository / "allocation.json"
    allocation.write_text(json.dumps({"allocation": {f"0,{expert}": {name: 1 + expert % 3
        for name in ("gate_proj", "up_proj", "down_proj")} for expert in range(4)}}))
    checkpoint = output / "avg2"
    common = ["--config", str(config_path), "--seed", "0", "--average-bit", "2", "--saving-path", str(checkpoint)]
    main(common + ["--phase", "quantize", "--allocation-path", str(allocation)])
    digest = sha256_file(checkpoint / "qmodel.pt")
    with patch("pmq.quantize.apply_allocation", side_effect=AssertionError("evaluation must not quantize again")):
        main(common + ["--phase", "ppl"])
        raw = {"results": {task: {"acc,none": .5} for task in SEVEN_TASKS}, "n-shot": {task: 0 for task in SEVEN_TASKS}}
        with patch("lm_eval.models.huggingface.HFLM") as wrapped, patch("lm_eval.evaluator.simple_evaluate", return_value=raw):
            main(common + ["--phase", "zeroshot"])
            evaluated = wrapped.call_args.kwargs["pretrained"]
            assert any(isinstance(module, GroupedExperts) for module in evaluated.modules())
            assert evaluated.config._attn_implementation == "sdpa"
    assert sha256_file(checkpoint / "qmodel.pt") == digest
    ppl = json.loads((checkpoint / "evaluation.json").read_text())
    seven = json.loads((checkpoint / "zeroshot.json").read_text())
    assert seven["experiment"]["attn_implementation"] == "sdpa"
    assert ppl["evaluation_backend"] == seven["experiment"]["evaluation_backend"] == "grouped_cached"
    assert ppl["checkpoint_sha256"] == seven["experiment"]["checkpoint_sha256"] == digest
    assert json.loads((checkpoint / "complete.json").read_text())["status"] == "complete"
