"""CPU-only native phases using a tiny model and local synthetic Arrow data."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from pmq_queue import build_steps, stage_complete
from pmq_runner import main
from pmq.zeroshot import SEVEN_TASKS


class PipelineTest(unittest.TestCase):
    def test_queue_keeps_all_ppl_before_seven(self):
        steps = list(build_steps(Path("model.yaml"), Path("output"), [1.25, 1.5], 0, "balanced_exact", 5))
        stages = [stage for stage, _, _ in steps]
        self.assertEqual(stages.count("factors"), 1)
        self.assertEqual(stages.count("ppl"), 2)
        self.assertLess(max(i for i, stage in enumerate(stages) if stage == "ppl"), stages.index("zeroshot"))

    def test_tiny_native_phases_keep_original_files(self):
        from datasets import Dataset, DatasetDict
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import AutoModelForCausalLM, Qwen2MoeConfig, PreTrainedTokenizerFast

        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder) / "workspace"
            repository = workspace / "worktrees" / "PMQ"
            config_path = repository / "configs" / "tiny.yaml"
            config_path.parent.mkdir(parents=True)
            model_path = workspace / "models" / "tiny"
            data_path = workspace / "datasets"
            cfg = Qwen2MoeConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
                moe_intermediate_size=4, shared_expert_intermediate_size=4,
                num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
                num_experts=4, num_experts_per_tok=2, max_position_embeddings=64)
            cfg._attn_implementation = "eager"
            AutoModelForCausalLM.from_config(cfg).save_pretrained(model_path)
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(model_path)
            text = "a b c d " * 12
            for name in ("wiki", "c4"):
                DatasetDict({split: Dataset.from_dict({"text": [text, text]})
                             for split in ("train", "test", "validation")}).save_to_disk(data_path / name)
            output = repository / "results" / "tiny"
            config = {
                "model": {"name": "tiny", "id": "tiny", "architecture": "qwen2_moe", "dtype": "float32", "attn_implementation": "eager", "device_map": "cpu"},
                "dataset": {"factors": {"name": "c4", "split": "train", "protocol": "random_document_windows"},
                    "gptq_calibration": {"name": "wiki", "split": "train", "protocol": "corpus_windows"},
                    "evaluations": {"wikitext2": {"name": "wiki", "split": "test", "protocol": "corpus_windows", "seq_len": 8, "max_blocks": 2},
                                    "c4": {"name": "c4", "split": "validation", "protocol": "gptq_c4_new", "text_count": 2, "seq_len": 8, "max_blocks": 2}}},
                "calibration": {"factors": {"samples": 2, "seq_len": 8}, "gptq": {"samples": 2, "seq_len": 8}},
                "quantization": {"group_size": 4, "percdamp": .01, "candidate_quantizer": "normal", "standard_linear_bit": 4, "router_bit": 16},
                "pmq": {"output_dir": str(output), "candidate_bits": [1, 2, 3], "shared_expert_bit": 3,
                        "alpha": 1., "beta": 1.5, "normalize_factors": True, "scale_factor": 1000., "ilp_backend": "highs"},
            }
            config_path.write_text(yaml.safe_dump(config))
            common = ["--config", str(config_path), "--seed", "0"]
            sample = {"time": 0, "pid": 1, "rss_bytes": 10, "available_bytes": 20,
                      "system_swap_used_bytes": 0, "gpu": [], "cuda_process_memory": []}
            with patch("pmq.monitor.resource_snapshot", return_value=sample):
                main(common + ["--phase", "factors"])
                for phase in ("allocate", "quantize", "ppl"):
                    main(common + ["--phase", phase, "--average-bit", "2"])
                result = {"results": {task: {"acc,none": .5} for task in SEVEN_TASKS}, "n-shot": {task: 0 for task in SEVEN_TASKS}}
                with patch("lm_eval.models.huggingface.HFLM"), patch("lm_eval.evaluator.simple_evaluate", return_value=result):
                    main(common + ["--phase", "zeroshot", "--average-bit", "2"])
            for phase in ("factors", "allocate", "quantize", "ppl", "zeroshot"):
                self.assertTrue(stage_complete(phase, output, None if phase == "factors" else 2.))
            self.assertTrue((output / "experts_act_frequency.pkl").exists())
            self.assertTrue((output / "experts_mixture_bit_selection" / "experts_mixture_bitwidth_combination_avg2bit.pkl").exists())
            self.assertTrue((output / "avg2" / "qmodel.pt").exists())
            self.assertEqual(json.loads((output / "avg2" / "complete.json").read_text())["status"], "complete")
            self.assertTrue((output / "logs" / "ppl_avg2_resources.jsonl").exists())


if __name__ == "__main__": unittest.main()
