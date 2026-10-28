"""CPU regressions for real native MoE layouts, packing and stage observability."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pmq.adapters import PmqMoeAdapter
from pmq.gptq import TensorGPTQ
from pmq.packed import PackedLinear, load_packed_model, save_packed_model
from pmq.quantize import apply_allocation
from pmq.runtime import align_deepseek_softmax_scale
from pmq.monitor import RunMonitor
from pmq.zeroshot import summarize_seven, SEVEN_TASKS


class NativePmqTest(unittest.TestCase):
    def test_deepseek_remote_router_tuple(self):
        adapter = PmqMoeAdapter("deepseek_moe")
        indices = torch.tensor([[2, 0]])
        weights = torch.tensor([[.3, .2]])
        actual = adapter.route_state(None, (indices, weights, None), None)
        self.assertTrue(torch.equal(actual[0], indices))
        self.assertTrue(torch.equal(actual[1], weights))

    def test_module_list_experts_roundtrip(self):
        expert = nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(expert, name, nn.Linear(4, 4, bias=False))
        module = nn.Module(); module.experts = nn.ModuleList([expert])
        adapter = PmqMoeAdapter("deepseek_moe")
        values = {key: value.clone() for key, value in adapter.expert_weights(module, 0).items()}
        adapter.set_expert_weights(module, 0, {key: value + 1 for key, value in values.items()})
        for key, value in adapter.expert_weights(module, 0).items():
            self.assertTrue(torch.equal(value, values[key] + 1))

    def test_packed_grid_exactness_and_dtype_moves(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for bit in (1, 2, 3, 4):
                torch.manual_seed(17)
                source = torch.randn(5, 11, dtype=dtype)
                solver = TensorGPTQ(source, bits=bit)
                solver.add_batch(torch.randn(1, 15, 11).to(dtype))
                expected = solver.quantize(group_size=4, percdamp=.01)
                layer = PackedLinear(solver.packed_state)
                self.assertTrue(torch.equal(layer.dequantize(), expected), (dtype, bit))
                layer.to("cpu")
                self.assertEqual(layer.dequantize().dtype, dtype)
                self.assertTrue(torch.equal(layer.dequantize(), expected))
                self.assertEqual(layer.scale.dtype, torch.float32)

    def test_yarn_scale_idempotent(self):
        model = nn.Module()
        model.config = SimpleNamespace(model_type="deepseek_v2", rope_scaling={"factor": 40, "mscale_all_dim": .707})
        attention = nn.Module(); attention.qk_head_dim = 192; attention.scaling = 192 ** -.5
        model.attention = attention
        self.assertEqual(align_deepseek_softmax_scale(model), 1)
        first = attention.scaling
        self.assertEqual(align_deepseek_softmax_scale(model), 0)
        self.assertEqual(attention.scaling, first)

    def _native_checkpoint(self, architecture):
        from transformers import AutoModelForCausalLM
        if architecture == "deepseek_moe":
            from transformers import DeepseekV2Config
            cfg = DeepseekV2Config(
                vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
                q_lora_rank=8, kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4,
                v_head_dim=4, first_k_dense_replace=1, max_position_embeddings=64,
            )
        else:
            from transformers import Qwen2MoeConfig
            cfg = Qwen2MoeConfig(
                vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
                shared_expert_intermediate_size=16, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=2, num_experts=4,
                num_experts_per_tok=2, max_position_embeddings=64,
            )
        cfg._attn_implementation = "eager"
        cfg._experts_implementation = "eager"
        torch.manual_seed(3)
        model = AutoModelForCausalLM.from_config(cfg).eval()
        adapter = PmqMoeAdapter(architecture)
        allocation = {(layer, expert): 1 + expert % 3
                      for layer, module in adapter.collect_moe_modules(model).items()
                      for expert in range(adapter.num_experts(module))}
        ids = torch.randint(0, 32, (2, 8))
        protocol = SimpleNamespace(
            model_path=Path("unused"), gptq_calibration_path=Path("unused"),
            data={"dataset": {"gptq_calibration": {}}, "calibration": {"gptq": {"samples": 2, "seq_len": 8}},
                  "quantization": {"group_size": 4, "percdamp": .01, "standard_linear_bit": 4, "router_bit": 16},
                  "pmq": {"shared_expert_bit": 3}},
        )
        from pmq.packed import install_packed_model as install
        def verified_install(target, layout, states):
            with torch.inference_mode():
                dense = target(input_ids=ids, use_cache=False).logits.clone()
                install(target, layout, states)
                packed = target(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(packed, dense, rtol=0, atol=0)
        with patch("pmq.quantize.load_tokenizer"), patch("pmq.quantize.calibration_inputs", return_value=ids), patch("pmq.packed.install_packed_model", side_effect=verified_install):
            apply_allocation(model, adapter, allocation, protocol=protocol, seed=0)
        with torch.inference_mode():
            before = model(input_ids=ids, use_cache=False).logits.clone()
        self.assertTrue(torch.isfinite(before).all())
        class Tokenizer:
            def save_pretrained(self, path):
                Path(path, "tokenizer_config.json").write_text("{}")
        with tempfile.TemporaryDirectory() as directory:
            save_packed_model(model, Tokenizer(), directory)
            state = torch.load(Path(directory, "qmodel.pt"), weights_only=False)
            self.assertTrue(any("W_q" in item for item in state.values()))
            from inference import load_quantized_model
            loaded = load_quantized_model(directory, {"device_map": "cpu"})
            with torch.inference_mode():
                after = loaded(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(after, before, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                save_packed_model(model, Tokenizer(), directory)

    def test_native_deepseek_runtime_loads_a_tiny_checkpoint(self):
        from transformers import DeepseekV2Config, AutoModelForCausalLM
        from pmq.runtime import load_model, load_model_config
        cfg = DeepseekV2Config(vocab_size=16, hidden_size=8, intermediate_size=16,
            moe_intermediate_size=4, num_hidden_layers=1, num_attention_heads=1,
            num_key_value_heads=1, n_routed_experts=4, n_shared_experts=1,
            num_experts_per_tok=2, q_lora_rank=4, kv_lora_rank=4,
            qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4, first_k_dense_replace=0)
        cfg._attn_implementation = "eager"
        with tempfile.TemporaryDirectory() as directory:
            AutoModelForCausalLM.from_config(cfg).save_pretrained(directory)
            path = Path(directory, "config.json")
            raw = json.loads(path.read_text()); raw["auto_map"] = {"AutoModelForCausalLM": "missing_legacy.Model"}
            path.write_text(json.dumps(raw))
            protocol = SimpleNamespace(model_path=Path(directory), data={"model": {"dtype": "float32", "device_map": "cpu", "attn_implementation": "eager"}})
            loaded = load_model(protocol)
            self.assertTrue(type(loaded).__module__.startswith("transformers.models.deepseek_v2"))
            self.assertFalse(loaded.training)
            self.assertEqual(load_model_config(directory).model_type, "deepseek_v2")

    def test_real_qwen_native_packed_checkpoint(self):
        self._native_checkpoint("qwen2_moe")

    def test_real_deepseek_native_packed_checkpoint(self):
        self._native_checkpoint("deepseek_moe")

    def test_monitor_records_failure_and_closes(self):
        sample = {"time": 0, "pid": 1, "rss_bytes": 10, "available_bytes": 20,
                  "system_swap_used_bytes": 0, "gpu": [], "cuda_process_memory": []}
        with tempfile.TemporaryDirectory() as directory, patch("pmq.monitor.resource_snapshot", return_value=sample):
            with self.assertRaisesRegex(ValueError, "test failure"):
                with RunMonitor(directory, "test", interval=.01):
                    raise ValueError("test failure")
            result = json.loads(Path(directory, "test_resources.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertGreaterEqual(result["samples"], 2)
            self.assertTrue(Path(directory, "test_resources.jsonl").exists())

    def test_seven_summary_does_not_count_mmlu_subjects_again(self):
        raw = {"results": {task: {"acc,none": .4, "acc_norm,none": .6} for task in SEVEN_TASKS}}
        raw["results"]["mmlu_x"] = {"acc,none": 0}
        self.assertAlmostEqual(summarize_seven(raw)["mean"], .6)


if __name__ == "__main__":
    unittest.main()
