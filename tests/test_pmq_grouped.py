"""Real grouped GEMM, packed-weight identity and native-model regression checks."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from pmq.evaluation_backend import CachedLinear, GroupedExperts, prepare_evaluation_model
from pmq.packed import PackedLinear, pack_matrix


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_packed_projection_decodes_once_and_keeps_grid(bits):
    torch.manual_seed(bits)
    codes = torch.randint(0, 2**bits, (16, 16), dtype=torch.uint8)
    scale = torch.full((16, 4), .125)
    zero = torch.full((16, 4), .5 if bits == 1 else 1.25)
    state = pack_matrix(codes, scale, zero, bits=bits, group_size=4, dtype=torch.bfloat16)
    source = PackedLinear(state)
    before = state["W_q"].clone()
    with patch.object(source, "dequantize", wraps=source.dequantize) as decode:
        cached = CachedLinear(source)
        expected = nn.functional.linear(torch.ones(3, 16).bfloat16(), source.dequantize())
        for _ in range(2):
            assert torch.equal(cached(torch.ones(3, 16).bfloat16()), expected)
        assert decode.call_count == 2  # one conversion, one independent reference
    assert torch.equal(state["W_q"], before)
    assert cached.weight.dtype == torch.bfloat16


def tiny_model(kind, dtype=torch.bfloat16):
    transformers = pytest.importorskip("transformers")
    shared = dict(vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                  num_key_value_heads=2, max_position_embeddings=64)
    if kind == "deepseek_v2":
        cfg = transformers.DeepseekV2Config(**shared, intermediate_size=32, moe_intermediate_size=16,
            n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, q_lora_rank=8,
            kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
            first_k_dense_replace=0)
    elif kind == "qwen2_moe":
        cfg = transformers.Qwen2MoeConfig(**shared, intermediate_size=32, moe_intermediate_size=16,
            shared_expert_intermediate_size=16, num_experts=4, num_experts_per_tok=2)
    elif kind == "qwen3_moe":
        cfg = transformers.Qwen3MoeConfig(**shared, intermediate_size=32, moe_intermediate_size=16,
            num_experts=4, num_experts_per_tok=2, head_dim=8)
    else:
        cfg = transformers.MixtralConfig(**shared, intermediate_size=16, num_local_experts=4,
            num_experts_per_tok=2, head_dim=8)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    torch.manual_seed(23)
    return transformers.AutoModelForCausalLM.from_config(cfg).to(dtype).eval()


@pytest.mark.parametrize("kind", ["deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"])
def test_native_model_logits_and_router_identity(kind):
    model = tiny_model(kind)
    ids = torch.randint(0, 32, (2, 9))
    routers = [(name, item, item.weight.detach().clone()) for name, item in model.named_modules()
               if name.endswith(".gate")]
    with torch.inference_mode():
        reference = model(ids, use_cache=False).logits
        assert prepare_evaluation_model(model, backend="grouped_cached") == "grouped_cached"
        actual = model(ids, use_cache=False).logits
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert prepare_evaluation_model(model) == "grouped_cached"
    for name, router, values in routers:
        assert model.get_submodule(name) is router
        assert torch.equal(router.weight, values)
    assert all(not item.training for item in model.modules() if isinstance(item, GroupedExperts))


def test_auto_keeps_float32_validation_models_and_explicit_request_fails():
    model = nn.Module(); model.config = SimpleNamespace(model_type="qwen3_moe")
    model.weight = nn.Parameter(torch.ones(1))
    assert prepare_evaluation_model(model) == "pytorch"
    with pytest.raises(ValueError, match="FP16/BF16"):
        prepare_evaluation_model(model, backend="grouped_cached")


@pytest.mark.parametrize("kind", ["deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantized_checkpoint_reload_and_grouped_logits(kind, dtype, tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from pmq.adapters import PmqMoeAdapter
    from pmq.quantize import apply_allocation
    from pmq.packed import load_packed_model, save_packed_model
    from pmq.artifacts import sha256_file

    model = tiny_model(kind, dtype=dtype).to(device)
    architecture = "deepseek_moe" if kind == "deepseek_v2" else kind
    adapter = PmqMoeAdapter(architecture)
    ids = torch.randint(0, 32, (2, 8), device=device)
    allocation = {(layer, expert): 1 + expert % 3
                  for layer, module in adapter.collect_moe_modules(model).items()
                  for expert in range(adapter.num_experts(module))}
    protocol = SimpleNamespace(model_path=Path("unused"), gptq_calibration_path=Path("unused"),
        data={"dataset": {"gptq_calibration": {}}, "calibration": {"gptq": {"samples": 2, "seq_len": 8}},
              "quantization": {"group_size": 4, "percdamp": .01, "standard_linear_bit": 4, "router_bit": 16},
              "pmq": {"shared_expert_bit": 3}})
    with patch("pmq.quantize.load_tokenizer"), patch("pmq.quantize.calibration_inputs", return_value=ids):
        apply_allocation(model, adapter, allocation, protocol=protocol, seed=0)
    with torch.inference_mode():
        reference = model(ids, use_cache=False).logits.clone()
    class Tokenizer:
        def save_pretrained(self, path):
            Path(path, "tokenizer_config.json").write_text("{}")
    save_packed_model(model, Tokenizer(), tmp_path)
    digest = sha256_file(tmp_path / "qmodel.pt")
    reloaded = load_packed_model(tmp_path, device_map=device)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids, use_cache=False).logits, reference, rtol=0, atol=0)
        original_decode = PackedLinear.dequantize
        calls = []
        def decode_once(layer):
            calls.append(id(layer))
            return original_decode(layer)
        with patch.object(PackedLinear, "dequantize", decode_once):
            prepare_evaluation_model(reloaded, backend="grouped_cached")
            assert len(calls) == len(set(calls)) and calls
            count = len(calls)
            for _ in range(2):
                torch.testing.assert_close(reloaded(ids, use_cache=False).logits, reference, rtol=0, atol=0)
            assert len(calls) == count
    # PPL keeps checkpoint attention; zeroshot constructs its own attention classes.
    assert reloaded.config._attn_implementation == "eager"
    from pmq.zeroshot import load_zeroshot_model
    zeroshot = load_zeroshot_model(tmp_path, device_map=device)
    expected_attention = "eager" if kind == "deepseek_v2" else "sdpa"
    assert zeroshot.config._attn_implementation == expected_attention
    prepare_evaluation_model(zeroshot, backend="grouped_cached")
    with torch.inference_mode(), patch("torch.nn.functional.scaled_dot_product_attention",
            wraps=nn.functional.scaled_dot_product_attention) as sdpa:
        logits = zeroshot(ids, use_cache=False).logits
    assert torch.isfinite(logits).all()
    assert (sdpa.call_count > 0) == (expected_attention == "sdpa")
    # The override never changes saved attention or quantized weights.
    import json
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["pmq_quantization"]["attn_implementation"] == "eager"
    assert sha256_file(tmp_path / "qmodel.pt") == digest
