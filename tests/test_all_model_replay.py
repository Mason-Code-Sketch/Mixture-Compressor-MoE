"""Native MoE replay must preserve every capture and sequential GPTQ result."""
import copy
import os
from types import SimpleNamespace

import pytest
import torch
import transformers

from pmq import quantize as q
from pmq.adapters import PmqMoeAdapter
from pmq.layer_inputs import LayerInputs, can_capture_native_layers, can_replay_layers


KINDS = ("deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral")


def make_model(kind, dtype):
    common = dict(vocab_size=32, hidden_size=16, num_hidden_layers=3,
                  num_attention_heads=2, num_key_value_heads=2,
                  max_position_embeddings=64, attention_dropout=0.)
    if kind == "deepseek_v2":
        cfg = transformers.DeepseekV2Config(**common, intermediate_size=32,
            moe_intermediate_size=8, n_routed_experts=4, n_shared_experts=1,
            num_experts_per_tok=2, first_k_dense_replace=1, q_lora_rank=8,
            kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4)
    elif kind == "qwen2_moe":
        cfg = transformers.Qwen2MoeConfig(**common, intermediate_size=32,
            moe_intermediate_size=8, shared_expert_intermediate_size=16,
            num_experts=4, num_experts_per_tok=2)
    elif kind == "qwen3_moe":
        cfg = transformers.Qwen3MoeConfig(**common, intermediate_size=32,
            moe_intermediate_size=8, num_experts=4, num_experts_per_tok=2)
    else:
        cfg = transformers.MixtralConfig(**common, intermediate_size=16,
            num_local_experts=4, num_experts_per_tok=2, router_jitter_noise=0.)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    torch.manual_seed(23)
    return transformers.AutoModelForCausalLM.from_config(cfg).eval().to(
        device=os.environ.get("PMQ_TEST_DEVICE", "cpu"), dtype=dtype)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_native_gptq_captures_states_logits_and_routing_exact(monkeypatch, kind, dtype):
    torch.set_num_threads(1)
    model = make_model(kind, dtype)
    adapter = PmqMoeAdapter("deepseek_moe" if kind == "deepseek_v2" else kind)
    modules = adapter.collect_moe_modules(model)
    allocation = {(l, e): 1 + e % 3 for l, module in modules.items()
                  for e in range(adapter.num_experts(module))}
    ids = torch.stack((torch.arange(9), torch.arange(9, 18)))
    protocol = SimpleNamespace(model_path="unused", gptq_calibration_path="unused", data={
        "dataset": {"gptq_calibration": {}},
        "calibration": {"gptq": {"samples": 2, "seq_len": 9}},
        "quantization": {"group_size": 4, "percdamp": .01, "mse": True,
                         "mcmoe_rules": kind == "mixtral",
                         "standard_linear_bit": 4, "router_bit": 16},
        "pmq": {"shared_expert_bit": 3}})
    monkeypatch.setattr(q, "load_tokenizer", lambda _: None)
    monkeypatch.setattr(q, "calibration_inputs", lambda *a, **kw: ids.clone())
    replay_guard, routing_guard = q.can_replay_layers, q._can_cache_routing
    # Exercise production layout guards while permitting CPU equivalence checks.
    monkeypatch.setattr(q, "can_replay_layers", lambda m, xs, d: replay_guard(m, xs, "cuda:0"))
    monkeypatch.setattr(q, "_can_cache_routing", lambda m, bit, d: routing_guard(m, bit, "cuda:0"))
    original = q._capture_many_inputs
    captures = []
    full_model_reference = False

    def record(*args, **kwargs):
        if full_model_reference:
            kwargs["stop_after_capture"] = False
        values = original(*args, **kwargs)
        captures.append([[x.clone() for x in batches] for batches in values.values()])
        return values

    monkeypatch.setattr(q, "_capture_many_inputs", record)

    def run(replay, routes):
        nonlocal full_model_reference
        full_model_reference = not replay and not routes
        candidate = copy.deepcopy(model)
        captures.clear()
        q.apply_allocation(candidate, adapter, allocation, protocol=protocol, seed=0,
                           layer_replay=replay, routing_cache=routes)
        with torch.inference_mode():
            output = candidate(input_ids=ids.to(next(candidate.parameters()).device),
                               use_cache=False).logits
        return candidate.state_dict(), copy.deepcopy(captures), output

    expected_state, expected_captures, expected_logits = run(False, False)
    for replay, routes in ((False, True), (True, False), (True, True)):
        state, actual_captures, logits = run(replay, routes)
        assert state.keys() == expected_state.keys()
        for key in state:
            assert torch.equal(state[key], expected_state[key]), (kind, replay, routes, key)
        assert len(actual_captures) == len(expected_captures)
        for before, after in zip(expected_captures, actual_captures):
            assert len(before) == len(after)
            for xs, ys in zip(before, after):
                assert len(xs) == len(ys)
                assert all(torch.equal(x, y) for x, y in zip(xs, ys))
        assert torch.equal(logits, expected_logits)


@pytest.mark.parametrize("kind", KINDS)
def test_supported_native_guards_and_fallbacks(kind):
    model = make_model(kind, torch.float32)
    inputs = [torch.arange(9)]
    assert can_capture_native_layers(model)
    assert can_replay_layers(model, inputs, "cuda:0")
    assert q._can_cache_routing(model, 16, "cuda:0")
    assert not can_replay_layers(model, inputs, "cpu")
    assert not can_replay_layers(model, [torch.arange(9), torch.arange(8)], "cuda:0")
    assert not q._can_cache_routing(model, 4, "cuda:0")
    model.config.layer_types = ["full_attention", "sliding_attention", "full_attention"]
    assert not can_replay_layers(model, inputs, "cuda:0")
    model.config.layer_types = None
    model._hf_hook = object()
    assert not can_capture_native_layers(model)
    assert not q._can_cache_routing(model, 16, "cuda:0")
    del model._hf_hook
    model.train()
    assert not can_capture_native_layers(model)


def test_checkpoint_decoder_tuple_output_reuses_buffer_storage():
    model = make_model("deepseek_v2", torch.float32)
    replay = LayerInputs(model, [torch.arange(9)], next(model.parameters()).device)
    pointers = [x.data_ptr() for x in replay.buffers]
    layer = model.model.layers[0]
    with torch.inference_mode():
        expected = replay.forward(layer, 0).clone()
    original = layer.forward
    layer.forward = lambda *a, **kw: (original(*a, **kw), None)
    replay.advance(layer)
    assert [x.data_ptr() for x in replay.buffers] == pointers
    assert torch.equal(replay.buffers[0], expected)


def test_replay_rejects_hidden_dtype_change_before_overwriting_buffer():
    model = make_model("mixtral", torch.float32)
    replay = LayerInputs(model, [torch.arange(9)], next(model.parameters()).device)
    original = replay.buffers[0].clone()
    replay.forward = lambda *a: original.double()
    with pytest.raises(RuntimeError, match="output interface"):
        replay.advance(model.model.layers[0])
    assert torch.equal(replay.buffers[0], original)
