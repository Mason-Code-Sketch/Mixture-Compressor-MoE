"""Capture changes must preserve sequential GPTQ and every packed tensor."""
import copy
import os
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from pmq import quantize as quant
from pmq.adapters import PmqMoeAdapter


def make_model(dtype):
    torch.manual_seed(23)
    cfg = Qwen3MoeConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=8, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    return AutoModelForCausalLM.from_config(cfg).eval().to(device=os.environ.get("PMQ_TEST_DEVICE", "cpu"), dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("layer", [0, 1, 2])
@pytest.mark.parametrize("kind", ["single", "many"])
def test_native_inputs_exact_and_unused_forward_skipped(dtype, layer, kind):
    model = make_model(dtype)
    inputs = [torch.arange(9), torch.arange(9, 18)]
    module = model.model.layers[layer].mlp
    targets = list(reversed([x for _, x in quant._standard_linears(model.model.layers[layer], module)])) if kind == "many" else [module]
    def capture(stop):
        if kind == "single":
            return {module: quant._capture_inputs(model, module, inputs, next(model.parameters()).device, stop_after_capture=stop)}
        return quant._capture_many_inputs(model, targets, inputs, next(model.parameters()).device, stop_after_capture=stop)
    expected = capture(False)
    head = Mock(return_value=None)
    handle = model.lm_head.register_forward_hook(head)
    try:
        actual = capture(True)
        head.assert_not_called()
        assert actual.keys() == expected.keys()
        for target in expected:
            assert len(actual[target]) == len(inputs)
            for a, b in zip(actual[target], expected[target]):
                assert torch.equal(a, b)
        capture(False)
        assert head.call_count == len(inputs)
    finally:
        handle.remove()
    assert all(not target._forward_pre_hooks for target in targets)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_complete_gptq_packed_states_and_logits_unchanged(monkeypatch, dtype):
    model = make_model(dtype)
    reference = copy.deepcopy(model)
    ids = torch.stack([torch.arange(9), torch.arange(9, 18)])
    monkeypatch.setattr(quant, "load_tokenizer", lambda _: None)
    monkeypatch.setattr(quant, "calibration_inputs", lambda *args, **kwargs: ids.clone())
    protocol = SimpleNamespace(model_path="unused", gptq_calibration_path="unused", data={
        "dataset": {"gptq_calibration": {}},
        "calibration": {"gptq": {"samples": 2, "seq_len": 9}},
        "quantization": {"group_size": 4, "percdamp": .01, "mse": True,
            "standard_linear_bit": 4, "router_bit": 16},
        "pmq": {"shared_expert_bit": 3},
    })
    adapter = PmqMoeAdapter("qwen3_moe")
    allocation = {(layer, expert): 1 + expert % 3 for layer in range(3) for expert in range(4)}
    single, many = quant._capture_inputs, quant._capture_many_inputs
    with monkeypatch.context() as m:
        m.setattr(quant, "_capture_inputs", partial(single, stop_after_capture=False))
        m.setattr(quant, "_capture_many_inputs", partial(many, stop_after_capture=False))
        quant.apply_allocation(reference, adapter, allocation, protocol=protocol, seed=0)
    quant.apply_allocation(model, adapter, allocation, protocol=protocol, seed=0)
    expected, actual = reference.state_dict(), model.state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    with torch.inference_mode():
        assert torch.equal(reference(input_ids=ids.to(next(reference.parameters()).device), use_cache=False).logits,
                           model(input_ids=ids.to(next(model.parameters()).device), use_cache=False).logits)


def test_offload_hook_requires_full_forward():
    model = make_model(torch.float32)
    model._hf_hook = object()
    calls = Mock(return_value=None)
    handle = model.lm_head.register_forward_hook(calls)
    try:
        quant._capture_inputs(model, model.model.layers[0].mlp, [torch.arange(9)], next(model.parameters()).device)
        calls.assert_called_once()
    finally:
        handle.remove()
        del model._hf_hook


def test_real_exception_propagates_and_hooks_are_removed():
    model = make_model(torch.float32)
    target = model.model.layers[0].mlp
    def fail(*args):
        raise RuntimeError("injected real failure")
    handle = model.model.layers[0].self_attn.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected real failure"):
            quant._capture_inputs(model, target, [torch.arange(9)], next(model.parameters()).device)
    finally:
        handle.remove()
    assert not target._forward_pre_hooks
