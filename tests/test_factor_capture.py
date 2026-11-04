"""Native Qwen3 capture must remain exact while skipping unused suffixes."""
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from pmq.factors import _capture_layer_io


def model_for_capture(dtype):
    torch.manual_seed(7)
    config = Qwen3MoeConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=8, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return AutoModelForCausalLM.from_config(config).eval().to(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("target", [0, 1, 2])
def test_capture_is_exact_and_skips_later_layers(dtype, target):
    model = model_for_capture(dtype)
    inputs = [torch.arange(9) % 32, torch.arange(9, 18) % 32]
    module = model.model.layers[target].mlp
    expected = _capture_layer_io(model, module, inputs, torch.device("cpu"), stop_after_capture=False)
    complete_calls = Mock(return_value=None)
    handle = model.lm_head.register_forward_hook(complete_calls)
    later_calls = Mock(return_value=None)
    later = None if target == 2 else model.model.layers[target + 1].register_forward_pre_hook(later_calls)
    try:
        actual = _capture_layer_io(model, module, inputs, torch.device("cpu"))
        complete_calls.assert_not_called()
        later_calls.assert_not_called()
        for expected_list, actual_list in zip(expected, actual):
            assert len(actual_list) == len(inputs)
            for a, b in zip(expected_list, actual_list):
                assert torch.equal(a, b)
        _capture_layer_io(model, module, inputs, torch.device("cpu"), stop_after_capture=False)
        assert complete_calls.call_count == len(inputs)
    finally:
        handle.remove()
        if later is not None:
            later.remove()
    assert not module._forward_hooks


def test_capture_keeps_accelerate_cleanup_path():
    model = model_for_capture(torch.float32)
    model.model.layers[0]._hf_hook = object()
    complete_calls = Mock(return_value=None)
    handle = model.lm_head.register_forward_hook(complete_calls)
    try:
        _capture_layer_io(model, model.model.layers[0].mlp, [torch.arange(9)], torch.device("cpu"))
        complete_calls.assert_called_once()
    finally:
        handle.remove()
        del model.model.layers[0]._hf_hook


def test_capture_propagates_real_errors_and_removes_hook():
    model = model_for_capture(torch.float32)
    module = model.model.layers[0].mlp
    def fail(*args):
        raise RuntimeError("capture test failure")
    handle = module.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="capture test failure"):
            _capture_layer_io(model, module, [torch.arange(9)], torch.device("cpu"))
    finally:
        handle.remove()
    assert not module._forward_hooks
