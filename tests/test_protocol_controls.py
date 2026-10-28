"""Runtime coverage of PMQ flags and standalone configuration forwarding."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pmq.config import resolve_quantization_settings, validate_native_cli_args
from utils.quantizer_moe import Quantizer


def test_perchannel_and_weight_flags_control_the_parameter_axes():
    weights = torch.tensor([[-1., 2., 3.], [-8., 4., 2.]])
    row = Quantizer(); row.configure(2, perchannel=True, sym=False, mse=False)
    row.find_params(weights, weight=True)
    global_grid = Quantizer(); global_grid.configure(2, perchannel=False, sym=False, mse=False)
    global_grid.find_params(weights, weight=True)
    activation = Quantizer(); activation.configure(2, perchannel=True, sym=False, mse=False)
    activation.find_params(weights, weight=False)
    assert row.scale.shape == global_grid.scale.shape == (2, 1)
    assert row.scale[0] != row.scale[1]
    assert global_grid.scale[0] == global_grid.scale[1]
    assert activation.scale.shape == (1, 3)
    assert activation.quantize(weights).shape == weights.shape


def test_symmetry_is_respected_during_mse_search():
    weights = torch.tensor([[-3., -2., 1., 8.]])
    for symmetric in (False, True):
        solver = Quantizer()
        solver.configure(2, perchannel=True, sym=symmetric)
        solver.find_params(weights, weight=True)
        if symmetric:
            assert solver.zero.item() == 2
        else:
            assert solver.zero.item() != round(solver.zero.item())


def test_zero_binary_channel_does_not_disable_other_channels():
    weights = torch.tensor([[0., 0.], [-1., 3.]])
    solver = Quantizer(); solver.configure(1, perchannel=True, sym=False)
    solver.find_params(weights, weight=True)
    assert solver.ready()
    assert torch.equal(solver.quantize(weights), torch.tensor([[0., 0.], [-2., 2.]]))


def test_native_flash_attention_alias_sets_model_loading_option():
    args = SimpleNamespace(load_quantized=False, multigpu=False, tasks="",
                           num_fewshot=0, batch_size=1, r=7,
                           use_flash_attention_2=True, attn_implementation="eager")
    validate_native_cli_args(args)
    assert args.attn_implementation == "flash_attention_2"
    args.attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="Conflicting attention"):
        validate_native_cli_args(args)


@pytest.mark.parametrize("setting", [{"mse": "false"}, {"candidate_quantizer": "unknown"}, {"mse_typo": False}])
def test_standalone_rejects_unknown_or_invalid_controls(setting):
    with pytest.raises(ValueError):
        resolve_quantization_settings(setting)


@pytest.mark.parametrize("field,value", [("load_quantized", True), ("multigpu", True),
    ("tasks", "piqa"), ("num_fewshot", 5), ("batch_size", 4), ("r", 4)])
def test_native_cli_rejects_unimplemented_requests(field, value):
    args = SimpleNamespace(load_quantized=False, multigpu=False, tasks="", num_fewshot=0, batch_size=1)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_native_cli_args(args)


@pytest.mark.parametrize("kind", ["deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"])
@pytest.mark.parametrize("mse", [False, True])
def test_standalone_mse_reaches_every_quantized_projection(kind, mse):
    from test_pmq_grouped import tiny_model
    from pmq.adapters import PmqMoeAdapter
    from pmq.quantize import apply_allocation
    from utils.quantizer_moe import normal_quantize
    model = tiny_model(kind, dtype=torch.float16)
    adapter = PmqMoeAdapter("deepseek_moe" if kind == "deepseek_v2" else kind)
    ids = torch.randint(0, 32, (2, 8))
    allocation = {(layer, expert): 2 for layer, module in adapter.collect_moe_modules(model).items()
                  for expert in range(adapter.num_experts(module))}
    protocol = SimpleNamespace(model_path=Path("unused"), gptq_calibration_path=Path("unused"),
        data={"dataset": {"gptq_calibration": {}}, "calibration": {"gptq": {"samples": 2, "seq_len": 8}},
              "quantization": {"group_size": 4, "percdamp": .01, "mse": mse,
                               "standard_linear_bit": 4, "router_bit": 16},
              "pmq": {"shared_expert_bit": 3}})
    seen = []
    original = Quantizer.configure
    def configure(self, *args, **kwargs):
        seen.append(kwargs["mse"])
        return original(self, *args, **kwargs)
    with patch("pmq.quantize.load_tokenizer"), patch("pmq.quantize.calibration_inputs", return_value=ids), \
         patch.object(Quantizer, "configure", configure), \
         patch("utils.quantizer_moe.normal_quantize", wraps=normal_quantize) as search:
        apply_allocation(model, adapter, allocation, protocol=protocol, seed=0)
    assert seen and all(flag is mse for flag in seen)
    assert bool(search.call_count) is mse
