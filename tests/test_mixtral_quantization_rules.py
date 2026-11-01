"""Mixtral GPTQ rules preserve precision, packing and other-model defaults."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from pmq.gptq import TensorGPTQ, _mcmoe_params, _quantize_uniform
from pmq.hessian import inverse_hessian_factor
from pmq.packed import PackedLinear
from utils.quantizer_moe import Quantizer


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_low_bit_block_and_cross_block_compensation(bits, dtype):
    torch.manual_seed(17)
    w = torch.randn(4, 16).to(dtype)
    solver = TensorGPTQ(w, bits=bits, mcmoe_rules=True)
    solver.add_batch(torch.randn(2, 20, 16).to(dtype))
    factor = inverse_hessian_factor(solver.hessian.clone(), .01)
    remaining = w.float().clone()
    expected = torch.empty_like(remaining)
    for start in range(0, 16, 4):
        scale, zero = _mcmoe_params(remaining[:, start:start+4], bits, mcmoe_rules=True)
        if bits <= 3:
            # With no in-block update, all columns use the unchanged block.
            block = remaining[:, start:start+4].clone()
            quant = _quantize_uniform(block, bits, scale, zero)
            error = (block - quant) / factor.diag()[start:start+4]
            expected[:, start:start+4] = quant
            remaining[:, start+4:] -= error @ factor[start:start+4, start+4:]
        else:
            # Attention retains the full sequential compensation.
            for col in range(start, start+4):
                value = remaining[:, col].clone()
                quant = _quantize_uniform(value, bits, scale, zero)
                expected[:, col] = quant
                error = (value - quant) / factor[col, col]
                remaining[:, col:] -= error[:, None] @ factor[col:col+1, col:]
    actual = solver.quantize(group_size=4, percdamp=.01)
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
    assert torch.equal(PackedLinear(solver.packed_state).dequantize(), actual)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_range_and_unrounded_zero(bits):
    w = torch.tensor([[-3., -2., 1., 8.], [-2., .1, .3, .5]])
    solver = Quantizer(mcmoe_rules=True)
    solver.configure(bits, perchannel=True, sym=False)
    from utils.quantizer_moe import normal_quantize
    with patch("utils.quantizer_moe.normal_quantize", wraps=normal_quantize) as call:
        solver.find_params(w, weight=True)
    assert call.call_count == 101
    scales = torch.stack([c.args[1].flatten() for c in call.call_args_list])
    base = ((w.half().amax(1) - w.half().amin(1)) / (2**bits-1))
    torch.testing.assert_close((scales/base).max(0).values, torch.full((2,), 1.1).half(), rtol=.003, atol=.003)
    torch.testing.assert_close((scales/base).min(0).values, torch.full((2,), .9).half(), rtol=.003, atol=.003)
    assert torch.any(solver.zero != solver.zero.round())


def test_only_mixtral_config_selects_restored_rules():
    from pmq.config import resolve_quantization_settings
    for path in (Path(__file__).parents[1] / "configs").glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        if "quantization" not in data:
            continue
        settings = resolve_quantization_settings(data["quantization"])
        assert settings["mcmoe_rules"] == (data["model"]["architecture"] == "mixtral")


def test_mixtral_dispatch_reaches_experts_and_attention():
    from types import SimpleNamespace
    from test_pmq_grouped import tiny_model
    from pmq.adapters import PmqMoeAdapter
    from pmq.quantize import apply_allocation
    model = tiny_model("mixtral", dtype=torch.float16)
    adapter = PmqMoeAdapter("mixtral")
    allocation = {(layer, expert): 1 + expert % 3 for layer, module in adapter.collect_moe_modules(model).items()
                  for expert in range(adapter.num_experts(module))}
    protocol = SimpleNamespace(model_path=Path("unused"), gptq_calibration_path=Path("unused"),
        data={"dataset": {"gptq_calibration": {}}, "calibration": {"gptq": {"samples": 2, "seq_len": 8}},
              "quantization": {"group_size": 4, "mcmoe_rules": True}, "pmq": {"shared_expert_bit": 3}})
    seen = []
    original = TensorGPTQ.__init__
    def init(self, *args, **kwargs):
        seen.append((kwargs["bits"], kwargs["mcmoe_rules"]))
        original(self, *args, **kwargs)
    with patch("pmq.quantize.load_tokenizer"), patch("pmq.quantize.calibration_inputs", return_value=torch.randint(0,32,(2,8))), \
         patch.object(TensorGPTQ, "__init__", init):
        apply_allocation(model, adapter, allocation, protocol=protocol, seed=0)
    assert set(bit for bit, _ in seen) == {1, 2, 3, 4}
    assert all(flag for _, flag in seen)
