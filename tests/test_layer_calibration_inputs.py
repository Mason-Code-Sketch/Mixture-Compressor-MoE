"""Compare GPTQ calibration batches with the untouched native first decoder."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn.functional import linear

from pmq import quantize
from pmq.adapters import PmqMoeAdapter
from test_pmq_grouped import tiny_model


@pytest.mark.parametrize("kind", ["deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"])
@torch.inference_mode()
def test_expert_hessians_receive_original_layer_inputs(kind, monkeypatch):
    model = tiny_model(kind)
    adapter = PmqMoeAdapter("deepseek_moe" if kind == "deepseek_v2" else kind)
    modules = adapter.collect_moe_modules(model)
    layer, block = next(iter(sorted(modules.items())))
    ids = torch.arange(16).reshape(2, 8)
    hidden = []
    hook = block.register_forward_pre_hook(lambda _, args: hidden.append(args[0].clone()))
    try:
        for sample in ids:
            model(input_ids=sample[None, :], use_cache=False)
    finally:
        hook.remove()
    names = {id(module): name for name, module in model.named_modules()}
    expected = {}
    for e in range(adapter.num_experts(block)):
        prefix = f"{names[id(block)]}.experts.{e}"
        for h in hidden:
            indices, _ = adapter.route_state(block, adapter.router_module(block)(h), model.config)
            mask = (indices.reshape(-1, indices.shape[-1]) == e).any(dim=-1)
            routed = h.flatten(0, 1)[mask].unsqueeze(0)
            if routed.numel() == 0:
                continue
            if isinstance(block.experts, torch.nn.ModuleList):
                expert = block.experts[e]
                gate, up, down = ("w1", "w3", "w2") if kind == "mixtral" else ("gate_proj", "up_proj", "down_proj")
                activation = expert.act_fn(getattr(expert, gate)(routed)) * getattr(expert, up)(routed)
            else:
                gate, up = linear(routed, block.experts.gate_up_proj[e]).chunk(2, dim=-1)
                activation = block.experts.act_fn(gate) * up
                down = "down_proj"
            expected.setdefault(f"{prefix}.{down}", []).append(activation)
    shared = getattr(block, "shared_expert", None) or getattr(block, "shared_experts", None)
    if shared is not None:
        expected[f"{names[id(shared)]}.down_proj"] = [
            shared.act_fn(shared.gate_proj(h)) * shared.up_proj(h) for h in hidden]
    assert expected
    original = quantize._quantize_matrix
    checked = set()
    def check(weight, batches, **kwargs):
        name = kwargs["name"]
        if name in expected:
            assert len(batches) == len(expected[name])
            for actual, reference in zip(batches, expected[name]):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            checked.add(name)
        return original(weight, batches, **kwargs)

    monkeypatch.setattr(quantize, "_quantize_matrix", check)
    monkeypatch.setattr(quantize, "load_tokenizer", lambda _: None)
    monkeypatch.setattr(quantize, "calibration_inputs", lambda *a, **kw: ids)
    protocol = SimpleNamespace(model_path="", gptq_calibration_path="", data={
        "dataset": {"gptq_calibration": {}}, "calibration": {"gptq": {"samples": 2, "seq_len": 8}},
        "quantization": {"group_size": 8, "mcmoe_rules": kind == "mixtral"},
        "pmq": {"shared_expert_bit": 2}})
    allocation = {(i, e): 2 for i, module in modules.items() for e in range(adapter.num_experts(module))}
    quantize.apply_allocation(model, adapter, allocation, protocol=protocol, seed=0)
    assert checked == set(expected)
