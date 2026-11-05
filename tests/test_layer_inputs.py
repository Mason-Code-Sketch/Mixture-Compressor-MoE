"""Sequential layer inputs must preserve the original GPTQ sample sequence."""
import copy
from types import SimpleNamespace

import pytest
import torch

from pmq import quantize as quant
from pmq.layer_inputs import LayerInputs, can_replay_layers
from pmq.adapters import PmqMoeAdapter
from test_quant_capture import make_model


def protocol():
    return SimpleNamespace(model_path='unused', gptq_calibration_path='unused', data={
        'dataset': {'gptq_calibration': {}},
        'calibration': {'gptq': {'samples': 2, 'seq_len': 9}},
        'quantization': {'group_size': 4, 'percdamp': .01, 'mse': True,
                         'standard_linear_bit': 4, 'router_bit': 16},
        'pmq': {'shared_expert_bit': 3}})


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_sequential_gptq_every_capture_packed_state_and_logits_exact(monkeypatch, dtype):
    model = make_model(dtype); reference = copy.deepcopy(model)
    ids = torch.stack([torch.arange(9), torch.arange(9, 18)])
    monkeypatch.setattr(quant, 'load_tokenizer', lambda _: None)
    monkeypatch.setattr(quant, 'calibration_inputs', lambda *a, **kw: ids.clone())
    monkeypatch.setattr(quant, 'can_replay_layers', lambda *a: True)
    adapter = PmqMoeAdapter('qwen3_moe')
    allocation = {(l, e): 1 + e % 3 for l in range(3) for e in range(4)}
    captures = []
    original = quant._capture_many_inputs
    def recorded(*args, **kwargs):
        values = original(*args, **kwargs)
        captures.append([[x.clone() for x in batches] for batches in values.values()])
        return values
    monkeypatch.setattr(quant, '_capture_many_inputs', recorded)
    quant.apply_allocation(reference, adapter, allocation, protocol=protocol(), seed=0, layer_replay=False, routing_cache=False)
    expected = captures[:];captures.clear()
    quant.apply_allocation(model, adapter, allocation, protocol=protocol(), seed=0)
    assert len(expected) == len(captures) == 6
    for old, new in zip(expected, captures):
        for before, after in zip(old, new):
            for a, b in zip(before, after):assert torch.equal(a, b)
    assert reference.state_dict().keys() == model.state_dict().keys()
    for name, before in reference.state_dict().items():
        assert torch.equal(before, model.state_dict()[name]), name
    with torch.inference_mode():
        device = next(model.parameters()).device
        assert torch.equal(reference(input_ids=ids.to(device), use_cache=False).logits,
                           model(input_ids=ids.to(device), use_cache=False).logits)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_native_decoder_outputs_and_buffer_storage_reused(dtype):
    model = make_model(dtype);ids = [torch.arange(9), torch.arange(9,18)]
    device = next(model.parameters()).device
    cache = LayerInputs(model, ids, device)
    pointers = [b.data_ptr() for b in cache.buffers]
    for i, layer in enumerate(model.model.layers):
        expected=[]
        handle=layer.register_forward_hook(lambda m,a,o: expected.append(o.detach().cpu().clone()))
        with torch.inference_mode():
            for sample in ids:model(input_ids=sample.unsqueeze(0).to(device),use_cache=False)
        handle.remove()
        cache.advance(layer)
        assert pointers == [b.data_ptr() for b in cache.buffers]
        assert all(torch.equal(x,y) for x,y in zip(expected,cache.buffers))
    assert all(not layer._forward_pre_hooks for layer in model.model.layers)


def test_single_layer_replay_does_not_execute_other_layers():
    model = make_model(torch.float32);ids=[torch.arange(9)];device=next(model.parameters()).device
    cache=LayerInputs(model,ids,device)
    def fail(*args,**kwargs):raise AssertionError('Unneeded decoder executed')
    handles=[l.register_forward_pre_hook(fail) for l in model.model.layers[1:]]
    try:
        target=model.model.layers[0].mlp
        quant._capture_inputs(model,target,ids,device,forward_sample=lambda i:cache.forward(model.model.layers[0],i))
    finally:
        for handle in handles:handle.remove()


def test_capture_cpu_aliases_share_exact_qkv_inputs():
    model=make_model(torch.float32);ids=[torch.arange(9)];device=next(model.parameters()).device
    attn=model.model.layers[0].self_attn;targets=[attn.q_proj,attn.k_proj,attn.v_proj]
    values=quant._capture_many_inputs(model,targets,ids,device,deduplicate_inputs=True)
    assert values[targets[0]][0] is values[targets[1]][0] is values[targets[2]][0]


def test_replay_guards_offload_training_variable_length_and_cpu():
    model=make_model(torch.float32);ids=[torch.arange(9)]
    assert not can_replay_layers(model,ids,'cpu')
    assert can_replay_layers(model,ids,'cuda:0')
    model.train();assert not can_replay_layers(model,ids,'cuda:0');model.eval()
    model._hf_hook=object();assert not can_replay_layers(model,ids,'cuda:0');del model._hf_hook
    assert not can_replay_layers(model,[torch.arange(9),torch.arange(8)],'cuda:0')


def test_initialization_failure_removes_hook():
    model=make_model(torch.float32)
    with pytest.raises((IndexError,RuntimeError)):
        LayerInputs(model,[torch.tensor([1000])],next(model.parameters()).device)
    assert not model.model.layers[0]._forward_pre_hooks
