"""Routing reuse preserves samples, selected token order and complete GPTQ."""
import copy
from unittest.mock import Mock

import pytest
import torch

from pmq import quantize as q
from pmq.adapters import PmqMoeAdapter
from test_quant_capture import make_model
from test_layer_inputs import protocol


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
def test_all_expert_inputs_exact_and_router_calls_reduced(dtype):
    model=make_model(dtype);module=model.model.layers[0].mlp;adapter=PmqMoeAdapter('qwen3_moe')
    hidden=[torch.randn(1,9,16,dtype=dtype) for _ in range(2)]
    calls=Mock(return_value=None);handle=module.gate.register_forward_hook(calls)
    try:
        with torch.inference_mode():
            expected=[q._routed_inputs(module,adapter,hidden,model.config,e) for e in range(4)]
            assert calls.call_count==8;calls.reset_mock()
            cached=q._cache_routing_indices(module,adapter,hidden,model.config)
            actual=[q._routed_inputs(module,adapter,hidden,model.config,e,cached) for e in range(4)]
            assert calls.call_count==2
    finally:handle.remove()
    for old,new in zip(expected,actual):
        assert len(old)==len(new)
        for a,b in zip(old,new):assert torch.equal(a,b)
    assert all(x.device.type=='cpu' for x in cached)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
def test_full_gptq_packed_states_and_logits_exact(monkeypatch,dtype):
    model=make_model(dtype);reference=copy.deepcopy(model);ids=torch.stack([torch.arange(9),torch.arange(9,18)])
    monkeypatch.setattr(q,'load_tokenizer',lambda _:None)
    monkeypatch.setattr(q,'calibration_inputs',lambda *a,**kw:ids.clone())
    monkeypatch.setattr(q,'_can_cache_routing',lambda *a:True)
    monkeypatch.setattr(q,'can_replay_layers',lambda *a:True)
    adapter=PmqMoeAdapter('qwen3_moe');allocation={(l,e):1+e%3 for l in range(3) for e in range(4)}
    q.apply_allocation(reference,adapter,allocation,protocol=protocol(),seed=0,routing_cache=False)
    q.apply_allocation(model,adapter,allocation,protocol=protocol(),seed=0,routing_cache=True)
    a,b=reference.state_dict(),model.state_dict();assert a.keys()==b.keys()
    for key in a:assert torch.equal(a[key],b[key]),key
    with torch.inference_mode():
        device=next(model.parameters()).device
        assert torch.equal(reference(input_ids=ids.to(device),use_cache=False).logits,
                           model(input_ids=ids.to(device),use_cache=False).logits)


def test_empty_batches_and_unselected_experts():
    model=make_model(torch.float32);module=model.model.layers[0].mlp;adapter=PmqMoeAdapter('qwen3_moe')
    assert q._cache_routing_indices(module,adapter,[],model.config)==[]
    assert q._routed_inputs(module,adapter,[],model.config,0,[])==[]
    hidden=[torch.randn(1,4,16)];indices=[torch.zeros(4,2,dtype=torch.int64)]
    assert q._routed_inputs(module,adapter,hidden,model.config,3,indices)==[]
    values=q._routed_inputs(module,adapter,hidden,model.config,0,indices)
    assert len(values)==1 and torch.equal(values[0],hidden[0])


def test_cached_shape_mismatch_rejected():
    model=make_model(torch.float32);module=model.model.layers[0].mlp;adapter=PmqMoeAdapter('qwen3_moe');hidden=[torch.randn(1,4,16)]
    with pytest.raises(ValueError,match='sample count'):q._routed_inputs(module,adapter,hidden,model.config,0,[])
    with pytest.raises(ValueError,match='shape/device'):q._routed_inputs(module,adapter,hidden,model.config,0,[torch.zeros(3,2)])


def test_cache_guard_requires_native_eval_CUDA_and_unchanged_router():
    model=make_model(torch.float32)
    assert q._can_cache_routing(model,16,'cuda:0')
    assert not q._can_cache_routing(model,4,'cuda:0')
    assert not q._can_cache_routing(model,16,'cpu')
    model.train();assert not q._can_cache_routing(model,16,'cuda:0');model.eval()
    model._hf_hook=object();assert not q._can_cache_routing(model,16,'cuda:0');del model._hf_hook
    model.config.model_type='mixtral';assert not q._can_cache_routing(model,16,'cuda:0')
