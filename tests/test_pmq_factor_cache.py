import unittest
import torch
from transformers import AutoModelForCausalLM, DeepseekV2Config, Qwen2MoeConfig
from pmq.adapters import PmqMoeAdapter
from pmq.factors import _candidate_loss_for_layer, _candidate_loss_for_layer_reference
from pmq.factor_cache import NativeExpertCache


def make_module(architecture, dtype=torch.float32):
    common=dict(vocab_size=32,hidden_size=16,intermediate_size=32,moe_intermediate_size=8,
                num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,num_experts_per_tok=2,max_position_embeddings=64)
    if architecture=='deepseek_moe':
        config=DeepseekV2Config(**common,n_routed_experts=4,n_shared_experts=1,
            q_lora_rank=8,kv_lora_rank=4,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,first_k_dense_replace=1)
    else:
        config=Qwen2MoeConfig(**common,num_experts=4,shared_expert_intermediate_size=16)
    config._attn_implementation='eager';config._experts_implementation='eager'
    torch.manual_seed(19)
    model=AutoModelForCausalLM.from_config(config).eval().to(dtype)
    adapter=PmqMoeAdapter(architecture)
    return next(iter(adapter.collect_moe_modules(model).values())),adapter


class FactorCacheTest(unittest.TestCase):
    def test_native_losses_and_restoration(self):
        for architecture in ['deepseek_moe','qwen2_moe']:
            for dtype in [torch.float32,torch.float16]:
                with self.subTest(architecture=architecture,dtype=dtype):
                    module,adapter=make_module(architecture,dtype)
                    inputs=[torch.randn(1,9,16,dtype=dtype) for _ in range(10)]
                    with torch.inference_mode():outputs=[module(x).clone() for x in inputs]
                    weights={k:v.clone() for k,v in module.state_dict().items()}
                    kwargs=dict(module=module,adapter=adapter,inputs=inputs,outputs=outputs,bits=(1,2,3),group_size=4)
                    expected=_candidate_loss_for_layer_reference(**kwargs)
                    actual=_candidate_loss_for_layer(**kwargs)
                    self.assertEqual(actual,expected)
                    for k,v in module.state_dict().items():self.assertTrue(torch.equal(v,weights[k]),k)
                    self.assertNotIn('forward',module.experts.__dict__)
                    for name in ['shared_expert','shared_experts']:
                        obj=getattr(module,name,None)
                        if obj is not None:self.assertNotIn('forward',obj.__dict__)
                    with torch.inference_mode():
                        for x,y in zip(inputs,outputs):self.assertTrue(torch.equal(module(x),y))

    def test_forward_restored_on_exception(self):
        module,_=make_module('qwen2_moe')
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with NativeExpertCache(module):raise RuntimeError('injected')
        self.assertNotIn('forward',module.experts.__dict__)
        self.assertNotIn('forward',module.shared_expert.__dict__)


if __name__=='__main__':unittest.main()
