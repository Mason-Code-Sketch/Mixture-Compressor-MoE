"""Resident decoded weights and grouped expert GEMMs for PMQ evaluation."""

from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F


SUPPORTED_MODELS = frozenset({"deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"})
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


class CachedLinear(nn.Module):
    """Decode a saved PMQ projection once, without solving a new quantization grid."""

    @torch.no_grad()
    def __init__(self, packed):
        super().__init__()
        self.register_buffer("weight", packed.dequantize().detach(), persistent=False)
        self.register_buffer("bias", None if packed.bias is None else packed.bias.detach(), persistent=False)

    def forward(self, inputs):
        return F.linear(inputs, self.weight, self.bias)


def _cache_linears(module):
    for name, child in list(module.named_children()):
        if hasattr(child, "W_q") and callable(getattr(child, "dequantize", None)):
            setattr(module, name, CachedLinear(child))
        else:
            _cache_linears(child)


def _layout(experts):
    # HF 5 fused tensors, PMQ's packed fused container, and older ModuleLists.
    if hasattr(experts, "gate_up_proj"):
        groups = {name: getattr(experts, name) for name in ("gate_up_proj", "down_proj")}
        return groups, experts.act_fn
    entries = experts if isinstance(experts, nn.ModuleList) else experts.experts
    if hasattr(entries[0], "gate_up_proj"):
        names = ("gate_up_proj", "down_proj")
    elif hasattr(entries[0], "w1"):
        names = ("w1", "w3", "w2")
    else:
        names = ("gate_proj", "up_proj", "down_proj")
    groups = {name: [getattr(expert, name) for expert in entries] for name in names}
    activation = getattr(experts, "act_fn", None) or entries[0].act_fn
    return groups, activation


def _linear(inputs, weights, offsets):
    return torch._grouped_mm(inputs, weights.transpose(1, 2), offs=offsets)


class GroupedExperts(nn.Module):
    """Preserve native expert weighting and ascending-expert accumulation order."""

    @torch.no_grad()
    def __init__(self, experts, *, fp32_topk=False):
        super().__init__()
        self.fp32_topk = fp32_topk
        groups, self.act_fn = _layout(experts)
        for name, projections in groups.items():
            if isinstance(projections, torch.Tensor):
                weight = projections.detach()
            else:
                if any(proj.bias is not None for proj in projections):
                    raise ValueError("Grouped PMQ experts require bias-free projections")
                weight = torch.stack([proj.weight.detach() for proj in projections])
            if weight.ndim != 3 or weight.dtype not in SUPPORTED_DTYPES:
                raise ValueError("Grouped PMQ experts require FP16/BF16 three-dimensional weights")
            name = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}.get(name, name)
            self.register_buffer(name, weight, persistent=False)
        tensors = list(self.buffers())
        if len({t.dtype for t in tensors}) != 1 or len({t.device for t in tensors}) != 1:
            raise ValueError("Expert projections must share one dtype and device")
        if len({t.shape[0] for t in tensors}) != 1:
            raise ValueError("Expert projection counts differ")
        self.compute_dtype = tensors[0].dtype
        self.num_experts = tensors[0].shape[0]

    def __len__(self):
        return self.num_experts

    def forward(self, hidden, indices, weights):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Grouped PMQ experts are restricted to gradient-free evaluation")
        if hidden.dtype != self.compute_dtype:
            raise ValueError("Grouped PMQ input and expert compute dtypes differ")
        tokens, top_k = indices.shape
        ids = indices.reshape(-1) if self.fp32_topk else indices.T.contiguous().reshape(-1)
        permutation = ids.argsort() if self.fp32_topk else ids.argsort(stable=True)
        token_ids, slots = ((permutation // top_k, permutation % top_k) if self.fp32_topk else
                           (permutation % tokens, permutation // tokens))
        offsets = torch.bincount(ids, minlength=self.num_experts).cumsum(0, dtype=torch.int32)
        selected = hidden[token_ids]
        if hasattr(self, "gate_up_proj"):
            gate, up = _linear(selected, self.gate_up_proj, offsets).chunk(2, dim=-1)
        else:
            gate = _linear(selected, self.gate_proj, offsets)
            up = _linear(selected, self.up_proj, offsets)
        values = _linear(self.act_fn(gate) * up, self.down_proj, offsets)
        restored = torch.empty_like(values)
        if self.fp32_topk:
            restored[permutation] = values
            return (restored.view(tokens, top_k, -1).to(weights.dtype)
                    * weights.unsqueeze(-1)).sum(dim=1).to(hidden.dtype)
        restored[permutation] = (values * weights[token_ids, slots, None]).to(hidden.dtype)
        contributions = restored.view(top_k, tokens, -1).transpose(0, 1).to(hidden.dtype)
        order = indices.argsort(dim=1)
        contributions = contributions.gather(1, order.unsqueeze(-1).expand_as(contributions))
        output = torch.zeros_like(hidden)
        for slot in range(top_k):
            output.add_(contributions[:, slot])
        return output


def _legacy_forward(self, hidden_states):
    shape = hidden_states.shape
    hidden = hidden_states.reshape(-1, shape[-1])
    router_logits = self.gate(hidden)
    weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    weights, indices = torch.topk(weights, self.top_k, dim=-1)
    if getattr(self, "norm_topk_prob", True):
        weights = weights / weights.sum(dim=-1, keepdim=True)
    output = self.experts(hidden, indices, weights.to(hidden.dtype))
    if hasattr(self, "shared_expert"):
        shared = self.shared_expert(hidden)
        output = output + torch.sigmoid(self.shared_expert_gate(hidden)) * shared
    return output.reshape(shape), router_logits


def _deepseek_infer(self, hidden, indices, weights):
    return self.experts(hidden, indices, weights)


@torch.no_grad()
def prepare_evaluation_model(model, *, backend="auto"):
    """Install an evaluation-only backend; checkpoint serialization stays unchanged."""
    if backend not in ("auto", "pytorch", "grouped_cached"):
        raise ValueError("PMQ evaluation backend must be auto, pytorch or grouped_cached")
    if backend == "pytorch":
        return "pytorch"
    if getattr(model, "_pmq_evaluation_backend", None) == "grouped_cached":
        return "grouped_cached"
    kind = getattr(getattr(model, "config", None), "model_type", None)
    if kind not in SUPPORTED_MODELS:
        if backend == "auto":
            return "pytorch"
        raise ValueError(f"Unsupported PMQ grouped model: {kind}")
    dtype = next(model.parameters()).dtype
    if dtype not in SUPPORTED_DTYPES or not hasattr(torch, "_grouped_mm"):
        if backend == "auto":
            return "pytorch"
        raise ValueError("Grouped PMQ evaluation requires FP16/BF16 and torch._grouped_mm")
    blocks = [getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
              for layer in model.model.layers]
    blocks = [block for block in blocks if hasattr(block, "experts")]
    if not blocks:
        raise ValueError("No PMQ routed expert blocks found")
    for block in blocks:
        if kind == "deepseek_v2" and isinstance(block.experts, nn.ModuleList):
            if not hasattr(block, "moe_infer") or getattr(block, "ep_size", 1) != 1:
                raise ValueError("Legacy DeepSeek grouped evaluation requires single-rank moe_infer")
    model.eval()
    _cache_linears(model)
    for block in blocks:
        legacy = isinstance(block.experts, nn.ModuleList)
        original_deepseek = kind == "deepseek_v2" and legacy
        block.experts = GroupedExperts(block.experts, fp32_topk=original_deepseek).eval()
        if original_deepseek:
            block.moe_infer = MethodType(_deepseek_infer, block)
        elif legacy:
            attribute = "_old_forward" if hasattr(block, "_hf_hook") else "forward"
            setattr(block, attribute, MethodType(_legacy_forward, block))
    model._pmq_evaluation_backend = "grouped_cached"
    return "grouped_cached"
