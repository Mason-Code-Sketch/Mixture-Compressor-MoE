"""Bounded caches for PMQ's unchanged native single-expert output score."""
from __future__ import annotations

import types
import torch
import torch.nn.functional as F


class NativeExpertCache:
    """Replay unchanged contributions in native expert order, including FP rounding."""
    def __init__(self, module):
        self.module = module
        self.experts = module.experts
        self.sample = 0
        self.changed_expert = None
        self.entries = {}
        self.shared_entries = {}
        self.originals = []

    @staticmethod
    def supports(module):
        return (type(module.experts).__name__ in {'DeepseekV2Experts', 'Qwen2MoeExperts'}
                and hasattr(module.experts, 'gate_up_proj')
                and hasattr(module.experts, 'down_proj'))

    def _expert_forward(self, experts, hidden_states, top_k_index, top_k_weights):
        if self.changed_expert is None:
            mask = F.one_hot(top_k_index, num_classes=experts.num_experts).permute(2, 1, 0)
            hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            entries = []
            for index in hits:
                expert = int(index[0])
                positions, tokens = torch.where(mask[expert])
                current = hidden_states[tokens]
                gate, up = F.linear(current, experts.gate_up_proj[expert]).chunk(2, dim=-1)
                value = F.linear(experts.act_fn(gate) * up, experts.down_proj[expert])
                value = value * top_k_weights[tokens, positions, None]
                entries.append((expert, positions, tokens, value))
            self.entries[self.sample] = entries
        entries = self.entries[self.sample]
        result = torch.zeros_like(hidden_states)
        for expert, positions, tokens, cached in entries:
            value = cached
            if expert == self.changed_expert:
                current = hidden_states[tokens]
                gate, up = F.linear(current, experts.gate_up_proj[expert]).chunk(2, dim=-1)
                value = F.linear(experts.act_fn(gate) * up, experts.down_proj[expert])
                value = value * top_k_weights[tokens, positions, None]
            result.index_add_(0, tokens, value.to(result.dtype))
        return result

    def __enter__(self):
        def replace(obj, forward):
            # Preserve whether forward originally belonged to the instance or its class.
            self.originals.append((obj, obj.__dict__.get('forward'), 'forward' in obj.__dict__))
            obj.forward = types.MethodType(forward, obj)
        replace(self.experts, lambda obj, hidden_states, top_k_index, top_k_weights:
                self._expert_forward(obj, hidden_states, top_k_index, top_k_weights))
        for name in ('shared_expert', 'shared_experts'):
            shared = getattr(self.module, name, None)
            if shared is None:
                continue
            original = shared.forward
            def shared_forward(obj, *args, _original=original, _name=name, **kwargs):
                key = (_name, self.sample)
                if self.changed_expert is None:
                    self.shared_entries[key] = _original(*args, **kwargs)
                return self.shared_entries[key]
            replace(shared, shared_forward)
        return self

    def __exit__(self, *args):
        for obj, original, owned in reversed(self.originals):
            if owned:
                obj.forward = original
            else:
                del obj.forward
        self.entries.clear()
        self.shared_entries.clear()
