"""Native MoE layout accessors used by standalone PMQ phases."""

from __future__ import annotations

import re

import torch


_FUSED_PATTERN = re.compile(r"model\.layers\.(\d+)\.mlp")
_MIXTRAL_PATTERN = re.compile(r"model\.layers\.(\d+)\.block_sparse_moe")


class PmqMoeAdapter:
    """Read and update expert tensors without changing native module forwards."""

    def __init__(self, architecture: str):
        if architecture not in {"deepseek_moe", "qwen2_moe", "qwen3_moe", "mixtral"}:
            raise ValueError(f"unsupported PMQ architecture: {architecture}")
        self.architecture = architecture

    def collect_moe_modules(self, model) -> dict[int, torch.nn.Module]:
        pattern = _MIXTRAL_PATTERN if self.architecture == "mixtral" else _FUSED_PATTERN
        modules = {}
        for name, module in model.named_modules():
            match = pattern.fullmatch(name)
            if match and hasattr(module, "experts") and hasattr(module, "gate"):
                modules[int(match.group(1))] = module
        if not modules:
            raise ValueError(f"no PMQ MoE modules found for {self.architecture}")
        return modules

    def num_experts(self, module) -> int:
        experts = module.experts
        if hasattr(experts, "gate_up_proj"):
            return int(experts.gate_up_proj.shape[0])
        return len(experts)

    def router_module(self, module):
        return module.gate

    def topk(self, module, model_config) -> int:
        for source, names in (
            (module, ("top_k", "num_experts_per_tok")),
            (getattr(module, "gate", None), ("top_k", "topk", "num_experts_per_tok")),
            (model_config, ("num_experts_per_tok", "num_experts_per_token")),
        ):
            if source is None:
                continue
            for name in names:
                value = getattr(source, name, None)
                if value is not None:
                    return int(value)
        raise ValueError("PMQ could not determine the native router top-k")

    def route_state(self, module, output, model_config):
        if isinstance(output, torch.Tensor):
            scores = torch.softmax(output, dim=-1, dtype=torch.float32)
            weights, indices = torch.topk(scores, self.topk(module, model_config), dim=-1)
            if bool(getattr(model_config, "norm_topk_prob", False)):
                weights = weights / weights.sum(dim=-1, keepdim=True)
            return indices, weights
        if isinstance(output, (tuple, list)) and len(output) >= 3:
            return output[2].detach(), output[1].detach()
        raise TypeError(f"unsupported native router output: {type(output)!r}")

    def expert_weights(self, module, expert: int) -> dict[str, torch.Tensor]:
        if self.architecture == "mixtral":
            item = module.experts[int(expert)]
            return {name: getattr(item, name).weight.detach() for name in ("w1", "w2", "w3")}
        gate_up = module.experts.gate_up_proj[int(expert)].detach()
        gate, up = gate_up.chunk(2, dim=0)
        return {
            "gate_proj": gate.contiguous(),
            "up_proj": up.contiguous(),
            "down_proj": module.experts.down_proj[int(expert)].detach().contiguous(),
        }

    def set_expert_weights(self, module, expert: int, weights: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            if self.architecture == "mixtral":
                item = module.experts[int(expert)]
                for name, value in weights.items():
                    target = getattr(item, name).weight
                    target.copy_(value.to(target.device, dtype=target.dtype))
                return
            target_gate_up = module.experts.gate_up_proj[int(expert)]
            target_down = module.experts.down_proj[int(expert)]
            target_gate_up.copy_(
                torch.cat((weights["gate_proj"], weights["up_proj"]), dim=0).to(
                    target_gate_up.device, dtype=target_gate_up.dtype
                )
            )
            target_down.copy_(weights["down_proj"].to(target_down.device, dtype=target_down.dtype))

    def shared_weights(self, module) -> dict[str, torch.Tensor] | None:
        shared = getattr(module, "shared_experts", None) or getattr(module, "shared_expert", None)
        if shared is None:
            return None
        return {
            name: getattr(shared, name).weight.detach()
            for name in ("gate_proj", "up_proj", "down_proj")
            if hasattr(shared, name)
        }

    def set_shared_weights(self, module, weights: dict[str, torch.Tensor]) -> None:
        shared = getattr(module, "shared_experts", None) or getattr(module, "shared_expert", None)
        if shared is None:
            return
        with torch.no_grad():
            for name, value in weights.items():
                target = getattr(shared, name).weight
                target.copy_(value.to(target.device, dtype=target.dtype))

    def shared_budget_units(self, model_config) -> int:
        if self.architecture == "deepseek_moe":
            return int(getattr(model_config, "n_shared_experts", 0) or 0)
        if self.architecture == "qwen2_moe":
            shared = int(getattr(model_config, "shared_expert_intermediate_size", 0) or 0)
            routed = int(getattr(model_config, "moe_intermediate_size", 0) or 0)
            if shared == 0:
                return 0
            if routed <= 0 or shared % routed:
                raise ValueError("Qwen shared expert width must be a routed-expert multiple")
            return shared // routed
        return 0
