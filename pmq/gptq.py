"""Tensor-based GPTQ used only by PMQ final-plan materialization."""

from __future__ import annotations

import math

import torch


def _quantize_uniform(
    values: torch.Tensor,
    bits: int,
    scale: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    if values.ndim == 1:
        scale = scale.reshape(-1)
        zero = zero.reshape(-1)
    else:
        scale = scale.reshape(-1, 1)
        zero = zero.reshape(-1, 1)
    if bits == 1:
        levels = torch.where(values >= 0, torch.ones_like(values), torch.zeros_like(values))
    else:
        levels = (values / scale + zero).round().clamp_(0, 2**bits - 1)
    return (levels - zero) * scale


def _mcmoe_params(
    weight: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the original PMQ FP16 parameter solver with the existing MSE search."""
    from utils.quantizer_moe import Quantizer

    quantizer = Quantizer()
    quantizer.configure(bits, perchannel=True, sym=False, mse=True, pack=True)
    quantizer.find_params(weight, weight=True)
    return quantizer.scale.flatten(), quantizer.zero.flatten()


class TensorGPTQ:
    """Apply the repository's GPTQ update to one 2-D weight tensor."""

    def __init__(self, weight: torch.Tensor, *, bits: int):
        if weight.ndim != 2:
            raise ValueError(f"GPTQ requires a matrix weight, got {tuple(weight.shape)}")
        self.weight = weight.detach()
        self.bits = int(bits)
        self.columns = int(weight.shape[1])
        self.hessian = torch.zeros(
            (self.columns, self.columns), device=weight.device, dtype=torch.float32
        )
        self.samples = 0
        self.packed_state = None

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor) -> None:
        """Accumulate the same token-input Hessian used by GPTQ hooks."""
        if inputs.shape[-1] != self.columns:
            raise ValueError(
                f"input width {inputs.shape[-1]} does not match weight width {self.columns}"
            )
        batch_size = int(inputs.shape[0]) if inputs.ndim >= 2 else 1
        rows = inputs.reshape(-1, self.columns).transpose(0, 1).float()
        next_samples = self.samples + batch_size
        self.hessian.mul_(self.samples / next_samples)
        self.samples = next_samples
        rows.mul_(math.sqrt(2.0 / self.samples))
        self.hessian.add_(rows.matmul(rows.transpose(0, 1)))

    @torch.no_grad()
    def quantize(self, *, group_size: int, percdamp: float) -> torch.Tensor:
        """Quantize the tensor with the established PMQ GPTQ arithmetic."""
        weight = self.weight.float().clone()
        hessian = self.hessian
        self.hessian = None
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        diagonal = torch.arange(self.columns, device=weight.device)
        hessian[diagonal, diagonal] += percdamp * torch.mean(torch.diag(hessian))
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hessian = torch.linalg.cholesky(hessian, upper=True)
        denom_floor = torch.clamp(
            torch.diag(hessian).abs().mean() * 1e-6,
            min=1e-8,
        )

        result = torch.zeros_like(weight)
        codes = torch.zeros_like(weight, dtype=torch.uint8)
        group_scales, group_zeros = [], []
        block_size = int(group_size)
        for start in range(0, self.columns, block_size):
            end = min(start + block_size, self.columns)
            width = end - start
            source = weight[:, start:end].clone()
            quantized = torch.zeros_like(source)
            errors = torch.zeros_like(source)
            block_inverse = hessian[start:end, start:end]
            for offset in range(width):
                column = source[:, offset]
                denominator = block_inverse[offset, offset]
                if (start + offset) % block_size == 0:
                    scale, zero = _mcmoe_params(
                        weight[:, start : start + block_size],
                        self.bits,
                    )
                    group_scales.append(scale.detach().clone())
                    group_zeros.append(zero.detach().clone())
                if self.bits == 1:
                    levels = (column >= 0).to(torch.uint8)
                else:
                    levels = (column / scale + zero).round().clamp(0, 2**self.bits - 1).to(torch.uint8)
                codes[:, start + offset] = levels
                reconstructed = _quantize_uniform(column, self.bits, scale, zero)
                quantized[:, offset] = reconstructed
                denominator = torch.where(
                    torch.isfinite(denominator) & (denominator.abs() >= denom_floor),
                    denominator,
                    denom_floor,
                )
                error = torch.nan_to_num(
                    (column - reconstructed) / denominator,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if self.bits > 3:
                    source[:, offset:] -= error.unsqueeze(1).matmul(
                        block_inverse[offset, offset:].unsqueeze(0)
                    )
                errors[:, offset] = error
            result[:, start:end] = quantized
            weight[:, end:] -= errors.matmul(hessian[start:end, end:])
        from pmq.packed import pack_matrix
        self.packed_state = pack_matrix(
            codes, torch.stack(group_scales, dim=1), torch.stack(group_zeros, dim=1),
            bits=self.bits, group_size=group_size, dtype=self.weight.dtype,
        )
        return result.to(dtype=self.weight.dtype)
