"""Tensor-based GPTQ used only by PMQ final-plan materialization."""

from __future__ import annotations

import math

import torch

from utils.quantizer_moe import Quantizer


class TensorGPTQ:
    """Apply the repository's GPTQ update to one 2-D weight tensor."""

    def __init__(self, weight: torch.Tensor, *, bits: int):
        if weight.ndim != 2:
            raise ValueError(f"GPTQ requires a matrix weight, got {tuple(weight.shape)}")
        self.weight = weight.detach()
        self.columns = int(weight.shape[1])
        self.hessian = torch.zeros(
            (self.columns, self.columns), device=weight.device, dtype=torch.float32
        )
        self.samples = 0
        self.quantizer = Quantizer()
        self.quantizer.configure(int(bits), perchannel=True, sym=False, mse=True)

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

        result = torch.zeros_like(weight)
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
                    self.quantizer.find_params(
                        weight[:, start : start + block_size], weight=True
                    )
                reconstructed = self.quantizer.quantize(column.unsqueeze(1)).flatten()
                quantized[:, offset] = reconstructed
                error = (column - reconstructed) / denominator
                source[:, offset:] -= error.unsqueeze(1).matmul(
                    block_inverse[offset, offset:].unsqueeze(0)
                )
                errors[:, offset] = error
            result[:, start:end] = quantized
            weight[:, end:] -= errors.matmul(hessian[start:end, end:])
        return result.to(dtype=self.weight.dtype)
