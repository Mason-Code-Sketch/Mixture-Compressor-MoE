"""RTN scoring candidates using a floating zero point and an FP32 error search."""

import torch


def _candidate(group, bit, factors):
    if bit == 1:
        scale = 2 * group.abs().mean(dim=1, keepdim=True)
        return ((group >= 0).float() - 0.5) * scale
    maximum = 2**bit - 1
    low = group.amin(dim=1, keepdim=True).clamp_max(0)
    high = group.amax(dim=1, keepdim=True).clamp_min(0)
    high = torch.where((low == 0) & (high == 0), maximum, high)
    best_error = torch.full_like(low, float("inf"))
    best_scale = (high - low) / maximum
    best_zero = -low / best_scale
    for factor in factors:
        lower = factor * low
        scale = (factor * high - lower) / maximum
        zero = -lower / scale
        levels = (group / scale + zero).round().clamp_(0, maximum)
        error = ((levels - zero) * scale - group).abs_().pow_(2.4).sum(dim=1, keepdim=True)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_scale = torch.where(improved, scale, best_scale)
        best_zero = torch.where(improved, zero, best_zero)
    levels = (group / best_scale + best_zero).round().clamp_(0, maximum)
    return (levels - best_zero) * best_scale


@torch.no_grad()
def normal_quantize(w, blocksize=128, wbit=2, *, mcmoe_rules=False):
    """Search each row and input-column group independently, including a short tail."""
    if wbit == 0:
        return torch.zeros_like(w)
    if blocksize <= 0 or not 1 <= wbit <= 8 or w.ndim != 2:
        raise ValueError("RTN requires a matrix, a positive group size and 1..8 bits")
    factors = [1 - i / 100 for i in range(80)]
    if mcmoe_rules and wbit > 1:
        factors = torch.cat((w.new_ones(1, dtype=torch.float32),
                             torch.linspace(1, 1.1, 51, device=w.device, dtype=torch.float32)[1:],
                             torch.linspace(1, 0.9, 51, device=w.device, dtype=torch.float32)[1:]))
    rows, columns = w.shape
    full = columns - columns % blocksize
    result = torch.empty_like(w)
    chunk = max(1, (8 * 1024 * 1024) // columns)
    for start in range(0, rows, chunk):
        for left, right, width in ((0, full, blocksize), (full, columns, columns - full)):
            if left == right:
                continue
            source = w[start:start + chunk, left:right]
            group = source.reshape(-1, width).float()
            result[start:start + chunk, left:right] = _candidate(group, wbit, factors).reshape_as(source)
    return result
