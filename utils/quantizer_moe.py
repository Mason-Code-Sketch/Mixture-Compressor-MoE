import math
import torch
import torch.nn as nn

@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):
    return torch.clamp(torch.round(x / scale + zero), 0, maxq)

@torch.no_grad()
def binary_scale(x):
    scale_tensor = torch.mean(torch.abs(x), dim=1) * 2
    zero = 0.5 * torch.ones_like(scale_tensor)
    return scale_tensor, zero

@torch.no_grad()
def binary(x):
    x_ = torch.zeros_like(x)
    x_ += x
    binary_slice = torch.where(x_ >= 0, 1, -1)
    return binary_slice/2 + 0.5 # change (-1, 1) weights to (0, 1) weights

@torch.no_grad()
def residual_binary(x, scale, r_scale, zero, order=2):
    sum_order = torch.zeros_like(x)
    for od in range(2):
        residual = x - sum_order
        binary_slice = torch.where(residual >= 0, 1, -1)
        if od == 0:
            sum_order += scale * binary_slice
        else:
            sum_order += r_scale * binary_slice
    del binary_slice
    return sum_order

@torch.no_grad()
def residual_scale(x, order=2):
    sum_order = torch.zeros_like(x).to(x.device)
    scale_tensors = [torch.mean(torch.abs(x), dim=1)] * order
    zero = 0.5 * torch.ones_like(x)
    for od in range(order):
        residual = x - sum_order
        scale_tensors[od] = torch.mean(torch.abs(residual), dim=1).to(x.device)
        binary_slice = torch.where(residual >= 0, 1, -1)
        sum_order += scale_tensors[od].unsqueeze(1) * binary_slice
    del sum_order
    del binary_slice
    return scale_tensors[0], scale_tensors[1], zero

@torch.no_grad()
def r_residual_scale(x, order=3):
    sum_order = torch.zeros_like(x).to(x.device)
    scale_tensors = [torch.mean(torch.abs(x), dim=1)] * order
    zero = 0.5 * torch.ones_like(x)
    for od in range(order):
        residual = x - sum_order
        scale_tensors[od] = torch.mean(torch.abs(residual), dim=1).to(x.device)
        binary_slice = torch.where(residual >= 0, 1, -1)
        sum_order += scale_tensors[od].unsqueeze(1) * binary_slice
    del sum_order
    del binary_slice
    return scale_tensors[0], scale_tensors[1], scale_tensors[2], zero

@torch.no_grad()
def r_residual_binary(x, scale, r_scale, rr_scale, order=3):
    sum_order = torch.zeros_like(x)
    for od in range(3):
        residual = x - sum_order
        binary_slice = torch.where(residual >= 0, 1, -1)
        if od == 0:
            sum_order += scale * binary_slice
        elif od == 1:
            sum_order += r_scale * binary_slice
        else:
            sum_order += rr_scale * binary_slice
    del binary_slice
    return sum_order



class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self._params_ready = False
        self.register_buffer('maxq', torch.tensor(0, dtype=torch.float16))
        self.register_buffer('scale', torch.zeros(shape, dtype=torch.float16))
        self.register_buffer('zero', torch.zeros(shape, dtype=torch.float16))

    def configure(self, bits, perchannel=False, sym=False, mse=True, norm=2.4, grid=100, maxshrink=.8, trits=False, pack=False):

        if trits:
            raise ValueError("Ternary quantization is not supported by PMQ packing")
        if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 8:
            raise ValueError("PMQ bits must be an integer in 1..8")
        if any(not isinstance(value, bool) for value in (perchannel, sym, mse, pack)):
            raise ValueError("perchannel, sym, mse and pack must be boolean")
        if not isinstance(grid, int) or isinstance(grid, bool) or grid < 1:
            raise ValueError("MSE grid must be a positive integer")
        if not math.isfinite(maxshrink) or not 0 < maxshrink < 1 or int(maxshrink * grid) < 1:
            raise ValueError("MSE shrink range must contain at least one positive factor")
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("MSE norm must be finite and positive")
        self._params_ready = False
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.scale = torch.zeros_like(self.scale, dtype=torch.float16)
        self.pack = pack

    def _quantize(self, x, scale, zero, maxq):
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        if self.maxq == 1:
            q = binary(x)
        # elif self.maxq == 3 and (not self.pack):
        #     # TODO: pack the residual quantization
        #     return residual_binary(x, scale=scale, r_scale=self.r_scale, zero=zero, order=2)
        # elif self.maxq == 7 and (not self.pack):
        #     return r_residual_binary(x, scale=scale, r_scale=self.r_scale, rr_scale=self.rr_scale, order=3)
        else:
            q = normal_quantize(x, scale, zero, maxq)
        return scale * (q - zero) if not self.pack else q

    def find_params(self, w, weight=False):
        if not isinstance(weight, bool):
            raise ValueError("weight must be boolean")
        shape = w.shape
        reference = w.detach()
        if self.perchannel:
            if weight:
                reference = reference.flatten(1)
            elif len(shape) == 4:
                reference = reference.permute(1, 0, 2, 3).flatten(1)
            elif len(shape) == 3:
                reference = reference.reshape(-1, shape[-1]).t()
            elif len(shape) == 2:
                reference = reference.t()
            else:
                raise ValueError("Per-channel quantization requires a 2-D, 3-D or 4-D tensor")
        else:
            reference = reference.flatten().unsqueeze(0)

        x = reference.to(torch.float16)
        if not torch.isfinite(x).all():
            raise ValueError("PMQ parameters require weights representable in FP16")
        maxq = self.maxq.to(w.device)
        if maxq == 1:
            scale, zero = binary_scale(x)
        else:
            tmp = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            xmin = torch.minimum(x.amin(1), tmp)
            xmax = torch.maximum(x.amax(1), tmp)
            if self.sym:
                xmax = torch.maximum(xmin.abs(), xmax)
                xmin = -xmax
            empty = (xmin == 0) & (xmax == 0)
            xmin = torch.where(empty, -torch.ones_like(xmin), xmin)
            xmax = torch.where(empty, torch.ones_like(xmax), xmax)
            scale = (xmax - xmin) / maxq
            zero = torch.full_like(scale, (maxq + 1) / 2) if self.sym else -xmin / scale

            if self.mse:
                best = torch.full_like(scale, float('inf'))
                for index in range(int(self.maxshrink * self.grid)):
                    p = 1 - index / self.grid
                    xmin1 = p * xmin
                    xmax1 = p * xmax
                    scale1 = (xmax1 - xmin1) / maxq
                    zero1 = zero if self.sym else -xmin1 / scale1
                    codes = normal_quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), maxq)
                    error = scale1.unsqueeze(1) * (codes - zero1.unsqueeze(1))
                    error -= reference
                    error.abs_().pow_(self.norm)
                    error = error.sum(1)
                    improved = error < best
                    best[improved] = error[improved]
                    scale[improved] = scale1[improved]
                    if not self.sym:
                        zero[improved] = zero1[improved]
            scale[empty] = 1
            zero[empty] = 0

        if not self.perchannel:
            channels = shape[0] if weight else shape[1] if len(shape) != 3 else shape[2]
            scale = scale.repeat(channels)
            zero = zero.repeat(channels)
        if weight:
            output_shape = [-1] + [1] * (len(shape) - 1)
        elif len(shape) == 4:
            output_shape = (1, -1, 1, 1)
        elif len(shape) == 3:
            output_shape = (1, 1, -1)
        else:
            output_shape = (1, -1)
        self.scale = scale.reshape(output_shape)
        self.zero = zero.reshape(output_shape)
        self.maxq = maxq
        self._params_ready = True

    def quantize(self, x):
        if self.ready():
            if self.pack:
                return self._quantize(x, self.scale, self.zero, self.maxq), self.scale, self.zero
            return self._quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        if not self._params_ready:
            return False
        valid_scale = self.scale >= 0 if self.maxq == 1 else self.scale > 0
        return bool(valid_scale.all() and torch.isfinite(self.scale).all() and torch.isfinite(self.zero).all())


# random_tensor = torch.randn(10, 10)
# mask = torch.randn(10, 10) < 0.5  # Generate a mask where about 50% are True
# random_tensor[mask] = 0 

# sign_tensor = torch.sign(random_tensor)

# print(random_tensor)
# print(sign_tensor)

# sign_tensor = torch.where(random_tensor >= 0, 1, -1)
# print(sign_tensor)
