"""Finite, read-only Hessian factors and deterministic diagonal fallback."""
import pytest
import torch

from pmq.hessian import inverse_hessian_factor

@pytest.mark.parametrize("hessian", [torch.zeros(4, 4), torch.eye(4),
    torch.tensor([[1., 4.], [0., 1.]]), torch.tensor([[1., 1000.], [1000., 1.]])])
def test_factor_keeps_input_and_matches_stable_policy(hessian):
    source = hessian.clone()
    result = inverse_hessian_factor(hessian, .01)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert torch.equal(hessian, source)
    assert torch.all(result.diagonal() > 0)

def test_all_cholesky_failures_fall_back_to_diagonal(monkeypatch):
    original = torch.linalg.cholesky
    attempts = []
    def factor(matrix, **kwargs):
        attempts.append(matrix.clone())
        if len(attempts) <= 5:
            raise torch.linalg.LinAlgError("forced test failure")
        return original(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, "cholesky", factor)
    result = inverse_hessian_factor(torch.eye(3), .01)
    assert len(attempts) == 6
    assert torch.equal(result, torch.eye(3))

def test_nonfinite_hessian_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        inverse_hessian_factor(torch.full((2, 2), float("nan")), .01)


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_low_bit_compensation_changes_the_next_column(bits):
    from unittest.mock import patch
    from pmq.gptq import TensorGPTQ
    weight = torch.tensor([[-.25, .25]], dtype=torch.float16)
    scale = torch.ones(1, dtype=torch.float16)
    zero = torch.full((1,), 2 ** (bits - 1) - .5, dtype=torch.float16)
    factor = torch.tensor([[1., 2.], [0., 1.]])
    with patch("pmq.gptq._mcmoe_params", return_value=(scale, zero)), \
         patch("pmq.hessian.inverse_hessian_factor", return_value=factor):
        result = TensorGPTQ(weight, bits=bits).quantize(group_size=2, percdamp=.01)
    assert torch.equal(result, torch.tensor([[-.5, -.5]], dtype=torch.float16))
