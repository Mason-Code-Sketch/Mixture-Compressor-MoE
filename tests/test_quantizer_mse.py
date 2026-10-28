"""Regression checks for the original PMQ quantizer's MSE switch."""

import unittest
from unittest.mock import patch

import torch

from utils.quantizer_moe import Quantizer


class QuantizerMseTest(unittest.TestCase):
    def setUp(self):
        self.weights = torch.tensor([
            [-4., -2., 1., 8.],
            [1., 2., 3., 4.],
            [-8., -4., -1., -2.],
            [0., 0., 0., 0.],
        ])

    def test_disabled_search_keeps_minmax_parameters(self):
        for bits in (2, 3, 4):
            for symmetric in (False, True):
                for packed in (False, True):
                    with self.subTest(bits=bits, symmetric=symmetric, packed=packed):
                        quantizer = Quantizer()
                        quantizer.configure(bits, sym=symmetric, mse=False, pack=packed)
                        with patch.object(quantizer, '_quantize', side_effect=AssertionError(
                            'Disabled MSE must not quantize search candidates'
                        )):
                            quantizer.find_params(self.weights, weight=True)
                        if bits == 2:
                            expected_scale = torch.tensor(
                                [4., 4 / 3, 8 / 3, 2 / 3], dtype=torch.float16
                            ).reshape(-1, 1)
                            expected_zero = torch.full_like(expected_scale, 2.) if symmetric else torch.tensor(
                                [1., 0., 3., 1.5], dtype=torch.float16
                            ).reshape(-1, 1)
                            torch.testing.assert_close(quantizer.scale, expected_scale, rtol=0, atol=0)
                            torch.testing.assert_close(quantizer.zero, expected_zero, rtol=0, atol=0)

    def test_enabled_search_preserves_existing_numeric_result(self):
        # Captured from the original solver before the switch fix.
        quantizer = Quantizer()
        quantizer.configure(2, sym=False, mse=True, pack=True)
        with patch.object(quantizer, '_quantize', wraps=quantizer._quantize) as search:
            quantizer.find_params(self.weights, weight=True)
        self.assertEqual(search.call_count, 101)
        torch.testing.assert_close(
            quantizer.scale,
            torch.tensor([3.599609375, 1.2666015625, 2.431640625, .60009765625], dtype=torch.float16).reshape(-1, 1),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            quantizer.zero, torch.tensor([1., 0., 3., 1.5], dtype=torch.float16).reshape(-1, 1),
            rtol=0, atol=0,
        )

    def test_binary_parameters_do_not_depend_on_mse(self):
        for packed in (False, True):
            quantizers = []
            for enabled in (False, True):
                quantizer = Quantizer()
                quantizer.configure(1, sym=False, mse=enabled, pack=packed)
                quantizer.find_params(self.weights, weight=True)
                quantizers.append(quantizer)
            torch.testing.assert_close(quantizers[0].scale, quantizers[1].scale, rtol=0, atol=0)
            torch.testing.assert_close(quantizers[0].zero, quantizers[1].zero, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
