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
                        quantizer.configure(bits, perchannel=True, sym=symmetric, mse=False, pack=packed)
                        with patch.object(quantizer, '_quantize', side_effect=AssertionError(
                            'Disabled MSE must not quantize search candidates'
                        )):
                            quantizer.find_params(self.weights, weight=True)
                        if bits == 2:
                            expected_scale = torch.tensor(
                                [4., 4 / 3, 8 / 3, 1.], dtype=torch.float16
                            ).reshape(-1, 1)
                            if symmetric:
                                expected_scale = torch.tensor([16 / 3, 8 / 3, 16 / 3, 1.], dtype=torch.float16).reshape(-1, 1)
                            expected_zero = torch.full_like(expected_scale, 2.) if symmetric else torch.tensor(
                                [1., 0., 3., 0.], dtype=torch.float16
                            ).reshape(-1, 1)
                            expected_zero[-1] = 0
                            torch.testing.assert_close(quantizer.scale, expected_scale, rtol=0, atol=0)
                            torch.testing.assert_close(quantizer.zero, expected_zero, rtol=0, atol=0)

    def test_default_search_covers_1_to_point21_and_keeps_fp16(self):
        from utils.quantizer_moe import normal_quantize
        for packed in (False, True):
            quantizer = Quantizer()
            quantizer.configure(2, perchannel=True, sym=False, pack=packed)
            with patch("utils.quantizer_moe.normal_quantize", wraps=normal_quantize) as search:
                quantizer.find_params(self.weights, weight=True)
            self.assertEqual(search.call_count, 80)
            first = search.call_args_list[0].args[1]
            last = search.call_args_list[-1].args[1]
            self.assertEqual(first.dtype, torch.float16)
            self.assertEqual(last.dtype, torch.float16)
            torch.testing.assert_close(first[0], torch.tensor([4.], dtype=torch.float16), rtol=0, atol=0)
            torch.testing.assert_close(last[0], torch.tensor([4 * .21], dtype=torch.float16), rtol=0, atol=0)
            self.assertEqual(quantizer.scale.dtype, torch.float16)
            self.assertEqual(quantizer.scale[-1].item(), 1)
            self.assertEqual(quantizer.zero[-1].item(), 0)
            if packed:
                expected_scale, expected_zero = quantizer.scale.clone(), quantizer.zero.clone()
            else:
                unpacked_scale, unpacked_zero = quantizer.scale.clone(), quantizer.zero.clone()
        torch.testing.assert_close(expected_scale, unpacked_scale, rtol=0, atol=0)
        torch.testing.assert_close(expected_zero, unpacked_zero, rtol=0, atol=0)

    def test_binary_parameters_do_not_depend_on_mse(self):
        for packed in (False, True):
            quantizers = []
            for enabled in (False, True):
                quantizer = Quantizer()
                quantizer.configure(1, perchannel=True, sym=False, mse=enabled, pack=packed)
                quantizer.find_params(self.weights, weight=True)
                quantizers.append(quantizer)
            torch.testing.assert_close(quantizers[0].scale, quantizers[1].scale, rtol=0, atol=0)
            torch.testing.assert_close(quantizers[0].zero, quantizers[1].zero, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
