"""Exercise the authorized low-budget exception with the native HiGHS solver."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from pmq.allocation import allocation_requires_mid_and_high, solve_pmq_layers
from pmq.solver import solve_layer_ilp


BITS = (1.25, 1.5, 1.75, 2, 2.25, 2.5)


def losses():
    return {expert: {1: 100.0 + expert, 2: 20.0 + expert / 10, 3: 0.0}
            for expert in range(8)}


class MixtralLowBitAllocationTest(unittest.TestCase):
    def test_default_constraints_remain_enabled_for_all_models_and_budgets(self):
        for architecture in ("mixtral", "qwen3_moe", "qwen2_moe", "deepseek_moe"):
            for bit in BITS:
                with self.subTest(architecture=architecture, bit=bit):
                    self.assertTrue(allocation_requires_mid_and_high(architecture, bit))

    def test_exception_only_selects_mixtral_1p25(self):
        for bit in BITS:
            self.assertEqual(allocation_requires_mid_and_high(
                "mixtral", bit, relax_mixtral_1p25_c2c3=True), bit != 1.25)
        for architecture in ("qwen3_moe", "qwen2_moe", "deepseek_moe"):
            with self.assertRaisesRegex(ValueError, "restricted to Mixtral"):
                allocation_requires_mid_and_high(
                    architecture, 1.25, relax_mixtral_1p25_c2c3=True)
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            allocation_requires_mid_and_high(
                "mixtral", 1.25, relax_mixtral_1p25_c2c3="true")

    def test_native_1p25_infeasibility_and_feasible_unchanged_budget(self):
        original = losses()
        before = copy.deepcopy(original)
        with self.assertRaisesRegex(RuntimeError, "did not solve optimally"):
            solve_layer_ilp(original, bit_budget=10)
        result = solve_pmq_layers(
            {layer: original for layer in range(2)}, average_bit=1.25,
            shared_units=0, shared_bit=3, require_mid_and_high=False)
        self.assertEqual(original, before)
        for layer in range(2):
            values = [result[layer, expert] for expert in range(8)]
            self.assertEqual(sum(values), 10)
            self.assertTrue(set(values) <= {1, 2, 3})

    def test_other_five_budgets_match_native_assignments(self):
        for bit in BITS[1:]:
            with self.subTest(bit=bit):
                original = solve_layer_ilp(losses(), bit_budget=int(bit * 8))
                actual = solve_pmq_layers(
                    {0: losses()}, average_bit=bit, shared_units=0, shared_bit=3,
                    require_mid_and_high=allocation_requires_mid_and_high(
                        "mixtral", bit, relax_mixtral_1p25_c2c3=True))
                self.assertEqual(actual, {(0, expert): value
                                          for expert, value in original.items()})
                self.assertIn(2, actual.values())
                self.assertIn(3, actual.values())

    def test_native_runner_records_actual_constraint_and_budget(self):
        from pmq_runner import _allocate
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root / "configs/mixtral-8x7b-v0.1.yaml").read_text())
        self.assertIs(cfg["pmq"]["relax_mixtral_1p25_c2c3"], True)
        protocol = SimpleNamespace(data=cfg, architecture="mixtral",
            candidate_bits=(1, 2, 3), model_path=Path("unused"))
        payload = {"metadata": {"seed": 0, "architecture": "mixtral"},
            "selected_count": {0: torch.ones(8)},
            "selected_weight": {0: torch.ones(8)}, "candidate_loss": {0: losses()}}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with patch("pmq_runner._load_factor_payload", return_value=payload), \
                 patch("pmq_runner.load_model_config", return_value=object()), \
                 patch("pmq_runner.provenance", return_value={"fixture": True}):
                path = _allocate(protocol, factors_path=out, output_dir=out,
                                 average_bit=1.25, seed=0)
            result = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(result["average_bit"], 1.25)
            self.assertEqual(result["allocation_constraints"],
                             {"require_mid_and_high": False})
            self.assertEqual(sum(r["gate_proj"] for r in result["allocation"].values()), 10)
            self.assertTrue(all(len(set(r.values())) == 1
                                for r in result["allocation"].values()))


if __name__ == "__main__":
    unittest.main()
