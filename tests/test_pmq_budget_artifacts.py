"""Exact budget, native PKL integrity and zero-shot protocol regressions."""

import json
import pickle
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from pmq.allocation import solve_pmq_layers, describe_budget
from pmq.factors import PmqFactors, save_pmq_factors
from pmq.routing import RoutingStatistics
from pmq.zeroshot import evaluate_seven, SEVEN_TASKS
from pmq_runner import _load_factor_payload


class BudgetArtifactTest(unittest.TestCase):
    def test_balanced_budget_uses_largest_loss_reduction(self):
        losses = {layer: {expert: {1: 10. * weight, 2: 3. * weight, 3: 1. * weight}
                  for expert in range(5)} for layer, weight in [(0, 1.), (1, 2.)]}
        result = solve_pmq_layers(losses, average_bit=1.7, shared_units=0, shared_bit=3,
                                  budget_policy="balanced_exact")
        self.assertEqual(sum(value for (layer, _), value in result.items() if layer == 0), 8)
        self.assertEqual(sum(value for (layer, _), value in result.items() if layer == 1), 9)

    def test_all_six_deepseek_budgets_are_exact(self):
        losses = {layer: {expert: {1: 10. * (expert + 1), 2: 3. * (expert + 1), 3: float(expert + 1)}
                  for expert in range(64)} for layer in range(1, 27)}
        configuration = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                        "configs/deepseek-v2-lite.yaml").read_text())
        policy = configuration["pmq"]["budget_policy"]
        self.assertEqual(policy, "balanced_exact")
        started = time.perf_counter()
        for bit in (1.25, 1.5, 1.75, 2., 2.25, 2.5):
            assignments = solve_pmq_layers(losses, average_bit=bit, shared_units=2, shared_bit=3,
                                           budget_policy=policy)
            report = describe_budget(assignments, shared_units=2, shared_bit=3,
                                      requested_average_bit=bit, budget_policy="balanced_exact")
            self.assertEqual(report["actual_average_bit"], bit)
            budgets = list(report["layer_total_budgets"].values())
            self.assertLessEqual(max(budgets) - min(budgets), 1)
            for layer in losses:
                assigned = [value for (index, _), value in assignments.items() if index == layer]
                self.assertIn(2, assigned); self.assertIn(3, assigned)
            if bit in (1.25, 1.75, 2.25):
                self.assertEqual(budgets.count(max(budgets)), 13)
                self.assertEqual([int(layer) for layer, value in report["layer_total_budgets"].items()
                                  if value == max(budgets)], list(range(1, 14)))
        print('CPU_SIX_BUDGET_TEST_SECONDS',time.perf_counter() - started)

    def test_native_pkls_and_corruption_detection(self):
        factors = PmqFactors(RoutingStatistics({0: torch.tensor([1, 2])}, {0: torch.tensor([.5, 1.])}),
                             {0: {0: {1: .3, 2: .2, 3: .1}, 1: {1: .5, 2: .4, 3: .3}}})
        with tempfile.TemporaryDirectory() as folder:
            save_pmq_factors(folder, factors, {"seed": 0, "architecture": "deepseek_moe"})
            names = {p.name for p in Path(folder).iterdir()}
            self.assertEqual(names, {"experts_act_frequency.pkl", "experts_act_weight.pkl", "experts_quant_loss.pkl", "factors_complete.json"})
            payload = _load_factor_payload(folder)
            self.assertTrue(torch.equal(payload["selected_count"][0], torch.tensor([1, 2])))
            with self.assertRaises(FileExistsError):save_pmq_factors(folder, factors, {})
            Path(folder, "experts_quant_loss.pkl").write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, "checksum"):_load_factor_payload(folder)

    def test_seven_calls_harness_without_bos_or_limit(self):
        result = {"results": {task: {"acc,none": .5} for task in SEVEN_TASKS}, "n-shot": {"piqa": 0}}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "qmodel.pt").write_bytes(b'cpu-test')
            with patch('lm_eval.models.huggingface.HFLM') as wrapped, patch('lm_eval.evaluator.simple_evaluate', return_value=result) as evaluate:
                output = evaluate_seven(torch.nn.Linear(2, 2), object(), folder, seed=0)
            self.assertFalse(wrapped.call_args.kwargs["add_bos_token"])
            self.assertEqual(evaluate.call_args.kwargs["num_fewshot"], 0)
            self.assertIsNone(evaluate.call_args.kwargs["limit"])
            self.assertFalse(evaluate.call_args.kwargs["apply_chat_template"])
            self.assertAlmostEqual(json.loads(output.read_text())["summary"]["mean"], .5)


if __name__ == "__main__": unittest.main()
