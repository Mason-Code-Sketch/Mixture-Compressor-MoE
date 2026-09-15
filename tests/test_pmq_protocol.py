"""Deterministic checks for the standalone PMQ protocol."""

from pathlib import Path
import tempfile
import unittest

import torch

from pmq.allocation import (
    build_loss_by_layer,
    solve_pmq_layers,
    tied_projection_plan,
    validate_layer_budget,
)
from pmq.config import load_protocol_config
from pmq.quantize import quantized_weights
from pmq.routing import NativeRoutingCollector
from pmq.solver import solve_layer_ilp


class PmqProtocolTest(unittest.TestCase):
    def test_tied_projection_plan(self):
        plan = tied_projection_plan({(2, 1): 3, (0, 4): 1})
        self.assertEqual(plan["0,4"], {
            "gate_proj": 1, "up_proj": 1, "down_proj": 1,
        })
        self.assertEqual(plan["2,1"]["down_proj"], 3)

    def test_layer_budget_with_qwen_shared_units(self):
        self.assertEqual(
            validate_layer_budget(
                num_routed_experts=60,
                average_bit=1.5,
                shared_units=4,
                shared_bit=3,
            ),
            84,
        )

    def test_config_resolves_assets_without_cross_project_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "workspace" / "third_party" / "pmq-repository"
            config = repository / "configs" / "model.yaml"
            config.parent.mkdir(parents=True)
            assets = root / "workspace"
            (assets / "models" / "test").mkdir(parents=True)
            (assets / "datasets" / "wiki").mkdir(parents=True)
            (assets / "datasets" / "c4").mkdir(parents=True)
            config.write_text(
                "model:\n  id: test\n  architecture: mixtral\n"
                "dataset:\n"
                "  calibration:\n    name: wiki\n"
                "  evaluations:\n    c4:\n      name: c4\n"
                "pmq:\n  candidate_bits: [1, 2, 3]\n"
            )
            loaded = load_protocol_config(config)
            self.assertEqual(loaded.model_path, (assets / "models" / "test").resolve())
            self.assertEqual(loaded.calibration_path, (assets / "datasets" / "wiki").resolve())
            self.assertEqual(loaded.evaluation_paths["c4"], (assets / "datasets" / "c4").resolve())

    def test_repository_configs_do_not_reference_other_projects(self):
        repository = Path(__file__).parents[1]
        forbidden = ("moe" + "-ptq", "current" + "_project", "candidate" + "-cache-dir")
        for path in list((repository / "pmq").glob("*.py")) + [repository / "pmq_runner.py"]:
            content = path.read_text()
            self.assertFalse(any(token in content for token in forbidden), path)

    def test_native_routing_collector_keeps_weight_sum(self):
        class Gate(torch.nn.Module):
            def forward(self, values):
                return values

        class Moe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = Gate()

        class Adapter:
            def num_experts(self, _module):
                return 3

            def topk(self, _module, _config):
                return 2

            def router_module(self, module):
                return module.gate

            def route_state(self, _module, output, _config):
                return output

        module = Moe()
        collector = NativeRoutingCollector({5: module}, Adapter(), object())
        indices = torch.tensor([[[0, 2], [1, 2]]])
        weights = torch.tensor([[[0.8, 0.2], [0.7, 0.3]]])
        module.gate((indices, weights))
        stats = collector.close()
        self.assertEqual(stats.selected_count[5].tolist(), [1, 1, 2])
        self.assertTrue(torch.allclose(
            stats.selected_weight[5], torch.tensor([0.8, 0.7, 0.5], dtype=torch.float64)
        ))

    def test_pmq_objective_and_highs_assignment(self):
        count = {0: torch.tensor([8, 2, 1])}
        weight = {0: torch.tensor([7.0, 2.0, 1.0])}
        candidate_loss = {
            0: {
                0: {1: 3.0, 2: 2.0, 3: 1.0},
                1: {1: 2.0, 2: 1.0, 3: 0.5},
                2: {1: 1.0, 2: 0.5, 3: 0.25},
            }
        }
        objective = build_loss_by_layer(
            selected_count=count,
            selected_weight=weight,
            candidate_loss=candidate_loss,
        )
        assignments = solve_pmq_layers(
            objective,
            average_bit=2.0,
            shared_units=0,
            shared_bit=3,
            backend="highs",
        )
        self.assertEqual(sum(assignments.values()), 6)
        self.assertIn(2, assignments.values())
        self.assertIn(3, assignments.values())

    def test_normal_candidates_stay_local_and_preserve_shapes(self):
        source = {"w1": torch.tensor([[1.0, -2.0, 3.0, -4.0]])}
        candidate = quantized_weights(source, bit=2, group_size=4)
        self.assertEqual(candidate["w1"].shape, source["w1"].shape)
        self.assertEqual(candidate["w1"].device, source["w1"].device)

    def test_solver_requires_gurobi_only_when_called(self):
        self.assertTrue(callable(solve_layer_ilp))


if __name__ == "__main__":
    unittest.main()
