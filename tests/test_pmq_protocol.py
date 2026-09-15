"""Small deterministic checks for PMQ protocol conversion helpers."""

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
from pmq.factors import _candidate_loss_for_layer
from pmq.bridge import enable_current_project_imports
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

    def test_config_paths_are_relative_to_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "configs" / "model.yaml"
            config.parent.mkdir()
            config.write_text(
                "model:\n  path: ../models/test\n"
                "current_project_config: ../project/model.yaml\n"
                "dataset:\n"
                "  calibration:\n    path: ../datasets/wikitext2\n"
                "  scoring:\n    path: ../datasets/c4\n"
            )
            loaded = load_protocol_config(config)
            self.assertEqual(loaded.model_path, (root / "models" / "test").resolve())
            self.assertEqual(
                loaded.calibration_path,
                (root / "datasets" / "wikitext2").resolve(),
            )
            self.assertEqual(
                loaded.current_project_config,
                (root / "project" / "model.yaml").resolve(),
            )

    def test_repository_configs_resolve_to_existing_inputs(self):
        for config_path in sorted(Path("configs").glob("*.yaml")):
            loaded = load_protocol_config(config_path)
            self.assertTrue(loaded.model_path.is_dir(), config_path)
            self.assertTrue(loaded.calibration_path.is_dir(), config_path)
            self.assertTrue(loaded.scoring_path.is_dir(), config_path)
            self.assertTrue(loaded.current_project_config.is_file(), config_path)

    def test_native_routing_collector_keeps_weight_sum(self):
        class Gate(torch.nn.Module):
            def forward(self, values):
                return values

        class Moe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = Gate()

        class Adapter:
            name = "test"

            def num_experts(self, _config):
                return 3

            def num_experts_per_tok(self, _config):
                return 2

            def router_hook_module(self, module):
                return module.gate

            def scoring_route_state(self, _module, _inputs, output):
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

    def test_solver_requires_gurobi_only_when_called(self):
        self.assertTrue(callable(solve_layer_ilp))

    def test_mixtral_logits_hook_reconstructs_normalized_topk(self):
        class Gate(torch.nn.Module):
            def forward(self, values):
                return values

        class Moe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = Gate()

        class Adapter:
            name = "mixtral"

            def num_experts(self, _config):
                return 3

            def num_experts_per_tok(self, _config):
                return 2

            def router_hook_module(self, module):
                return module.gate

            def scoring_route_state(self, _module, _inputs, _output):
                return None

        module = Moe()
        collector = NativeRoutingCollector({0: module}, Adapter(), object())
        module.gate(torch.tensor([[[3.0, 2.0, 1.0]]]))
        statistics = collector.close()
        self.assertEqual(statistics.selected_count[0].tolist(), [1, 1, 0])
        self.assertTrue(
            torch.allclose(
                statistics.selected_weight[0].sum(),
                torch.tensor(1.0, dtype=torch.float64),
            )
        )

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

    def test_candidate_loss_uses_cached_weight_and_restores_native_module(self):
        enable_current_project_imports()
        from src.scoring.candidate_weights import CandidateWeightCache

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Parameter(torch.tensor([[1.0]]))
                self.up = torch.nn.Parameter(torch.tensor([[1.0]]))
                self.down = torch.nn.Parameter(torch.tensor([[1.0]]))

            def forward(self, hidden):
                return hidden * (self.gate + self.up + self.down)

        class Adapter:
            weight_types = ("gate_proj", "up_proj", "down_proj")

            def expert_weights(self, module, _expert):
                return {
                    "gate_proj": module.gate,
                    "up_proj": module.up,
                    "down_proj": module.down,
                }

            def set_expert_weights(self, module, _expert, weights):
                module.gate.copy_(weights["gate_proj"])
                module.up.copy_(weights["up_proj"])
                module.down.copy_(weights["down_proj"])

        cache = CandidateWeightCache()
        cache._record_expert(0, 0)
        for bit, value in ((1, 0.0), (2, 0.5), (3, 1.0)):
            for projection in Adapter.weight_types:
                cache._weights[(0, 0, projection, bit)] = torch.tensor([[value]])
        module = Module()
        hidden = torch.tensor([[[2.0]]])
        reference = module(hidden).detach()
        losses = _candidate_loss_for_layer(
            layer=0,
            module=module,
            adapter=Adapter(),
            cache=cache,
            bits=(1, 2, 3),
            inputs=[hidden],
            outputs=[reference],
        )
        self.assertGreater(losses[0][1], losses[0][2])
        self.assertEqual(losses[0][3], 0.0)
        self.assertTrue(torch.equal(module(hidden), reference))


if __name__ == "__main__":
    unittest.main()
