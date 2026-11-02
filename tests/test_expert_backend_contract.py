import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

class BackendContract(unittest.TestCase):
    def test_four_architectures_force_eager(self):
        source = Path(__file__).resolve().parents[1] / "pmq/runtime.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_model")
        for architecture in ("deepseek_moe", "qwen2_moe", "qwen3_moe", "mixtral"):
            with self.subTest(architecture=architecture):
                calls = []
                model = SimpleNamespace(eval=lambda: None)
                def load(path, **kwargs):
                    calls.append(kwargs)
                    return model
                scope = {"torch": SimpleNamespace(float16="float16"),
                         "AutoModelForCausalLM": SimpleNamespace(from_pretrained=load),
                         "_native_deepseek": lambda path: architecture == "deepseek_moe",
                         "align_deepseek_softmax_scale": lambda model: 0}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
                protocol = SimpleNamespace(model_path="test-model", data={"model": {
                    "architecture": architecture, "dtype": "float16", "attn_implementation": "eager"}})
                scope["load_model"](protocol)
                self.assertEqual(calls[0]["experts_implementation"], "eager")
                self.assertEqual(calls[0]["attn_implementation"], "eager")

if __name__ == "__main__":
    unittest.main()
