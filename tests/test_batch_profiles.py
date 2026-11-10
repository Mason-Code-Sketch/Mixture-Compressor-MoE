from types import SimpleNamespace
import pytest
from pmq.batch_profiles import auto_batch_cap, resolve_cli_batch_cap

@pytest.mark.parametrize("model_type,caps", [("mixtral", [34, 10, 13]), ("deepseek_v2", [32, 24, 20]), ("qwen2_moe", [32, 24, 56]), ("qwen3_moe", [51, 59, 84]), ("unknown", [51, 17, 96])])
def test_model_profiles_and_explicit_override(model_type, caps):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    for profile, cap in zip(["seven", "mmlu_5shot", "gsm8k_5shot"], caps):
        assert auto_batch_cap(model, profile) == cap
        assert auto_batch_cap(model, profile, 7) == 7
        args = SimpleNamespace(max_batch_size=51, _max_batch_size_explicit=False)
        assert resolve_cli_batch_cap(args, model, profile) == cap
        args._max_batch_size_explicit = True
        args.max_batch_size = 51
        assert resolve_cli_batch_cap(args, model, profile) == 51
    assert auto_batch_cap(model, "zeroshot") == caps[0]

def test_manual_namespace_retains_explicit_value():
    model = SimpleNamespace(config=SimpleNamespace(model_type="mixtral"))
    assert resolve_cli_batch_cap(SimpleNamespace(max_batch_size=11), model, "seven") == 11

import json
from unittest.mock import Mock, patch
from pmq import zeroshot as seven
from pmq_runner import parse_args

@pytest.mark.parametrize("model_type,cap", [("mixtral", 34), ("deepseek_v2", 32), ("qwen2_moe", 32), ("qwen3_moe", 51)])
@pytest.mark.parametrize("explicit", [None, 51])
def test_loaded_model_cap_reaches_harness_and_saved_result(tmp_path, model_type, cap, explicit):
    (tmp_path / "qmodel.pt").write_bytes(b"checkpoint")
    model = Mock()
    model.config.model_type = model_type
    raw = {"results": {task: {"acc,none": .5} for task in seven.SEVEN_TASKS},
           "n-shot": {task: 0 for task in seven.SEVEN_TASKS}}
    extra = [] if explicit is None else ["--max-batch-size", str(explicit)]
    args = parse_args(["--config", "model.yaml", "--phase", "zeroshot", *extra])
    expected = cap if explicit is None else explicit
    assert resolve_cli_batch_cap(args, model, "zeroshot") == expected
    with patch("lm_eval.models.huggingface.HFLM") as wrapper, patch("lm_eval.evaluator.simple_evaluate", return_value=raw), patch("pmq.evaluation_backend.prepare_evaluation_model", return_value="grouped_cached"):
        # Also cover direct API defaults independently of CLI resolution.
        seven.evaluate_seven(model, "tokenizer", tmp_path, max_batch_size=explicit)
    assert wrapper.call_args.kwargs["max_batch_size"] == expected
    assert json.loads((tmp_path / "zeroshot.json").read_text())["experiment"]["max_batch_size"] == expected

@pytest.mark.parametrize("profile,cap", [("mmlu_5shot", 10), ("gsm8k_5shot", 13)])
def test_supplemental_cli_resolves_after_loading(profile, cap):
    args = parse_args(["--config", "model.yaml", "--phase", profile])
    assert resolve_cli_batch_cap(args, SimpleNamespace(config=SimpleNamespace(model_type="mixtral")), profile) == cap
