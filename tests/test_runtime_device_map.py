from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pmq import runtime


@pytest.mark.parametrize("placement", [None, "cpu", "cuda:0"])
def test_loader_respects_explicit_placement(monkeypatch, placement):
    settings = {"dtype": "float16", "attn_implementation": "eager"}
    if placement is not None:
        settings["device_map"] = placement
    protocol = SimpleNamespace(data={"model": settings}, model_path=Path("model"))
    loader = Mock()
    monkeypatch.setattr(runtime.AutoModelForCausalLM, "from_pretrained", loader)
    assert runtime.load_model(protocol) is loader.return_value
    assert loader.call_args.kwargs["device_map"] == (placement or "auto")
    assert loader.call_args.kwargs["experts_implementation"] == "eager"
    assert "offload_folder" not in loader.call_args.kwargs
