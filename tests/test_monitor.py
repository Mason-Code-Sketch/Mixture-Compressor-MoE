"""Resource records keep retries, component boundaries and peak scopes distinct."""
import json

import pytest

from pmq import monitor


def test_failed_attempt_and_fresh_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "resource_snapshot", lambda: {
        "rss_bytes": 10, "available_bytes": 100,
        "gpu": [{"physical_device": 0, "used_mib": 999}],
    })
    with pytest.raises(ValueError, match="deliberate"):
        with monitor.RunMonitor(tmp_path, "quantize"):
            with monitor.component("quant_gptq"):
                raise ValueError("deliberate")
    path = tmp_path / "quantize_resources.json"
    failed = json.loads(path.read_text())
    assert failed["status"] == failed["components"]["quant_gptq"]["status"] == "failed"
    assert failed["max_sampled_gpu_global_used_mib"] == {"0": 999}
    assert failed["components"]["quant_gptq"]["per_gpu_peak_memory_gib"] == {}
    assert monitor._current_monitor.get() is None
    with monitor.RunMonitor(tmp_path, "quantize"):
        with monitor.component("evaluate_zeroshot"):
            pass
    fresh = json.loads(path.read_text())
    assert fresh["status"] == "complete"
    assert fresh["core_wall_seconds"] == 0
    assert "quant_gptq" not in fresh["components"]
    assert json.loads(path.with_name("quantize_resources." + failed["run_id"] + ".json").read_text()) == failed


def test_timer_includes_synchronization_and_failed_component_stays_failed(tmp_path, monkeypatch):
    run = monitor.RunMonitor(tmp_path, "quantize")
    run.start = 0.0
    clock = [10.0]
    monkeypatch.setattr(monitor.time, "perf_counter", lambda: clock[0])
    def peaks(reset=False):
        if not reset:
            clock[0] += 3.0
        return {"0": 1.0}
    monkeypatch.setattr(monitor, "_cuda_peaks", peaks)
    with pytest.raises(ValueError):
        with run.component("quant_gptq"):
            clock[0] += 2.0
            raise ValueError("deliberate")
    assert run.components["quant_gptq"]["wall_seconds"] == 5.0
    with run.component("quant_gptq"):
        pass
    assert run.components["quant_gptq"]["status"] == "failed"
