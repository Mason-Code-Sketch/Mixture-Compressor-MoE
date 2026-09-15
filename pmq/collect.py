"""Model and local-dataset loading for PMQ native routing statistics."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_from_disk

from pmq.bridge import enable_current_project_imports
from pmq.config import ProtocolConfig
from pmq.routing import NativeRoutingCollector, RoutingStatistics


def collect_native_routing_statistics(
    protocol: ProtocolConfig,
    *,
    seed: int,
) -> tuple[RoutingStatistics, dict]:
    """Run the unmodified model and collect PMQ router frequency/weight factors."""
    enable_current_project_imports()
    from src.experiment.data_preparation import prepare_calibration_inputs
    from src.models import get_moe_adapter
    from src.models.loading import get_input_device, load_model, load_model_config, load_tokenizer

    model_cfg = protocol.data["model"]
    calibration_cfg = protocol.data["dataset"]["calibration"]
    sample_cfg = protocol.data["calibration"]
    adapter = get_moe_adapter(protocol.architecture)
    model_config = load_model_config(str(protocol.model_path))
    tokenizer = load_tokenizer(str(protocol.model_path))
    dataset = load_from_disk(str(protocol.calibration_path))
    input_ids, attention_mask = prepare_calibration_inputs(
        tokenizer,
        dataset,
        protocol=str(calibration_cfg["protocol"]),
        split=str(calibration_cfg["split"]),
        num_samples=int(sample_cfg["samples"]),
        seq_len=int(sample_cfg["seq_len"]),
        seed=int(seed),
        add_special_tokens=bool(calibration_cfg.get("add_special_tokens", True)),
    )
    model = load_model(
        str(protocol.model_path),
        device_map="auto",
        attn_implementation="eager",
        torch_dtype="auto",
    )
    collector = NativeRoutingCollector(
        adapter.collect_moe_modules(model), adapter, model_config
    )
    try:
        device = get_input_device(model)
        with torch.inference_mode():
            for ids, mask in zip(input_ids, attention_mask):
                model(
                    input_ids=ids.unsqueeze(0).to(device, non_blocking=True),
                    attention_mask=mask.unsqueeze(0).to(device, non_blocking=True),
                    use_cache=False,
                )
        statistics = collector.close()
    except Exception:
        collector.close()
        raise
    metadata = {
        "model": model_cfg["name"],
        "architecture": adapter.name,
        "seed": int(seed),
        "calibration_path": str(calibration_cfg["path"]),
        "calibration_protocol": calibration_cfg["protocol"],
        "calibration_samples": int(input_ids.shape[0]),
        "calibration_seq_len": int(input_ids.shape[1]),
    }
    del model
    return statistics, metadata


def save_routing_statistics(
    output_dir: str | Path,
    statistics: RoutingStatistics,
    metadata: dict,
) -> Path:
    """Write PMQ routing factors under the run-owned output directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "routing_statistics.pt"
    torch.save(
        {
            "selected_count": statistics.selected_count,
            "selected_weight": statistics.selected_weight,
            "metadata": metadata,
        },
        output,
    )
    return output

