"""Configuration loading for standalone PMQ experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProtocolConfig:
    """Resolved model, dataset, and quantization settings for one PMQ run."""

    source_path: Path
    data: dict
    repository_root: Path
    asset_root: Path
    model_path: Path
    factor_path: Path
    gptq_calibration_path: Path
    evaluation_paths: dict[str, Path]

    @property
    def architecture(self) -> str:
        return str(self.data["model"]["architecture"])

    @property
    def candidate_bits(self) -> tuple[int, ...]:
        return tuple(int(bit) for bit in self.data["pmq"]["candidate_bits"])


def _resolve_asset_root(repository_root: Path) -> Path:
    """Locate the shared models/ and datasets/ parent without host paths."""
    candidates = (
        repository_root.parent.parent,
        repository_root.parent.parent.parent / "data",
    )
    for candidate in candidates:
        if (candidate / "models").is_dir() and (candidate / "datasets").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "PMQ requires a parent directory containing models/ and datasets/."
    )


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """Load a standalone PMQ model YAML and resolve local assets."""
    source_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(source_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {source_path}")
    resolve_quantization_settings(data.get("quantization", {}))
    model = data.get("model", {})
    dataset = data.get("dataset", {})
    factors = dataset.get("factors", {})
    gptq_calibration = dataset.get("gptq_calibration", {})
    evaluations = dataset.get("evaluations", {})
    model_id = model.get("id")
    for section, value in (
        ("model.id", model_id),
        ("dataset.factors.name", factors.get("name")),
        ("dataset.gptq_calibration.name", gptq_calibration.get("name")),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"configuration is missing {section}")
    if not isinstance(evaluations, dict) or not evaluations:
        raise ValueError("configuration is missing dataset.evaluations")
    evaluation_names = {
        str(name): details.get("name")
        for name, details in evaluations.items()
        if isinstance(details, dict)
    }
    if set(evaluation_names) != set(evaluations) or any(
        not isinstance(name, str) or not name for name in evaluation_names.values()
    ):
        raise ValueError("every dataset.evaluations entry requires a dataset name")
    repository_root = source_path.parent.parent
    asset_root = _resolve_asset_root(repository_root)
    return ProtocolConfig(
        source_path=source_path,
        data=data,
        repository_root=repository_root,
        asset_root=asset_root,
        model_path=asset_root / "models" / str(model_id),
        factor_path=asset_root / "datasets" / str(factors["name"]),
        gptq_calibration_path=asset_root / "datasets" / str(gptq_calibration["name"]),
        evaluation_paths={
            key: asset_root / "datasets" / value
            for key, value in evaluation_names.items()
        },
    )


def resolve_quantization_settings(settings):
    """Validate supported PMQ controls before any model or calibration work."""
    defaults = {
        "group_size": 128, "percdamp": 0.01, "mse": True,
        "candidate_quantizer": "normal", "standard_linear_bit": 4, "router_bit": 16,
    }
    if not isinstance(settings, dict):
        raise ValueError("PMQ quantization settings must be a mapping")
    unknown = set(settings) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown PMQ quantization settings: {sorted(unknown)}")
    resolved = dict(defaults, **settings)
    if not isinstance(resolved["mse"], bool):
        raise ValueError("PMQ quantization.mse must be boolean")
    if (
        isinstance(resolved["group_size"], bool)
        or not isinstance(resolved["group_size"], int)
        or resolved["group_size"] < 1
    ):
        raise ValueError("PMQ group_size must be a positive integer")
    damping = resolved["percdamp"]
    if (
        isinstance(damping, bool)
        or not isinstance(damping, Real)
        or not math.isfinite(damping)
        or damping <= 0
    ):
        raise ValueError("PMQ percdamp must be finite and positive")
    if resolved["candidate_quantizer"] != "normal":
        raise ValueError("PMQ supports only candidate_quantizer=normal")
    for key in ("standard_linear_bit", "router_bit"):
        if isinstance(resolved[key], bool) or resolved[key] not in (1, 2, 3, 4, 8, 16):
            raise ValueError(f"Unsupported PMQ {key}: {resolved[key]}")
    return resolved


def validate_native_cli_args(args):
    """Reject native CLI placeholders instead of silently ignoring requests."""
    if args.load_quantized or args.multigpu:
        raise ValueError("Use inference.py or pmq_runner.py for checkpoint loading and model placement")
    if args.tasks or args.num_fewshot != 0:
        raise ValueError("Use pmq_runner.py --phase zeroshot for downstream evaluation")
    if args.batch_size != 1:
        raise ValueError("The original PMQ quantization entry supports only batch_size=1")
