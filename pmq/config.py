"""Configuration loading for standalone PMQ experiments."""

from __future__ import annotations

from dataclasses import dataclass
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
    calibration_path: Path
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
    model = data.get("model", {})
    dataset = data.get("dataset", {})
    calibration = dataset.get("calibration", {})
    evaluations = dataset.get("evaluations", {})
    model_id = model.get("id")
    for section, value in (
        ("model.id", model_id),
        ("dataset.calibration.name", calibration.get("name")),
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
        calibration_path=asset_root / "datasets" / str(calibration["name"]),
        evaluation_paths={
            key: asset_root / "datasets" / value
            for key, value in evaluation_names.items()
        },
    )
