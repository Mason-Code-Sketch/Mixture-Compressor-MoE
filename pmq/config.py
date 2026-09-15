"""Configuration loading shared by the PMQ allocation bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProtocolConfig:
    """Resolved model, dataset, and quantization settings for one PMQ run."""

    source_path: Path
    data: dict
    model_path: Path
    calibration_path: Path
    scoring_path: Path
    current_project_config: Path

    @property
    def architecture(self) -> str:
        return str(self.data["model"]["architecture"])

    @property
    def candidate_bits(self) -> tuple[int, ...]:
        return tuple(int(bit) for bit in self.data["pmq"]["candidate_bits"])


def _resolve_path(config_path: Path, value: str) -> Path:
    """Resolve one repository-relative path from the configuration location."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """Load a current-project model YAML without embedding host-specific paths."""
    source_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(source_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {source_path}")
    model = data.get("model", {})
    dataset = data.get("dataset", {})
    calibration = dataset.get("calibration", {})
    scoring = dataset.get("scoring", dataset.get("evaluation", {}))
    current_project_config = data.get("current_project_config")
    for section, value in (
        ("model.path", model.get("path")),
        ("dataset.calibration.path", calibration.get("path")),
        ("dataset.scoring.path", scoring.get("path")),
        ("current_project_config", current_project_config),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"configuration is missing {section}")
    return ProtocolConfig(
        source_path=source_path,
        data=data,
        model_path=_resolve_path(source_path, model["path"]),
        calibration_path=_resolve_path(source_path, calibration["path"]),
        scoring_path=_resolve_path(source_path, scoring["path"]),
        current_project_config=_resolve_path(source_path, current_project_config),
    )
