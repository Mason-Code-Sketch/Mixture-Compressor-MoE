"""Explicit bridge to the adjacent MoE-PTQ repository.

The bridge keeps PMQ's allocation method in this repository while importing
the model adapters and data protocol from the current project.  Repository
locations are derived from this file, never embedded as machine-specific
absolute paths.
"""

from __future__ import annotations

import sys
from pathlib import Path


def current_project_root() -> Path:
    """Return the sibling MoE-PTQ repository required by this protocol."""
    repository = Path(__file__).resolve().parents[1]
    project = repository.parent.parent / "moe-ptq"
    if not project.is_dir():
        raise FileNotFoundError(
            "expected the current MoE-PTQ repository beside third-party; "
            f"missing {project}"
        )
    return project


def enable_current_project_imports() -> Path:
    """Expose the current project's public Python modules to the PMQ runner."""
    project = current_project_root()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    return project
