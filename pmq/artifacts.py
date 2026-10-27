"""Small provenance sidecars accompanying the upstream PMQ artifact files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
import subprocess
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, default=str, allow_nan=False)
        stream.write("\n")


def write_pickle(path, payload):
    with Path(path).open("xb") as stream:
        pickle.dump(payload, stream)


def provenance(protocol, *, seed):
    result = subprocess.run(
        ["git", "-C", str(protocol.repository_root), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    versions = {}
    for name in ("torch", "transformers", "accelerate", "datasets", "lm_eval", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"seed": int(seed), "source_commit": result.stdout.strip() if not result.returncode else None,
            "protocol_config": protocol.data, "protocol_config_sha256": sha256_file(protocol.source_path),
            "model_path": str(protocol.model_path), "model_config_sha256": sha256_file(protocol.model_path / "config.json"),
            "versions": versions}


def validate_provenance(protocol, recorded, *, seed):
    """Reject reuse from another model, protocol, seed or source revision."""
    expected = provenance(protocol, seed=seed)
    for key in ("seed", "model_config_sha256", "protocol_config_sha256"):
        if recorded.get(key) != expected[key]:
            raise ValueError(f"PMQ provenance mismatch: {key}")
    if recorded.get("source_commit") and expected["source_commit"] != recorded["source_commit"]:
        raise ValueError("PMQ source revision changed; verify artifacts before reuse")
