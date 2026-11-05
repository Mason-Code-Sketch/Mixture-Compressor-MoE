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


def _validate_source_inheritance(recorded, expected):
    """Accept only an explicit, content-bound validation receipt for this origin."""
    link = recorded.get("source_inheritance")
    if not isinstance(link, dict) or set(link) != {"receipt", "sha256"}:
        raise ValueError("PMQ source revision changed; verify artifacts before reuse")
    path = Path(link["receipt"])
    if sha256_file(path) != link["sha256"]:
        raise ValueError("PMQ source inheritance receipt hash mismatch")
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") != 1 or receipt.get("status") != "verified":
        raise ValueError("PMQ source inheritance receipt is not verified")
    if (receipt.get("from_commit") != recorded["source_commit"]
            or receipt.get("to_commit") != expected["source_commit"]):
        raise ValueError("PMQ source inheritance revisions differ")
    for key in ("seed", "model_config_sha256", "protocol_config_sha256"):
        if receipt.get(key) != expected[key]:
            raise ValueError(f"PMQ source inheritance mismatch: {key}")
    origin = {key: value for key, value in recorded.items() if key != "source_inheritance"}
    origin_hash = hashlib.sha256(json.dumps(origin, sort_keys=True).encode()).hexdigest()
    if receipt.get("origin_provenance_sha256") != origin_hash:
        raise ValueError("PMQ source inheritance origin differs")
    files = receipt.get("preserved_files", {})
    if not files or any(sha256_file(name) != digest for name, digest in files.items()):
        raise ValueError("PMQ source inheritance artifact hash mismatch")
    proof_path = Path(receipt["validation_receipt"])
    if sha256_file(proof_path) != receipt["validation_receipt_sha256"]:
        raise ValueError("PMQ source inheritance validation hash mismatch")
    proof = json.loads(proof_path.read_text())
    if not all(proof.get(key) is True for key in ("accepted", "captures_exact",
            "real_layer_native_weights_exact", "real_layer_packed_states_exact",
            "combined_peak_not_increased")):
        raise ValueError("PMQ source inheritance validation did not pass")
    if proof.get("source_commit") != receipt.get("validated_capture_commit"):
        raise ValueError("PMQ source inheritance validation revision differs")
    if proof.get("capture_speedup", 0) <= 1 or proof.get("quantization_speedup", 0) <= 1:
        raise ValueError("PMQ source inheritance speed validation did not pass")


def validate_provenance(protocol, recorded, *, seed):
    """Reject foreign provenance; require a pinned receipt for verified source reuse."""
    expected = provenance(protocol, seed=seed)
    for key in ("seed", "model_config_sha256", "protocol_config_sha256"):
        if recorded.get(key) != expected[key]:
            raise ValueError(f"PMQ provenance mismatch: {key}")
    if recorded.get("source_commit") and expected["source_commit"] != recorded["source_commit"]:
        _validate_source_inheritance(recorded, expected)
