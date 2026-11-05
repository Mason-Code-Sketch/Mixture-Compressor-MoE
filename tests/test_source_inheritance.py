import copy
import hashlib
import json

import pytest

from pmq import artifacts


def prepare(tmp_path, monkeypatch):
    origin = {"seed": 0, "model_config_sha256": "model", "protocol_config_sha256": "config", "source_commit": "old", "versions": {"torch": "same"}}
    expected = dict(origin, source_commit="new")
    monkeypatch.setattr(artifacts, "provenance", lambda *args, **kwargs: expected)
    data = tmp_path / "factors.pkl"
    data.write_bytes(b"valid factor data")
    proof = tmp_path / "validation.json"
    proof.write_text(json.dumps({"accepted": True, "captures_exact": True,
        "real_layer_native_weights_exact": True, "real_layer_packed_states_exact": True,
        "combined_peak_not_increased": True, "capture_speedup": 2., "quantization_speedup": 1.4,
        "source_commit": "validated"}))
    receipt = {"schema_version": 1, "status": "verified", "from_commit": "old", "to_commit": "new",
        "seed": 0, "model_config_sha256": "model", "protocol_config_sha256": "config",
        "origin_provenance_sha256": hashlib.sha256(json.dumps(origin, sort_keys=True).encode()).hexdigest(),
        "preserved_files": {str(data): artifacts.sha256_file(data)},
        "validation_receipt": str(proof), "validation_receipt_sha256": artifacts.sha256_file(proof),
        "validated_capture_commit": "validated"}
    path = tmp_path / "inheritance.json"
    path.write_text(json.dumps(receipt))
    recorded = dict(origin, source_inheritance={"receipt": str(path), "sha256": artifacts.sha256_file(path)})
    return recorded, expected, data, proof, path


def test_verified_origin_is_preserved_and_accepted(tmp_path, monkeypatch):
    recorded, _, _, _, _ = prepare(tmp_path, monkeypatch)
    before = copy.deepcopy(recorded)
    artifacts.validate_provenance(None, recorded, seed=0)
    assert recorded == before and recorded["source_commit"] == "old"


@pytest.mark.parametrize("damage", ["no_receipt", "origin", "source", "config", "artifact", "receipt", "proof", "unaccepted"])
def test_invalid_inheritance_cannot_bypass_checks(tmp_path, monkeypatch, damage):
    recorded, expected, data, proof, path = prepare(tmp_path, monkeypatch)
    if damage == "no_receipt": recorded.pop("source_inheritance")
    elif damage == "origin": recorded["versions"]["torch"] = "changed"
    elif damage == "source": expected["source_commit"] = "third"
    elif damage == "config": recorded["protocol_config_sha256"] = "changed"
    elif damage == "artifact": data.write_bytes(b"changed")
    elif damage == "receipt": path.write_text("{}")
    elif damage == "proof": proof.write_text("{}")
    else:
        value = json.loads(proof.read_text()); value["accepted"] = False; proof.write_text(json.dumps(value))
        receipt = json.loads(path.read_text()); receipt["validation_receipt_sha256"] = artifacts.sha256_file(proof); path.write_text(json.dumps(receipt))
        recorded["source_inheritance"]["sha256"] = artifacts.sha256_file(path)
    with pytest.raises(ValueError): artifacts.validate_provenance(None, recorded, seed=0)
