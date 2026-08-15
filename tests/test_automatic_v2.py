from __future__ import annotations

import json
from pathlib import Path

from graphrms.automatic_v2 import EXPECTED_RULE_SHA256, load_locked_rule

ROOT = Path(__file__).resolve().parents[1]


def test_development_rule_hash_is_exact():
    lock, rule = load_locked_rule(
        ROOT / "results/automatic_selector/development_lock.json"
    )
    assert lock["rule_sha256"] == EXPECTED_RULE_SHA256
    assert rule["version"] == "automatic-scale-v2-development"


def test_archive_replay_is_exact():
    replay = json.loads(
        (ROOT / "results/provenance/automatic_v2_archive_replay.json").read_text()
    )
    assert replay["status"] == "PASS"
    assert replay["datasets_verified"] == 9
    assert replay["development_rule_sha256"] == EXPECTED_RULE_SHA256
    development = [r for r in replay["records"] if r["dataset"] != "trento"]
    assert len(development) == 8
    assert all(r["metadata_exact"] for r in replay["records"])
    assert all(r["partition_ari_vs_archive"] == 1.0 for r in development)


def test_development_calibration_replay_is_exact():
    replay = json.loads(
        (ROOT / "results/provenance/automatic_v2_calibration_replay/"
         "development_calibration_replay.json").read_text()
    )
    assert replay["status"] == "PASS"
    assert replay["development_rule_sha256"] == EXPECTED_RULE_SHA256
    assert replay["best"] == {
        "stability_min": 0.8,
        "compression_max": 0.85,
        "beta_agreement_min": 0.9,
        "coverage": 8,
        "mean_OA": 0.574435460669256,
        "mean_BA": 0.5383551066811946,
        "mean_NMI": 0.7239183090352643,
        "mean_ARI": 0.5813199444346115,
        "structural_score": 0.6526191267349379,
    }


def test_public_wrapper_targets_v2():
    wrapper = (ROOT / "scripts/run_automatic_selector.py").read_text()
    assert "_run_automatic_v2.py" in wrapper
