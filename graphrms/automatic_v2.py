"""Frozen automatic-v2 selector used by the reported Graph-RMS audit.

The candidate endpoint surfaces are generated without reference labels by the
archived automatic-scale-v1 candidate generator.  Automatic-v2 then applies
the globally calibrated, frozen stability/compression rule to those surfaces.
This module is the recovered implementation used for the eight development
scenes and the retrospective Trento transfer audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score


DEVELOPMENT_DATASETS = [
    "salinas_a",
    "indian_pines",
    "ksc",
    "pavia_university",
    "whu_hi_longkou",
    "whu_hi_honghu",
    "whu_hi_hanchuan",
    "botswana",
]
RETROSPECTIVE_DATASET = "trento"
ALL_DATASETS = DEVELOPMENT_DATASETS + [RETROSPECTIVE_DATASET]
EXPECTED_RULE_SHA256 = (
    "87f894b50b18edd5d1ee31d5b1b856b2a27752beb0416206523bcfc9db67bb59"
)


def canonical_rule_sha256(rule: dict[str, Any]) -> str:
    """Hash a rule using the exact canonicalisation used in development."""
    payload = json.dumps(rule, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_locked_rule(lock_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and cryptographically verify the frozen automatic-v2 rule."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    rule = lock["rule"]
    observed = canonical_rule_sha256(rule)
    recorded = lock.get("rule_sha256")
    if observed != EXPECTED_RULE_SHA256 or recorded != EXPECTED_RULE_SHA256:
        raise RuntimeError(
            "automatic-v2 development rule does not match the reported lock: "
            f"observed={observed}, recorded={recorded}"
        )
    return lock, rule


def load_candidates(cache_dir: Path, checkpoints: tuple[int, ...] = (25, 50, 100, 200)):
    """Load the label-free endpoint candidates saved by the v1 generator."""
    candidates: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        path = cache_dir / f"checkpoint_T{checkpoint}_endpoints.npz"
        if not path.exists():
            raise RuntimeError(f"missing endpoint cache: {path}")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            labels = archive["labels"]
            for meta, partition in zip(metadata, labels):
                candidates.append(
                    {**meta, "labels": partition.astype(np.int32, copy=False)}
                )
    for index, candidate in enumerate(candidates):
        candidate["_candidate_index"] = index
    return candidates


def candidate_relationships(candidates: list[dict[str, Any]]):
    """Calculate the adjacent-scale and cross-beta ARIs used by v2."""
    edge_ari: dict[tuple[int, int], float] = {}

    def ari(a: dict[str, Any], b: dict[str, Any]) -> float:
        key = tuple(sorted((a["_candidate_index"], b["_candidate_index"])))
        if key not in edge_ari:
            edge_ari[key] = float(adjusted_rand_score(a["labels"], b["labels"]))
        return edge_ari[key]

    for candidate in candidates:
        checkpoint_peers = [
            peer
            for peer in candidates
            if peer["size_exponent"] == candidate["size_exponent"]
            and peer["gamma_index"] == candidate["gamma_index"]
            and abs(peer["checkpoint_index"] - candidate["checkpoint_index"]) == 1
        ]
        gamma_peers = [
            peer
            for peer in candidates
            if peer["size_exponent"] == candidate["size_exponent"]
            and peer["checkpoint_index"] == candidate["checkpoint_index"]
            and abs(peer["gamma_index"] - candidate["gamma_index"]) == 1
        ]
        beta_peers = [
            peer
            for peer in candidates
            if peer["checkpoint"] == candidate["checkpoint"]
            and peer["gamma"] == candidate["gamma"]
            and peer["size_exponent"] != candidate["size_exponent"]
        ]
        candidate["T_stability"] = max(
            [ari(candidate, peer) for peer in checkpoint_peers] or [0.0]
        )
        candidate["gamma_stability"] = max(
            [ari(candidate, peer) for peer in gamma_peers] or [0.0]
        )
        candidate["beta_agreement"] = max(
            [ari(candidate, peer) for peer in beta_peers] or [0.0]
        )
    return edge_ari


def select_v2(
    candidates: list[dict[str, Any]],
    edge_ari: dict[tuple[int, int], float],
    rule: dict[str, Any],
):
    """Apply the exact frozen automatic-v2 component-selection hierarchy."""
    stability = float(rule["chosen_stability_min"])
    compression_max = float(rule["chosen_compression_max"])
    beta_min = float(rule["chosen_beta_agreement_min"])
    good = [
        candidate
        for candidate in candidates
        if candidate["eligible"]
        and candidate["compression"] <= compression_max
        and max(candidate["T_stability"], candidate["gamma_stability"])
        >= stability
        and (
            candidate["size_exponent"] == 0.0
            or candidate["beta_agreement"] >= beta_min
        )
    ]

    components: list[list[dict[str, Any]]] = []
    seen: set[int] = set()
    for candidate in good:
        index = candidate["_candidate_index"]
        if index in seen:
            continue
        stack = [candidate]
        component: list[dict[str, Any]] = []
        seen.add(index)
        while stack:
            current = stack.pop()
            component.append(current)
            for peer in good:
                peer_index = peer["_candidate_index"]
                if (
                    peer_index in seen
                    or peer["size_exponent"] != current["size_exponent"]
                ):
                    continue
                adjacent = (
                    peer["gamma_index"] == current["gamma_index"]
                    and abs(peer["checkpoint_index"] - current["checkpoint_index"])
                    == 1
                ) or (
                    peer["checkpoint_index"] == current["checkpoint_index"]
                    and abs(peer["gamma_index"] - current["gamma_index"]) == 1
                )
                key = tuple(sorted((current["_candidate_index"], peer_index)))
                if adjacent and edge_ari.get(key, 0.0) >= stability:
                    seen.add(peer_index)
                    stack.append(peer)
        if len(component) >= 2:
            components.append(component)

    if not components:
        return None, {"eligible_candidates": len(good), "stable_components": 0}

    best_by_beta: dict[float, tuple[int, float, dict[str, Any]]] = {}
    for beta in (0.0, 0.5):
        pools = [
            component
            for component in components
            if component[0]["size_exponent"] == beta
        ]
        if not pools:
            continue
        component = max(
            pools,
            key=lambda values: (
                len(values),
                np.mean(
                    [
                        max(item["T_stability"], item["gamma_stability"])
                        for item in values
                    ]
                ),
            ),
        )
        centre_t = np.median([item["checkpoint_index"] for item in component])
        centre_g = np.median([item["gamma_index"] for item in component])
        chosen = max(
            component,
            key=lambda item: (
                max(item["T_stability"], item["gamma_stability"]),
                -abs(item["checkpoint_index"] - centre_t)
                - abs(item["gamma_index"] - centre_g),
                -item["compression"],
            ),
        )
        best_by_beta[beta] = (
            len(component),
            float(
                np.mean(
                    [
                        max(item["T_stability"], item["gamma_stability"])
                        for item in component
                    ]
                )
            ),
            chosen,
        )

    if 0.0 in best_by_beta and 0.5 in best_by_beta:
        conservative, size_aware = best_by_beta[0.0], best_by_beta[0.5]
        chosen = (
            size_aware[2]
            if (
                size_aware[0] > conservative[0]
                or (
                    size_aware[0] == conservative[0]
                    and size_aware[1] >= conservative[1] + 0.01
                )
            )
            else conservative[2]
        )
    else:
        chosen = next(iter(best_by_beta.values()))[2]

    audit = {
        "eligible_candidates": len(good),
        "stable_components": len(components),
        "component_support_by_beta": {
            str(beta): value[0] for beta, value in best_by_beta.items()
        },
        "component_mean_stability_by_beta": {
            str(beta): value[1] for beta, value in best_by_beta.items()
        },
    }
    return chosen, audit


def select_from_cache(cache_dir: Path, rule: dict[str, Any]):
    """Convenience entry point used by the public runner and audit tool."""
    candidates = load_candidates(cache_dir)
    relationships = candidate_relationships(candidates)
    return select_v2(candidates, relationships, rule)
