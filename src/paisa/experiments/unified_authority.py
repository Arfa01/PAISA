"""Authoritative source identities for the certified unified comparison pack.

This module is intentionally dependency-free so both the offline comparison
verifier and the read-only API repository can enforce one source allowlist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final


AUTHORITATIVE_SOURCE_ROOTS: Final[dict[str, dict[str, str]]] = {
    "baseline_training": {
        "run_id": "TRN-20261020-191815-624662-nogit",
        "manifest_sha256": (
            "8d5113bc8b67f43a76639d7397b905217bda1f94321183f0438e33448f689023"
        ),
    },
    "baseline_vertical_slice": {
        "run_id": "VSL-20261020-OGDC-main-02",
        "manifest_sha256": (
            "2ce2f7581a5baff8487b09ae695213745c704f11715e6bd5abd0e99009337f31"
        ),
        "portable_certificate_sha256": (
            "e948409e679061d0c503171be64b76d9d847a2161cb70f7baa5737089e19c314"
        ),
    },
    "lstm": {
        "run_id": "LSTM-20261024-OGDC-price-only-02",
        "manifest_sha256": (
            "e475e077f375071c79ce8743425831d8cd9cf5ceb37f42eca0acfa65a100f3d0"
        ),
    },
}


AUTHORITATIVE_COMPARISON_MANIFESTS: Final[dict[str, str]] = {
    "CMP-20261024-OGDC-price-only-02": (
        "e4b0b12bb565fa6df1f23b7e8855f84569ffcefb795c36c4bfeb4a1e2c82849d"
    ),
}


def expected_evidence_bindings(
    model_ids: Mapping[str, str],
    *,
    portfolio_files: Sequence[str],
    source_authority: Mapping[str, Mapping[str, str]] = AUTHORITATIVE_SOURCE_ROOTS,
) -> dict[str, tuple[str, str, str]]:
    """Build the canonical source-to-pack evidence binding contract."""

    variants = tuple(f"M-{index:03d}" for index in range(1, 7))
    if set(model_ids) != set(variants):
        raise ValueError("Evidence bindings require exactly M-001 through M-006")
    bindings: dict[str, tuple[str, str, str]] = {
        "evidence/source_manifests/baseline_training_run_manifest.json": (
            "baseline_training",
            "run_manifest.json",
            "source_integrity_root",
        ),
        "evidence/source_manifests/lstm_run_manifest.json": (
            "lstm",
            "run_manifest.json",
            "source_integrity_root",
        ),
        "evidence/source_manifests/lstm_artifact_manifest.json": (
            "lstm",
            "artifact_manifest.json",
            "source_payload_inventory",
        ),
        "evidence/splits/baseline_split_manifest.json": (
            "baseline_training",
            "split_manifest.json",
            "chronological_split_membership",
        ),
        "evidence/splits/lstm_split_manifest.json": (
            "lstm",
            "split_manifest.json",
            "lstm_sequence_split_membership",
        ),
        "evidence/market_data/baseline_psx_prices.csv": (
            "baseline_vertical_slice",
            "input_snapshot/psx_prices.csv",
            "portfolio_market_observations",
        ),
        "evidence/market_data/lstm_ogdc_prices.csv": (
            "lstm",
            "frozen_input/OGDC_prices.csv",
            "portfolio_market_observations",
        ),
    }
    for variant in variants:
        source_role = "baseline_training" if variant in variants[:4] else "lstm"
        bindings[f"evidence/predictions/{variant}/predictions_all_splits.csv"] = (
            source_role,
            f"models/{model_ids[variant]}/predictions_all_splits.csv",
            "model_prediction_rows",
        )
        if variant in variants[:4]:
            run_id = source_authority["baseline_vertical_slice"]["run_id"]
            source_prefix = f"experiments/{run_id}-{variant.lower()}"
            source_role = "baseline_vertical_slice"
        else:
            run_id = source_authority["lstm"]["run_id"]
            source_prefix = f"backtests/{run_id}-{variant.lower()}"
            source_role = "lstm"
        for relative in portfolio_files:
            bindings[f"evidence/portfolios/{variant}/{relative}"] = (
                source_role,
                f"{source_prefix}/{relative}",
                (
                    "portfolio_configuration"
                    if relative == "config.json"
                    else "portfolio_ledger"
                ),
            )
    return bindings
