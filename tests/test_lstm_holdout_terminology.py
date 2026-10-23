from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_architecture_contract_uses_held_out_test_terminology() -> None:
    source = (REPOSITORY_ROOT / "docs" / "ARCHITECTURE_CONTRACTS.md").read_text(
        encoding="utf-8"
    )

    assert "Test metrics come from the held-out chronological test split." in source
    assert "Test metrics come from the untouched chronological test split." not in source
