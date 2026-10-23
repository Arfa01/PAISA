from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts import doctor


def _package_spec(package_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        origin=str(package_root / "__init__.py"),
        submodule_search_locations=[str(package_root)],
    )


def test_node_version_contract_matches_frontend_engine_range() -> None:
    assert doctor.parse_node_version("v22.12.0") == (22, 12, 0)
    assert doctor.node_version_supported((22, 12, 0))
    assert doctor.node_version_supported((24, 0, 0))
    assert doctor.node_version_supported((26, 0, 0))
    assert not doctor.node_version_supported((22, 11, 9))
    assert not doctor.node_version_supported((23, 9, 0))
    assert not doctor.node_version_supported((25, 1, 0))
    assert doctor.parse_node_version("unknown") is None


def test_lstm_python_contract_has_both_lower_and_upper_bounds() -> None:
    assert doctor.lstm_python_supported((3, 10, 0))
    assert doctor.lstm_python_supported((3, 13, 99))
    assert not doctor.lstm_python_supported((3, 9, 99))
    assert not doctor.lstm_python_supported((3, 14, 0))


def test_missing_optional_artifacts_are_warnings_and_create_nothing(
    tmp_path: Path,
) -> None:
    before = tuple(tmp_path.iterdir())

    diagnostics = doctor.check_optional_artifacts(tmp_path)

    assert diagnostics
    assert all(item.level == "WARN" for item in diagnostics)
    assert not any(item.failed for item in diagnostics)
    assert tuple(tmp_path.iterdir()) == before


def test_missing_required_structure_fails_with_an_exact_checkout_instruction(
    tmp_path: Path,
) -> None:
    diagnostics = doctor.check_structure(tmp_path)

    assert any(item.failed for item in diagnostics)
    assert doctor.next_commands(diagnostics)[0] == (
        "Run this script from a complete PAISA repository checkout."
    )


def test_dependency_check_inspects_specs_without_importing_optional_modules(
    tmp_path: Path,
) -> None:
    requested_modules: list[str] = []

    def finder(module: str) -> object | None:
        requested_modules.append(module)
        return _package_spec(tmp_path / "src" / "paisa") if module == "paisa" else None

    def runner(
        command: tuple[str, ...], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path
        assert command[-2:] == ("pip", "check")
        return subprocess.CompletedProcess(command, 0, "No broken requirements found.\n", "")

    diagnostics = doctor.check_python_dependencies(
        tmp_path,
        runner=runner,
        finder=finder,
        version_reader=lambda _distribution: "test-version",
    )

    assert set(requested_modules) == {
        module for module, _distribution in doctor.REQUIRED_MODULES
    } | {module for module, _distribution, _purpose in doctor.OPTIONAL_MODULES}
    assert next(item for item in diagnostics if item.subject == "pip check").level == "OK"
    assert next(
        item for item in diagnostics if item.subject == "optional module: tensorflow"
    ).level == "WARN"
    assert next(
        item
        for item in diagnostics
        if item.subject == "optional workflow: FinBERT sentiment inference"
    ).level == "WARN"


def test_dependency_check_accepts_paisa_from_the_inspected_checkout(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "src" / "paisa"

    diagnostics = doctor.check_python_dependencies(
        tmp_path,
        runner=lambda command, _cwd: subprocess.CompletedProcess(
            command,
            0,
            "No broken requirements found.\n",
            "",
        ),
        finder=lambda module: _package_spec(package_root) if module == "paisa" else None,
        version_reader=lambda _distribution: "0.1.0",
    )

    paisa = next(
        item for item in diagnostics if item.subject == "Python module: paisa"
    )
    assert paisa.level == "OK"
    assert paisa.detail == "installed (0.1.0)"


def test_dependency_check_rejects_paisa_from_a_different_checkout(
    tmp_path: Path,
) -> None:
    inspected_checkout = tmp_path / "inspected"
    foreign_package_root = tmp_path / "other-checkout" / "src" / "paisa"

    diagnostics = doctor.check_python_dependencies(
        inspected_checkout,
        runner=lambda command, _cwd: subprocess.CompletedProcess(
            command,
            0,
            "No broken requirements found.\n",
            "",
        ),
        finder=lambda module: (
            _package_spec(foreign_package_root) if module == "paisa" else None
        ),
        version_reader=lambda _distribution: "0.1.0",
    )

    paisa = next(
        item for item in diagnostics if item.subject == "Python module: paisa"
    )
    assert paisa.level == "FAIL"
    assert "resolved outside inspected checkout" in paisa.detail
    assert str(foreign_package_root) in paisa.detail
    assert str(inspected_checkout / "src" / "paisa") in paisa.detail
    assert "python -m pip install --no-deps -e ." in doctor.next_commands(diagnostics)


def test_requirement_constraints_reject_an_installed_version_outside_bounds(
    tmp_path: Path,
) -> None:
    (tmp_path / "requirements.txt").write_text(
        "pandas>=2.2.0,<3.0\n",
        encoding="utf-8",
    )

    diagnostics = doctor.check_requirement_constraints(
        tmp_path,
        version_reader=lambda _distribution: "2.1.4",
    )

    assert diagnostics == [
        doctor.Diagnostic(
            "FAIL",
            "requirement: pandas",
            "installed 2.1.4; required <3.0,>=2.2.0",
        )
    ]
    assert "python -m pip install -r requirements.txt" in doctor.next_commands(
        diagnostics
    )


def test_optional_sentiment_workflow_rejects_an_out_of_range_component() -> None:
    installed_modules = {
        module for module, _distribution in doctor.REQUIRED_MODULES
    } | {module for module, _distribution, _purpose in doctor.OPTIONAL_MODULES}
    versions = {
        "tensorflow": "2.21.0",
        "torch": "2.10.0",
        "transformers": "4.99.0",
        "tokenizers": "0.23.3",
        "huggingface-hub": "1.32.0",
        "safetensors": "0.8.0",
        "pypsx-toolkit": "3.1.2",
    }

    def runner(
        command: tuple[str, ...], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == doctor.REPOSITORY_ROOT
        return subprocess.CompletedProcess(command, 0, "No broken requirements found.\n", "")

    diagnostics = doctor.check_python_dependencies(
        doctor.REPOSITORY_ROOT,
        runner=runner,
        finder=lambda module: object() if module in installed_modules else None,
        version_reader=lambda distribution: versions.get(distribution, "test-version"),
    )

    transformer = next(
        item
        for item in diagnostics
        if item.subject == "optional requirement (FinBERT): transformers"
    )
    workflow = next(
        item
        for item in diagnostics
        if item.subject == "optional workflow: FinBERT sentiment inference"
    )
    assert transformer.level == "WARN"
    assert workflow.level == "WARN"
    assert any(
        "requirements-sentiment.txt" in command
        for command in doctor.next_commands(diagnostics)
    )


def test_frontend_dependency_check_is_read_only_when_install_is_absent(
    tmp_path: Path,
) -> None:
    (tmp_path / "frontend").mkdir()

    diagnostics = doctor.check_frontend_dependencies(tmp_path)

    assert diagnostics == [
        doctor.Diagnostic(
            "WARN",
            "frontend dependencies",
            "frontend/node_modules is absent; run: cd frontend && npm ci",
        )
    ]
    assert tuple((tmp_path / "frontend").iterdir()) == ()
