#!/usr/bin/env python3
"""Read-only environment diagnostics for PAISA development.

The doctor inspects commands, dependency metadata, repository structure, and
optional local evidence. It never installs packages, creates directories, or
changes data and artifact files.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MINIMUM_PYTHON = (3, 10)
REQUIRED_COMMANDS = ("git", "bash", "node", "npm")
REQUIRED_PATHS = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-core.txt",
    "requirements-ml.txt",
    "requirements-backtesting.txt",
    "requirements-dev.txt",
    "requirements-lstm.txt",
    "requirements-sentiment.txt",
    "requirements-provider-pypsx.txt",
    "src/paisa",
    "scripts",
    "tests",
    "docs",
    "configs",
    "configs/news",
    "configs/news_coverage",
    "configs/evidence",
    "notebooks",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/src",
    "data/raw",
    "data/processed",
    "data/metadata",
    "data/external",
    "data/news",
    "models",
    "runs",
)
REQUIRED_MODULES = (
    ("paisa", "paisa-v0"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("defusedxml", "defusedxml"),
    ("dotenv", "python-dotenv"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("xgboost", "xgboost"),
    ("matplotlib", "matplotlib"),
    ("pytest", "pytest"),
    ("httpx", "httpx"),
    ("nbformat", "nbformat"),
)
OPTIONAL_MODULES = (
    ("tensorflow", "tensorflow", "price-only LSTM experiments"),
    ("torch", "torch", "FinBERT sentiment inference"),
    ("transformers", "transformers", "FinBERT sentiment inference"),
    ("tokenizers", "tokenizers", "FinBERT sentiment inference"),
    ("huggingface_hub", "huggingface-hub", "FinBERT sentiment inference"),
    ("safetensors", "safetensors", "FinBERT sentiment inference"),
    ("pypsx_toolkit", "pypsx-toolkit", "optional full-OHLCV provider"),
)
OPTIONAL_ARTIFACTS = (
    (
        "retained OGDC vertical slice",
        "artifacts/vertical_slice/VSL-20261020-OGDC-main-02",
    ),
    (
        "retained model-training run",
        "artifacts/model_training/TRN-20261020-191815-624662-nogit",
    ),
    (
        "certified price-only LSTM run",
        "artifacts/lstm_experiments/LSTM-20261024-OGDC-price-only-02",
    ),
    (
        "accepted RSS news run",
        "artifacts/news_ingestion/NEWS-20261022-131242-5ddffd2d",
    ),
    (
        "retained GDELT DOC news run",
        "artifacts/news_ingestion/NEWS-20261023-060000-gdelt-ogdc2",
    ),
    (
        "retained article-sentiment run",
        "artifacts/sentiment_inference/SENT-20261022-172211-real0002",
    ),
    (
        "retained RSS coverage-v2 pack",
        "artifacts/news_coverage/COVERAGE-20261023-120000-real0002",
    ),
    (
        "retained GDELT coverage-v2 pack",
        "artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001",
    ),
    (
        "canonical contemporaneous evidence run",
        "artifacts/contemporaneous_evidence/EVID-20261023-175847-822c5685",
    ),
    (
        "canonical contemporaneous coverage run",
        "artifacts/evidence_coverage/ECOV-20261023-175904-be50a499",
    ),
    ("local news database", "data/news/paisa_news.sqlite3"),
    ("local sentiment model cache", "data/sentiment_model_cache"),
)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    level: str
    subject: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.level == "FAIL"


CommandRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]
ModuleFinder = Callable[[str], object | None]
VersionReader = Callable[[str], str]


def _run(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _first_line(completed: subprocess.CompletedProcess[str]) -> str:
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else "no output"


def parse_node_version(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", value.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def node_version_supported(version: tuple[int, int, int]) -> bool:
    major, minor, _patch = version
    if major == 22:
        return minor >= 12
    if major == 24:
        return True
    return major >= 26


def lstm_python_supported(version: tuple[int, int, int]) -> bool:
    return (3, 10, 0) <= version < (3, 14, 0)


def check_python() -> list[Diagnostic]:
    version = sys.version_info[:3]
    supported = version >= MINIMUM_PYTHON
    details = f"{sys.executable} reports {'.'.join(str(part) for part in version)}"
    diagnostics = [
        Diagnostic("OK" if supported else "FAIL", "Python", details),
    ]
    if not lstm_python_supported(version):
        diagnostics.append(
            Diagnostic(
                "WARN",
                "LSTM Python compatibility",
                "TensorFlow 2.21 is supported here only on CPython 3.10 through 3.13.",
            )
        )
    else:
        diagnostics.append(
            Diagnostic(
                "OK",
                "LSTM Python compatibility",
                "This interpreter is within the documented CPython 3.10 through 3.13 range.",
            )
        )
    return diagnostics


def check_commands(
    repository_root: Path,
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: CommandRunner = _run,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    version_arguments = {
        "git": ("--version",),
        "bash": ("--version",),
        "node": ("--version",),
        "npm": ("--version",),
    }
    for command in REQUIRED_COMMANDS:
        location = which(command)
        if location is None:
            diagnostics.append(
                Diagnostic("FAIL", f"command: {command}", "not found on PATH")
            )
            continue
        try:
            completed = runner((command, *version_arguments[command]), repository_root)
        except (OSError, subprocess.TimeoutExpired) as exc:
            diagnostics.append(
                Diagnostic("FAIL", f"command: {command}", f"version check failed: {exc}")
            )
            continue
        level = "OK" if completed.returncode == 0 else "FAIL"
        detail = f"{location}; {_first_line(completed)}"
        if command == "node" and completed.returncode == 0:
            parsed = parse_node_version(_first_line(completed))
            if parsed is None or not node_version_supported(parsed):
                level = "FAIL"
                detail += "; requires ^22.12.0, ^24.0.0, or >=26.0.0"
        diagnostics.append(Diagnostic(level, f"command: {command}", detail))
    return diagnostics


def check_structure(repository_root: Path) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for relative_path in REQUIRED_PATHS:
        path = repository_root / relative_path
        diagnostics.append(
            Diagnostic(
                "OK" if path.exists() else "FAIL",
                f"required path: {relative_path}",
                "present" if path.exists() else "missing",
            )
        )
    return diagnostics


def _distribution_version(distribution: str, reader: VersionReader) -> str:
    try:
        return reader(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "version metadata unavailable"


def _resolved_spec_paths(spec: object, repository_root: Path) -> tuple[Path, ...]:
    """Return the local paths that define an import spec without importing it."""

    raw_paths: list[str] = []
    origin = getattr(spec, "origin", None)
    if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        raw_paths.append(origin)

    search_locations = getattr(spec, "submodule_search_locations", None)
    if search_locations is not None:
        try:
            raw_paths.extend(
                str(location) for location in search_locations if location is not None
            )
        except TypeError:
            # A malformed or synthetic spec should fail the origin check below rather
            # than make the read-only diagnostic crash.
            pass

    resolved_paths: list[Path] = []
    for raw_path in raw_paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = repository_root / candidate
        resolved = candidate.resolve()
        if resolved not in resolved_paths:
            resolved_paths.append(resolved)
    return tuple(resolved_paths)


def _paisa_checkout_resolution(
    spec: object,
    repository_root: Path,
) -> tuple[bool, str]:
    """Verify that ``paisa`` resolves from the checkout being diagnosed."""

    expected_package_root = (repository_root / "src" / "paisa").resolve()
    resolved_paths = _resolved_spec_paths(spec, repository_root)
    if not resolved_paths:
        return (
            False,
            f"import location unavailable; expected under {expected_package_root}",
        )

    foreign_paths = tuple(
        path for path in resolved_paths if not path.is_relative_to(expected_package_root)
    )
    if foreign_paths:
        locations = ", ".join(str(path) for path in foreign_paths)
        return (
            False,
            f"resolved outside inspected checkout: {locations}; "
            f"expected under {expected_package_root}",
        )

    locations = ", ".join(str(path) for path in resolved_paths)
    return True, f"resolved from inspected checkout: {locations}"


def _requirement_lines(path: Path, seen: set[Path] | None = None) -> list[str]:
    resolved = path.resolve()
    visited = seen if seen is not None else set()
    if resolved in visited or not resolved.is_file():
        return []
    visited.add(resolved)

    requirements: list[str] = []
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r ") or line.startswith("--requirement "):
            include = line.split(maxsplit=1)[1]
            requirements.extend(_requirement_lines(resolved.parent / include, visited))
            continue
        if not line.startswith("-"):
            requirements.append(line)
    return requirements


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def check_requirement_constraints(
    repository_root: Path,
    *,
    requirements_filename: str = "requirements.txt",
    include_distributions: set[str] | None = None,
    problem_level: str = "FAIL",
    entry_prefix: str = "requirement",
    summary_subject: str = "PAISA requirement constraints",
    version_reader: VersionReader = importlib.metadata.version,
) -> list[Diagnostic]:
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        return [
            Diagnostic(
                problem_level,
                summary_subject,
                "packaging is unavailable; installed versions could not be checked against "
                + requirements_filename,
            )
        ]

    requirements_file = repository_root / requirements_filename
    if not requirements_file.is_file():
        return [
            Diagnostic(
                problem_level,
                summary_subject,
                f"{requirements_filename} is missing",
            )
        ]

    included = (
        {_normalized_distribution_name(item) for item in include_distributions}
        if include_distributions is not None
        else None
    )
    diagnostics: list[Diagnostic] = []
    for raw_requirement in _requirement_lines(requirements_file):
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement as exc:
            diagnostics.append(
                Diagnostic(
                    problem_level,
                    f"{entry_prefix}: {raw_requirement}",
                    f"invalid committed requirement: {exc}",
                )
            )
            continue
        if (
            included is not None
            and _normalized_distribution_name(requirement.name) not in included
        ):
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed_version = version_reader(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            diagnostics.append(
                Diagnostic(
                    problem_level,
                    f"{entry_prefix}: {requirement.name}",
                    "not installed",
                )
            )
            continue
        compatible = requirement.specifier.contains(
            installed_version,
            prereleases=True,
        )
        required = str(requirement.specifier) or "any version"
        diagnostics.append(
            Diagnostic(
                "OK" if compatible else problem_level,
                f"{entry_prefix}: {requirement.name}",
                f"installed {installed_version}; required {required}",
            )
        )
    return diagnostics


def check_python_dependencies(
    repository_root: Path,
    *,
    runner: CommandRunner = _run,
    finder: ModuleFinder = importlib.util.find_spec,
    version_reader: VersionReader = importlib.metadata.version,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    try:
        completed = runner((sys.executable, "-m", "pip", "check"), repository_root)
    except (OSError, subprocess.TimeoutExpired) as exc:
        diagnostics.append(Diagnostic("FAIL", "pip check", str(exc)))
    else:
        diagnostics.append(
            Diagnostic(
                "OK" if completed.returncode == 0 else "FAIL",
                "pip check",
                _first_line(completed),
            )
        )

    for module, distribution in REQUIRED_MODULES:
        spec = finder(module)
        available = spec is not None
        detail = (
            f"installed ({_distribution_version(distribution, version_reader)})"
            if available
            else "missing"
        )
        level = "OK" if available else "FAIL"
        if module == "paisa" and spec is not None:
            belongs_to_checkout, resolution_detail = _paisa_checkout_resolution(
                spec,
                repository_root,
            )
            if not belongs_to_checkout:
                level = "FAIL"
                detail = f"{detail}; {resolution_detail}"
        diagnostics.append(
            Diagnostic(level, f"Python module: {module}", detail)
        )

    optional_presence: dict[str, bool] = {}
    for module, distribution, purpose in OPTIONAL_MODULES:
        available = finder(module) is not None
        optional_presence[module] = available
        detail = (
            f"installed ({_distribution_version(distribution, version_reader)}); component for {purpose}"
            if available
            else f"not installed; needed only for {purpose}"
        )
        diagnostics.append(
            Diagnostic("OK" if available else "WARN", f"optional module: {module}", detail)
        )

    sentiment_modules = (
        "torch",
        "transformers",
        "tokenizers",
        "huggingface_hub",
        "safetensors",
    )
    missing_sentiment = [
        module for module in sentiment_modules if not optional_presence[module]
    ]
    lstm_constraints = (
        check_requirement_constraints(
            repository_root,
            requirements_filename="requirements-lstm.txt",
            include_distributions={"tensorflow"},
            problem_level="WARN",
            entry_prefix="optional requirement (LSTM)",
            summary_subject="optional LSTM requirement constraints",
            version_reader=version_reader,
        )
        if optional_presence["tensorflow"]
        else []
    )
    sentiment_constraints = (
        check_requirement_constraints(
            repository_root,
            requirements_filename="requirements-sentiment.txt",
            include_distributions={
                "torch",
                "transformers",
                "tokenizers",
                "huggingface-hub",
                "safetensors",
            },
            problem_level="WARN",
            entry_prefix="optional requirement (FinBERT)",
            summary_subject="optional FinBERT requirement constraints",
            version_reader=version_reader,
        )
        if any(optional_presence[module] for module in sentiment_modules)
        else []
    )
    provider_constraints = (
        check_requirement_constraints(
            repository_root,
            requirements_filename="requirements-provider-pypsx.txt",
            include_distributions={"pypsx-toolkit"},
            problem_level="WARN",
            entry_prefix="optional requirement (pypsx)",
            summary_subject="optional pypsx requirement constraints",
            version_reader=version_reader,
        )
        if optional_presence["pypsx_toolkit"]
        else []
    )
    diagnostics.extend(lstm_constraints)
    diagnostics.extend(sentiment_constraints)
    diagnostics.extend(provider_constraints)

    lstm_ready = optional_presence["tensorflow"] and all(
        item.level == "OK" for item in lstm_constraints
    )
    diagnostics.append(
        Diagnostic(
            "OK" if lstm_ready else "WARN",
            "optional workflow: price-only LSTM experiments",
            (
                "runtime component and version constraint are satisfied"
                if lstm_ready
                else "unavailable or incompatible; inspect the optional LSTM module and requirement checks"
            ),
        )
    )
    sentiment_ready = not missing_sentiment and all(
        item.level == "OK" for item in sentiment_constraints
    )
    diagnostics.append(
        Diagnostic(
            "OK" if sentiment_ready else "WARN",
            "optional workflow: FinBERT sentiment inference",
            (
                "all optional runtime components and version constraints are satisfied"
                if sentiment_ready
                else (
                    "unavailable; missing components: " + ", ".join(missing_sentiment)
                    if missing_sentiment
                    else "unavailable; one or more version constraints are not satisfied"
                )
            ),
        )
    )
    provider_ready = optional_presence["pypsx_toolkit"] and all(
        item.level == "OK" for item in provider_constraints
    )
    diagnostics.append(
        Diagnostic(
            "OK" if provider_ready else "WARN",
            "optional workflow: pypsx provider",
            (
                "runtime component and version constraint are satisfied"
                if provider_ready
                else "unavailable or incompatible; inspect the optional pypsx module and requirement checks"
            ),
        )
    )
    return diagnostics


def check_frontend_dependencies(
    repository_root: Path,
    *,
    runner: CommandRunner = _run,
) -> list[Diagnostic]:
    node_modules = repository_root / "frontend" / "node_modules"
    if not node_modules.is_dir():
        return [
            Diagnostic(
                "WARN",
                "frontend dependencies",
                "frontend/node_modules is absent; run: cd frontend && npm ci",
            )
        ]
    try:
        completed = runner(
            ("npm", "--prefix", "frontend", "ls", "--depth=0"), repository_root
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [Diagnostic("FAIL", "frontend dependencies", str(exc))]
    return [
        Diagnostic(
            "OK" if completed.returncode == 0 else "FAIL",
            "frontend dependencies",
            _first_line(completed),
        )
    ]


def check_optional_artifacts(repository_root: Path) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for label, relative_path in OPTIONAL_ARTIFACTS:
        present = (repository_root / relative_path).exists()
        diagnostics.append(
            Diagnostic(
                "OK" if present else "WARN",
                f"optional artifact: {label}",
                f"{'present' if present else 'absent'}: {relative_path}",
            )
        )
    return diagnostics


def next_commands(diagnostics: Iterable[Diagnostic]) -> tuple[str, ...]:
    items = tuple(diagnostics)
    subjects = {item.subject for item in items if item.failed}
    commands: list[str] = []
    if any(subject.startswith("required path:") for subject in subjects):
        commands.append("Run this script from a complete PAISA repository checkout.")
    if any(subject in {"command: node", "command: npm"} for subject in subjects):
        commands.append(
            "Install Node.js 22.20.0 (including npm), then rerun: python scripts/doctor.py"
        )
    missing_shell_commands = sorted(
        subject.removeprefix("command: ")
        for subject in subjects
        if subject in {"command: git", "command: bash"}
    )
    if missing_shell_commands:
        commands.append(
            "Install the required command(s) "
            + ", ".join(missing_shell_commands)
            + ", then rerun: python scripts/doctor.py"
        )
    if (
        "Python" in subjects
        or "pip check" in subjects
        or "PAISA requirement constraints" in subjects
        or any(
            subject.startswith("Python module:")
            or subject.startswith("requirement:")
            for subject in subjects
        )
    ):
        commands.extend(
            (
                "python3 -m venv .venv",
                "source .venv/bin/activate",
                "python -m pip install --upgrade pip",
                "python -m pip install -r requirements.txt",
                "python -m pip install --no-deps -e .",
            )
        )
    if any(item.subject == "frontend dependencies" and item.level != "OK" for item in items):
        commands.append("cd frontend && npm ci")
    optional_repairs = (
        ("optional requirement (LSTM)", "python -m pip install -r requirements-lstm.txt"),
        (
            "optional requirement (FinBERT)",
            "python -m pip install -r requirements-sentiment.txt",
        ),
        (
            "optional requirement (pypsx)",
            "python -m pip install -r requirements-provider-pypsx.txt",
        ),
    )
    for subject_prefix, command in optional_repairs:
        if any(
            item.subject.startswith(subject_prefix) and item.level != "OK"
            for item in items
        ):
            commands.append(command)
    if not commands:
        commands.append("Environment prerequisites look ready; run the validation matrix below.")
    commands.extend(
        (
            "python -m pip check",
            "python -m pytest",
            "python -m compileall -q src scripts tests",
            "(cd frontend && npm run test && npm run lint && npm run build)",
            "git diff --check",
            "git status --short",
        )
    )
    return tuple(dict.fromkeys(commands))


def collect_diagnostics(
    repository_root: Path,
    *,
    runner: CommandRunner = _run,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(check_python())
    diagnostics.extend(check_commands(repository_root, runner=runner))
    diagnostics.extend(check_structure(repository_root))
    diagnostics.extend(check_requirement_constraints(repository_root))
    diagnostics.extend(check_python_dependencies(repository_root, runner=runner))
    diagnostics.extend(check_frontend_dependencies(repository_root, runner=runner))
    diagnostics.extend(check_optional_artifacts(repository_root))
    return diagnostics


def _print_section(title: str, diagnostics: Iterable[Diagnostic]) -> None:
    print(f"\n{title}")
    for item in diagnostics:
        print(f"[{item.level}] {item.subject}: {item.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only PAISA environment diagnostics. No package, data, model, "
            "or artifact is installed, downloaded, created, or changed."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="PAISA checkout to inspect (default: repository containing this script).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repository_root = arguments.repository_root.expanduser().resolve()
    diagnostics = collect_diagnostics(repository_root)

    print("PAISA environment doctor (read-only)")
    print(f"Repository: {repository_root}")
    _print_section("Diagnostics", diagnostics)
    print("\nExact next commands")
    for command in next_commands(diagnostics):
        print(f"  {command}")

    failures = sum(item.failed for item in diagnostics)
    warnings = sum(item.level == "WARN" for item in diagnostics)
    print(f"\nSummary: {failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
