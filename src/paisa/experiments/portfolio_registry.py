"""Registry for immutable experiment identities and their independent ledgers."""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path

from paisa.trading.models import ExperimentConfig
from paisa.trading.portfolio import PortfolioLedger


@dataclass(frozen=True, slots=True)
class RegisteredPortfolios:
    """The three independent ledgers owned by one registered experiment."""

    strategy: PortfolioLedger
    cash_only: PortfolioLedger
    buy_and_hold: PortfolioLedger


class PortfolioRegistry:
    """Prevent experiment and portfolio identity reuse across registered runs."""

    def __init__(self, runs_root: str | Path) -> None:
        self.runs_root = Path(runs_root)
        self._lock = threading.RLock()
        self._experiment_ids: set[str] = set()
        self._portfolio_ids: set[str] = set()
        self._registered: dict[str, RegisteredPortfolios] = {}
        self._reservation_tokens: dict[str, str] = {}
        self._scan_completed_runs()

    def validate_available(self, config: ExperimentConfig) -> None:
        """Raise if the config's experiment or portfolio IDs are already used."""

        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        with self._lock:
            self._scan_completed_runs()
            self._validate_ids_available(config)

    def register(self, config: ExperimentConfig) -> RegisteredPortfolios:
        """Reserve all IDs, then construct a fresh strategy and benchmark ledger."""

        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")

        with self._lock:
            self._scan_completed_runs()
            self._validate_ids_available(config)

            portfolio_ids = (
                config.portfolio_id,
                config.benchmark_portfolio_ids.cash_only,
                config.benchmark_portfolio_ids.buy_and_hold,
            )
            token = secrets.token_hex(16)
            self._create_disk_reservations(config, token)
            try:
                # Update both identity sets together before constructing any ledger.
                self._experiment_ids.add(config.experiment_id)
                self._portfolio_ids.update(portfolio_ids)
                portfolios = RegisteredPortfolios(
                    strategy=PortfolioLedger(config.portfolio_id, config.initial_cash),
                    cash_only=PortfolioLedger(
                        config.benchmark_portfolio_ids.cash_only, config.initial_cash
                    ),
                    buy_and_hold=PortfolioLedger(
                        config.benchmark_portfolio_ids.buy_and_hold, config.initial_cash
                    ),
                )
            except Exception:
                # A failed construction must not leave a partial reservation.
                self._experiment_ids.discard(config.experiment_id)
                self._portfolio_ids.difference_update(portfolio_ids)
                self._remove_disk_reservations(config, token)
                raise
            self._reservation_tokens[config.experiment_id] = token
            self._registered[config.experiment_id] = portfolios
            return portfolios

    def release(self, config: ExperimentConfig) -> None:
        """Release this process's reservations after a failed unpublished run."""

        with self._lock:
            token = self._reservation_tokens.pop(config.experiment_id, None)
            if token is None:
                return
            self._remove_disk_reservations(config, token)
            self._registered.pop(config.experiment_id, None)
            self._experiment_ids.discard(config.experiment_id)
            self._portfolio_ids.difference_update(self._config_portfolio_ids(config))

    def finalize(self, config: ExperimentConfig) -> None:
        """Remove transient reservations after the immutable directory exists."""

        with self._lock:
            token = self._reservation_tokens.pop(config.experiment_id, None)
            if token is not None:
                self._remove_disk_reservations(config, token)

    def _validate_ids_available(self, config: ExperimentConfig) -> None:
        if config.experiment_id in self._experiment_ids:
            raise ValueError(f"experiment_id is already registered: {config.experiment_id}")
        requested_portfolio_ids = {
            config.portfolio_id,
            config.benchmark_portfolio_ids.cash_only,
            config.benchmark_portfolio_ids.buy_and_hold,
        }
        duplicate_ids = requested_portfolio_ids & self._portfolio_ids
        if duplicate_ids:
            duplicate = sorted(duplicate_ids)[0]
            raise ValueError(f"portfolio_id is already registered: {duplicate}")

    def _scan_completed_runs(self) -> None:
        """Refresh IDs from completed, visible run directories on disk."""

        disk_experiment_ids: set[str] = set()
        disk_portfolio_ids: set[str] = set()
        if self.runs_root.exists():
            for run_dir in sorted(self.runs_root.iterdir()):
                if run_dir.name.startswith(".") or not run_dir.is_dir():
                    continue
                config_path = run_dir / "config.json"
                if not config_path.is_file():
                    continue
                try:
                    with config_path.open(encoding="utf-8") as config_file:
                        raw_config = json.load(config_file)
                    if not isinstance(raw_config, dict):
                        raise ValueError("configuration root must be a JSON object")
                    config = ExperimentConfig.from_mapping(raw_config)
                except Exception as exc:
                    raise ValueError(f"malformed experiment config: {config_path}") from exc

                if config.experiment_id in disk_experiment_ids:
                    raise ValueError(f"duplicate experiment_id in completed runs: {config.experiment_id}")
                disk_experiment_ids.add(config.experiment_id)
                config_portfolio_ids = {
                    config.portfolio_id,
                    config.benchmark_portfolio_ids.cash_only,
                    config.benchmark_portfolio_ids.buy_and_hold,
                }
                duplicate_ids = config_portfolio_ids & disk_portfolio_ids
                if duplicate_ids:
                    duplicate = sorted(duplicate_ids)[0]
                    raise ValueError(f"duplicate portfolio_id in completed runs: {duplicate}")
                disk_portfolio_ids.update(config_portfolio_ids)

        reservation_dir = self.runs_root / ".paisa-registry"
        if reservation_dir.exists():
            for reservation_path in sorted(reservation_dir.glob("*.json")):
                try:
                    with reservation_path.open(encoding="utf-8") as reservation_file:
                        reservation = json.load(reservation_file)
                    kind = reservation["kind"]
                    identifier = str(reservation["identifier"])
                    if kind == "experiment":
                        disk_experiment_ids.add(identifier)
                    elif kind == "portfolio":
                        disk_portfolio_ids.add(identifier)
                    else:
                        raise ValueError("unknown reservation kind")
                    if not reservation.get("experiment_id") or not reservation.get("token"):
                        raise ValueError("incomplete reservation")
                except Exception as exc:
                    raise ValueError(
                        f"malformed experiment reservation: {reservation_path}"
                    ) from exc

        self._experiment_ids.update(disk_experiment_ids)
        self._portfolio_ids.update(disk_portfolio_ids)

    @staticmethod
    def _config_portfolio_ids(config: ExperimentConfig) -> tuple[str, str, str]:
        return (
            config.portfolio_id,
            config.benchmark_portfolio_ids.cash_only,
            config.benchmark_portfolio_ids.buy_and_hold,
        )

    def _reservation_specs(self, config: ExperimentConfig) -> list[tuple[str, str, Path]]:
        reservation_dir = self.runs_root / ".paisa-registry"
        specs = [
            (
                "experiment",
                config.experiment_id,
                reservation_dir / f"experiment--{config.experiment_id}.json",
            )
        ]
        specs.extend(
            (
                "portfolio",
                portfolio_id,
                reservation_dir / f"portfolio--{portfolio_id}.json",
            )
            for portfolio_id in sorted(self._config_portfolio_ids(config))
        )
        return specs

    def _create_disk_reservations(self, config: ExperimentConfig, token: str) -> None:
        reservation_dir = self.runs_root / ".paisa-registry"
        reservation_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []
        try:
            for kind, identifier, path in self._reservation_specs(config):
                payload = {
                    "kind": kind,
                    "identifier": identifier,
                    "experiment_id": config.experiment_id,
                    "token": token,
                }
                with path.open("x", encoding="utf-8") as reservation_file:
                    json.dump(payload, reservation_file, sort_keys=True)
                    reservation_file.write("\n")
                created.append(path)
        except FileExistsError as exc:
            for path in created:
                path.unlink(missing_ok=True)
            raise ValueError("experiment or portfolio ID is concurrently reserved") from exc
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise

    def _remove_disk_reservations(self, config: ExperimentConfig, token: str) -> None:
        for _, _, path in self._reservation_specs(config):
            if not path.exists():
                continue
            try:
                reservation = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if reservation.get("token") == token:
                path.unlink(missing_ok=True)
        reservation_dir = self.runs_root / ".paisa-registry"
        try:
            reservation_dir.rmdir()
        except OSError:
            # Another in-flight experiment may still own reservations.
            pass
