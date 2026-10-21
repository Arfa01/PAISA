"""Versioned, read-only routes for immutable PAISA experiment results."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query

from paisa.api.artifact_repository import (
    ACADEMIC_DISCLAIMER,
    ACTUAL_CLASS_LIMITATION,
    ArtifactIntegrityError,
    ArtifactRepository,
    DefaultRunNotConfiguredError,
    ExperimentNotFoundError,
    InvalidDateRangeError,
    InvalidIdentifierError,
    RunNotCompletedError,
    RunNotFoundError,
    StockNotFoundError,
    get_artifact_repository,
)
from paisa.api.schemas import (
    BenchmarksResponse,
    DrawdownsResponse,
    ErrorResponse,
    EquityCurvesResponse,
    ExperimentDetailResponse,
    ExperimentsResponse,
    LeaderboardResponse,
    ModelsResponse,
    PortfoliosResponse,
    SnapshotsResponse,
    SourceResponse,
    StockPricesResponse,
    TradesResponse,
    VerticalSliceDetailResponse,
    VerticalSliceListResponse,
    VerticalSliceSummaryResponse,
)


router = APIRouter(prefix="/api/v1", tags=["experiment-results-v1"])
MAX_TABLE_LIMIT = 1000
MAX_PRICE_LIMIT = 5000
T = TypeVar("T")
_WEB_URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_FILE_URI = re.compile(r"\bfile:(?://)?[^\s\"'<>]*", re.IGNORECASE)
_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:[^\s\"'<>]*)")
_UNC_PATH = re.compile(
    r"(?<![A-Za-z0-9\\])(?:\\\\|//)[^\\/\s\"'<>]+[\\/][^\s\"'<>]*"
)
_ROOTED_WINDOWS_PATH = re.compile(
    r"(?<![A-Za-z0-9\\])\\(?!\\)[^\\/\s\"'<>]+\\[^\s\"'<>]*"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")


def _assert_no_absolute_local_paths(value: Any) -> None:
    if isinstance(value, str):
        without_urls = _WEB_URL.sub("", value)
        if (
            _FILE_URI.search(without_urls)
            or _POSIX_ABSOLUTE_PATH.search(without_urls)
            or _UNC_PATH.search(without_urls)
            or _ROOTED_WINDOWS_PATH.search(without_urls)
            or _WINDOWS_ABSOLUTE_PATH.search(without_urls)
        ):
            raise ArtifactIntegrityError(
                "Artifact response contains a prohibited absolute local path"
            )
    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_no_absolute_local_paths(key)
            _assert_no_absolute_local_paths(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_absolute_local_paths(nested)


def _repository_call(operation: Callable[[], T]) -> T:
    try:
        result = operation()
        _assert_no_absolute_local_paths(result)
        return result
    except (InvalidIdentifierError, InvalidDateRangeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RunNotFoundError, ExperimentNotFoundError, StockNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DefaultRunNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (RunNotCompletedError, ArtifactIntegrityError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail="Artifact contents do not satisfy the PAISA result schema",
        ) from exc


def _table_response(
    repository: ArtifactRepository,
    run_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **repository.get_context(run_id),
        "total": len(rows),
        "rows": rows,
    }


@router.get("/vertical-slices", response_model=VerticalSliceListResponse)
def list_vertical_slices(
    limit: int = Query(default=200, ge=1, le=MAX_TABLE_LIMIT),
    offset: int = Query(default=0, ge=0),
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    """List only checksum-valid, transitively complete PASS runs."""

    rows = _repository_call(repository.list_completed_runs)
    return {
        "schema_version": "1.0",
        "total": len(rows),
        "limit": limit,
        "offset": offset,
        "rows": rows[offset : offset + limit],
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


@router.get(
    "/vertical-slices/{run_id}", response_model=VerticalSliceDetailResponse
)
def get_vertical_slice(
    run_id: str,
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(lambda: repository.load_run_detail(run_id))


@router.get(
    "/vertical-slices/{run_id}/summary",
    response_model=VerticalSliceSummaryResponse,
)
def get_vertical_slice_summary(
    run_id: str,
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(lambda: repository.load_summary(run_id))


@router.get(
    "/vertical-slices/{run_id}/stocks/{ticker}/prices",
    response_model=StockPricesResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Malformed run/ticker identifier or an inverted date range.",
        },
        404: {
            "model": ErrorResponse,
            "description": "The run does not exist or the ticker is absent from its snapshot.",
        },
        409: {
            "model": ErrorResponse,
            "description": "The run is incomplete, failed, or violates its integrity contract.",
        },
        422: {
            "description": "A query parameter is not a valid ISO date or is out of bounds."
        },
    },
)
def get_stock_prices(
    run_id: str,
    ticker: str,
    limit: int = Query(default=500, ge=1, le=MAX_PRICE_LIMIT),
    offset: int = Query(default=0, ge=0),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    """Read historical prices from the completed run's immutable input snapshot."""

    return _repository_call(
        lambda: repository.load_stock_prices(
            run_id,
            ticker,
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
        )
    )


@router.get(
    "/vertical-slices/{run_id}/leaderboard",
    response_model=LeaderboardResponse,
)
def get_vertical_slice_leaderboard(
    run_id: str,
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: _table_response(
            repository, run_id, repository.load_leaderboard(run_id)
        )
    )

@router.get("/leaderboard", response_model=LeaderboardResponse)
def get_default_leaderboard(
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    """Return the explicitly configured default run; never guess one."""

    return _repository_call(
        lambda: _table_response(
            repository,
            repository.default_run_id,
            repository.load_leaderboard(repository.default_run_id),
        )
    )


@router.get("/vertical-slices/{run_id}/models", response_model=ModelsResponse)
def get_models(
    run_id: str,
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: _table_response(
            repository, run_id, repository.load_model_metrics(run_id)
        )
    )


@router.get(
    "/vertical-slices/{run_id}/portfolios", response_model=PortfoliosResponse
)
def get_portfolios(
    run_id: str,
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: _table_response(
            repository, run_id, repository.load_portfolio_metrics(run_id)
        )
    )


@router.get(
    "/vertical-slices/{run_id}/equity-curves",
    response_model=EquityCurvesResponse,
)
def get_equity_curves(
    run_id: str,
    limit: int = Query(default=200, ge=1, le=MAX_TABLE_LIMIT),
    offset: int = Query(default=0, ge=0),
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: {
            **repository.get_context(run_id),
            **repository.load_equity_curves(run_id, limit=limit, offset=offset),
        }
    )

@router.get(
    "/vertical-slices/{run_id}/drawdowns", response_model=DrawdownsResponse
)
def get_drawdowns(
    run_id: str,
    limit: int = Query(default=200, ge=1, le=MAX_TABLE_LIMIT),
    offset: int = Query(default=0, ge=0),
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: {
            **repository.get_context(run_id),
            **repository.load_drawdowns(run_id, limit=limit, offset=offset),
        }
    )


@router.get("/vertical-slices/{run_id}/source", response_model=SourceResponse)
def get_source(
    run_id: str,
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(lambda: repository.load_source(run_id))


@router.get(
    "/vertical-slices/{run_id}/experiments", response_model=ExperimentsResponse
)
def get_experiments(
    run_id: str,
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: _table_response(
            repository, run_id, repository.list_experiments(run_id)
        )
    )


@router.get(
    "/vertical-slices/{run_id}/experiments/{experiment_id}",
    response_model=ExperimentDetailResponse,
)
def get_experiment(
    run_id: str,
    experiment_id: str,
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: repository.load_experiment(run_id, experiment_id)
    )


@router.get(
    "/vertical-slices/{run_id}/experiments/{experiment_id}/trades",
    response_model=TradesResponse,
)
def get_trades(
    run_id: str,
    experiment_id: str,
    limit: int = Query(default=200, ge=1, le=MAX_TABLE_LIMIT),
    offset: int = Query(default=0, ge=0),
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: {
            **repository.get_context(run_id),
            "experiment_id": experiment_id,
            "actual_class_usage": ACTUAL_CLASS_LIMITATION,
            **repository.load_trades(
                run_id, experiment_id, limit=limit, offset=offset
            ),
        }
    )


@router.get(
    "/vertical-slices/{run_id}/experiments/{experiment_id}/snapshots",
    response_model=SnapshotsResponse,
)
def get_snapshots(
    run_id: str,
    experiment_id: str,
    limit: int = Query(default=200, ge=1, le=MAX_TABLE_LIMIT),
    offset: int = Query(default=0, ge=0),
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: {
            **repository.get_context(run_id),
            "experiment_id": experiment_id,
            **repository.load_snapshots(
                run_id, experiment_id, limit=limit, offset=offset
            ),
        }
    )


@router.get(
    "/vertical-slices/{run_id}/experiments/{experiment_id}/benchmarks",
    response_model=BenchmarksResponse,
)
def get_benchmarks(
    run_id: str,
    experiment_id: str,
    limit: int = Query(default=200, ge=1, le=MAX_TABLE_LIMIT),
    offset: int = Query(default=0, ge=0),
    repository: ArtifactRepository = Depends(get_artifact_repository),
) -> dict[str, Any]:
    return _repository_call(
        lambda: {
            **repository.get_context(run_id),
            "experiment_id": experiment_id,
            **repository.load_benchmarks(
                run_id, experiment_id, limit=limit, offset=offset
            ),
        }
    )
