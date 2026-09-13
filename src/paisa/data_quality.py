from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "source"]
REQUIRED_DATASET_COLUMNS = ["date", "ticker", "target_next_day_up"]


def _safe_date(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(pd.to_datetime(value).date())


def _largest_calendar_gap_days(dates: pd.Series) -> int:
    cleaned = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(cleaned) < 2:
        return 0
    gaps = cleaned.diff().dropna().dt.days
    return int(gaps.max()) if not gaps.empty else 0


def _missing_count(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    return int(frame[column].isna().sum())


def _required_missing(frame: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [column for column in required_columns if column not in frame.columns]


def summarize_data_quality(
    prices: pd.DataFrame,
    features: pd.DataFrame | None = None,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a per-ticker quality report for PAISA stock, feature, and label data.

    The report intentionally stays simple and explainable because its job is to
    prove that the data foundation is usable before adding sentiment/NLP layers.
    """
    if prices.empty:
        raise ValueError("prices is empty; run the pipeline before generating a quality report.")

    missing_price_columns = _required_missing(prices, REQUIRED_PRICE_COLUMNS)
    if missing_price_columns:
        raise ValueError(f"prices missing required columns: {missing_price_columns}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()

    features_copy: pd.DataFrame | None = None
    if features is not None and not features.empty:
        features_copy = features.copy()
        features_copy["date"] = pd.to_datetime(features_copy["date"], errors="coerce")
        features_copy["ticker"] = features_copy["ticker"].astype(str).str.upper().str.strip()

    dataset_copy: pd.DataFrame | None = None
    if dataset is not None and not dataset.empty:
        missing_dataset_columns = _required_missing(dataset, REQUIRED_DATASET_COLUMNS)
        if missing_dataset_columns:
            raise ValueError(f"dataset missing required columns: {missing_dataset_columns}")
        dataset_copy = dataset.copy()
        dataset_copy["date"] = pd.to_datetime(dataset_copy["date"], errors="coerce")
        dataset_copy["ticker"] = dataset_copy["ticker"].astype(str).str.upper().str.strip()

    ticker_reports: list[dict[str, Any]] = []
    for ticker, group in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        duplicate_dates = int(group.duplicated(subset=["ticker", "date"]).sum())
        dataset_rows = 0
        up_days = 0
        down_days = 0
        up_ratio: float | None = None

        if dataset_copy is not None:
            ticker_dataset = dataset_copy[dataset_copy["ticker"] == ticker]
            dataset_rows = int(len(ticker_dataset))
            if "target_next_day_up" in ticker_dataset.columns and not ticker_dataset.empty:
                up_days = int((ticker_dataset["target_next_day_up"] == 1).sum())
                down_days = int((ticker_dataset["target_next_day_up"] == 0).sum())
                total_labelled = up_days + down_days
                up_ratio = round(up_days / total_labelled, 4) if total_labelled else None

        complete_feature_rows = None
        if features_copy is not None:
            ticker_features = features_copy[features_copy["ticker"] == ticker]
            if ticker_features.empty:
                complete_feature_rows = 0
            else:
                feature_check_columns = [
                    col
                    for col in ticker_features.columns
                    if col not in {"date", "ticker", "source", "high", "low"}
                ]
                complete_feature_rows = int(ticker_features.dropna(subset=feature_check_columns).shape[0])

        ticker_reports.append(
            {
                "ticker": ticker,
                "source": ",".join(sorted(group["source"].dropna().astype(str).unique().tolist())),
                "price_rows": int(len(group)),
                "dataset_rows": dataset_rows,
                "complete_feature_rows": complete_feature_rows,
                "start_date": _safe_date(group["date"].min()),
                "end_date": _safe_date(group["date"].max()),
                "missing_open": _missing_count(group, "open"),
                "missing_high": _missing_count(group, "high"),
                "missing_low": _missing_count(group, "low"),
                "missing_close": _missing_count(group, "close"),
                "missing_volume": _missing_count(group, "volume"),
                "duplicate_dates": duplicate_dates,
                "largest_calendar_gap_days": _largest_calendar_gap_days(group["date"]),
                "label_up_days": up_days,
                "label_down_days": down_days,
                "label_up_ratio": up_ratio,
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tickers_count": len(ticker_reports),
        "price_rows": int(len(prices)),
        "feature_rows": int(len(features)) if features is not None else None,
        "dataset_rows": int(len(dataset)) if dataset is not None else None,
        "date_range": {
            "start": _safe_date(prices["date"].min()),
            "end": _safe_date(prices["date"].max()),
        },
        "issues": _collect_issues(ticker_reports),
        "tickers": ticker_reports,
    }
    return summary


def _collect_issues(ticker_reports: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for item in ticker_reports:
        ticker = item["ticker"]
        if item["price_rows"] < 200:
            issues.append(f"{ticker}: fewer than 200 price rows")
        if item["dataset_rows"] == 0:
            issues.append(f"{ticker}: no ML-ready dataset rows")
        if item["missing_close"]:
            issues.append(f"{ticker}: missing close values")
        if item["missing_volume"]:
            issues.append(f"{ticker}: missing volume values")
        if item["duplicate_dates"]:
            issues.append(f"{ticker}: duplicate ticker/date rows")
        if item["missing_high"] and item["missing_high"] == item["price_rows"]:
            issues.append(f"{ticker}: high column unavailable from this source")
        if item["missing_low"] and item["missing_low"] == item["price_rows"]:
            issues.append(f"{ticker}: low column unavailable from this source")
    return issues


def ticker_report_frame(report: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(report.get("tickers", []))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append("" if pd.isna(value) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def render_markdown_report(report: dict[str, Any]) -> str:
    ticker_frame = ticker_report_frame(report)
    issue_lines = "\n".join(f"- {issue}" for issue in report.get("issues", [])) or "- No blocking data-quality issues detected."

    if ticker_frame.empty:
        table = "No ticker rows available."
    else:
        display_columns = [
            "ticker",
            "price_rows",
            "dataset_rows",
            "start_date",
            "end_date",
            "missing_close",
            "missing_volume",
            "duplicate_dates",
            "label_up_ratio",
        ]
        table = _markdown_table(ticker_frame[display_columns])

    return f"""# PAISA v0 Data Quality Report

Generated at: `{report.get('generated_at')}`

## Summary

- Tickers checked: `{report.get('tickers_count')}`
- Price rows: `{report.get('price_rows')}`
- Feature rows: `{report.get('feature_rows')}`
- ML-ready dataset rows: `{report.get('dataset_rows')}`
- Date range: `{report.get('date_range', {}).get('start')}` to `{report.get('date_range', {}).get('end')}`

## Issues / limitations

{issue_lines}

## Per-ticker checks

{table}

## How to interpret this report

- `price_rows` proves that raw/cleaned PSX data exists for the ticker.
- `dataset_rows` proves that feature engineering and label generation produced ML-ready rows.
- `missing_close`, `missing_volume`, and `duplicate_dates` should normally be `0`.
- `missing_high` and `missing_low` may be high for the current `psx-dps` provider because the source currently provides open, close, and volume but may not provide high/low.
- `label_up_ratio` should not be extremely close to `0` or `1`; otherwise the target is imbalanced.
"""


def write_data_quality_report(
    prices_path: str | Path,
    features_path: str | Path | None,
    dataset_path: str | Path | None,
    output_dir: str | Path,
    docs_dir: str | Path | None = None,
) -> dict[str, Any]:
    prices = pd.read_csv(prices_path)
    features = pd.read_csv(features_path) if features_path and Path(features_path).exists() else None
    dataset = pd.read_csv(dataset_path) if dataset_path and Path(dataset_path).exists() else None

    report = summarize_data_quality(prices=prices, features=features, dataset=dataset)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ticker_report_frame(report).to_csv(out / "data_quality_report.csv", index=False)

    if docs_dir is not None:
        docs = Path(docs_dir)
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "data_quality_report_v0.md").write_text(render_markdown_report(report), encoding="utf-8")

    return report
