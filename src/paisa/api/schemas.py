"""Explicit response schemas for the read-only PAISA v1 results API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DateRange(BaseModel):
    start: str
    end: str


class EvaluationDateRange(BaseModel):
    feature_start: str
    feature_end: str
    target_start: str
    target_end: str


class RunContextResponse(BaseModel):
    schema_version: str
    run_id: str
    source_mode: str
    data_date_range: DateRange
    warnings: list[str]
    limitations: list[str]
    disclaimer: str


class VerticalSliceListItem(BaseModel):
    schema_version: str
    run_id: str
    status: str
    symbol: str
    completed_at: str
    source_mode: str
    data_date_range: DateRange
    model_training_run_id: str
    model_count: int
    experiment_count: int


class VerticalSliceListResponse(BaseModel):
    schema_version: str
    total: int
    limit: int
    offset: int
    rows: list[VerticalSliceListItem]
    disclaimer: str


class ExperimentAssumptions(BaseModel):
    initial_cash_pkr: str
    buy_probability_up_gte: float
    sell_probability_up_lte: float
    maximum_allocation: float
    long_only: bool
    whole_shares: bool
    transaction_cost_bps: float
    slippage_bps: float
    signal_timing: str
    execution: str
    valuation: str
    ending_position: str


class VerticalSliceDetailResponse(RunContextResponse):
    status: str
    symbol: str
    created_at: str
    evaluation_dates: EvaluationDateRange
    model_training_run_id: str
    model_variants: list[str]
    registered_model_ids: dict[str, str]
    experiment_ids: list[str]
    shared_experiment_assumptions: ExperimentAssumptions


class BestValidationModel(BaseModel):
    model_id: str
    registered_model_id: str
    algorithm: str
    validation_balanced_accuracy: float | None
    validation_f1: float
    selection_basis: str


class BestPortfolio(BaseModel):
    model_id: str
    portfolio_id: str
    ending_portfolio_value: str
    total_return: float
    sharpe_ratio: float | None
    selection_basis: str


class LowestDrawdownPortfolio(BaseModel):
    model_id: str
    portfolio_id: str
    maximum_drawdown: float
    total_return: float


class BuyAndHoldSummary(BaseModel):
    portfolio_id: str
    ending_portfolio_value: str
    total_return: float
    maximum_drawdown: float
    fees_paid: str


class TrainingDataSummary(BaseModel):
    provider: str
    actual_feature_start: str
    actual_feature_end: str
    dataset_sha256: str


class VerticalSliceSummaryResponse(RunContextResponse):
    status: str
    symbol: str
    completed_at: str
    model_training_run_id: str
    evaluation_dates: EvaluationDateRange
    best_validation_prediction_model: BestValidationModel
    best_test_period_simulated_portfolio: BestPortfolio
    lowest_drawdown_portfolio: LowestDrawdownPortfolio
    buy_and_hold_result: BuyAndHoldSummary
    any_model_portfolio_beat_buy_and_hold: bool
    selection_policy: str
    target_interpretation: str
    portfolio_return_interpretation: str
    training_data: TrainingDataSummary


class ModelMetricRow(BaseModel):
    model_id: str
    registered_model_id: str
    algorithm: str
    validation_accuracy: float
    validation_balanced_accuracy: float | None
    validation_f1: float
    test_accuracy: float
    test_balanced_accuracy: float | None
    test_f1: float
    train_count: int
    validation_count: int
    test_count: int
    train_class_0: int
    train_class_1: int
    validation_class_0: int
    validation_class_1: int
    test_class_0: int
    test_class_1: int


class PortfolioMetricRow(BaseModel):
    portfolio_type: str
    model_id: str
    portfolio_id: str
    ending_portfolio_value: str
    total_return: float
    maximum_drawdown: float
    volatility: float | None
    sharpe_ratio: float | None
    number_of_trades: int
    fees_paid: str
    result_versus_buy_and_hold: float


class LeaderboardRow(BaseModel):
    model_id: str
    registered_model_id: str
    algorithm: str
    validation_balanced_accuracy: float | None
    validation_f1: float
    test_accuracy: float
    test_balanced_accuracy: float | None
    test_f1: float
    portfolio_id: str
    ending_portfolio_value: str
    total_return: float
    maximum_drawdown: float
    volatility: float | None
    sharpe_ratio: float | None
    number_of_trades: int
    fees_paid: str
    result_versus_buy_and_hold: float


class ModelsResponse(RunContextResponse):
    total: int
    rows: list[ModelMetricRow]


class PortfoliosResponse(RunContextResponse):
    total: int
    rows: list[PortfolioMetricRow]


class LeaderboardResponse(RunContextResponse):
    total: int
    rows: list[LeaderboardRow]


class EquityCurveRow(BaseModel):
    date: str
    series_id: str
    portfolio_type: str
    model_id: str | None
    portfolio_id: str
    cash: str
    shares: int
    market_price: str
    total_equity: str
    drawdown: float


class DrawdownRow(BaseModel):
    date: str
    series_id: str
    portfolio_type: str
    model_id: str | None
    portfolio_id: str
    drawdown: float


class EquityCurvesResponse(RunContextResponse):
    total: int
    limit: int
    offset: int
    rows: list[EquityCurveRow]


class DrawdownsResponse(RunContextResponse):
    total: int
    limit: int
    offset: int
    rows: list[DrawdownRow]


class Thresholds(BaseModel):
    buy_probability_up_gte: float
    sell_probability_up_lte: float


class BenchmarkPortfolioIds(BaseModel):
    cash_only: str
    buy_and_hold: str


class TransactionCost(BaseModel):
    type: str
    value: float


class ExecutionPolicy(BaseModel):
    signal_timing: str
    execution_session: str
    reference_price_field: str
    slippage_bps: float
    missing_execution_price: str
    allocation_mode: str
    whole_share_rounding: str
    valuation_price_field: str
    missing_valuation_price: str
    ending_position: str


class MetricAssumptions(BaseModel):
    annualization_factor: int
    risk_free_rate_annual: float


class ExperimentConfiguration(BaseModel):
    schema_version: str
    experiment_id: str
    model_id: str
    portfolio_id: str
    benchmark_portfolio_ids: BenchmarkPortfolioIds
    symbol: str
    initial_cash: str
    thresholds: Thresholds
    maximum_allocation: float
    transaction_cost: TransactionCost
    evaluation_dates: EvaluationDateRange
    execution_policy: ExecutionPolicy
    metric_assumptions: MetricAssumptions
    created_at: str


class PredictionMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int
    positive_label: int
    average: str
    zero_division: int


class BenchmarkResult(BaseModel):
    portfolio_id: str
    ending_equity: str
    total_return: float


class BenchmarkComparison(BaseModel):
    cash_only: BenchmarkResult
    buy_and_hold: BenchmarkResult
    excess_return_vs_cash_only: float
    excess_return_vs_buy_and_hold: float


class ExperimentResult(BaseModel):
    schema_version: str
    experiment_id: str
    model_id: str
    portfolio_id: str
    prediction_metrics: PredictionMetrics
    initial_equity: str
    ending_equity: str
    total_return: float
    maximum_drawdown: float
    volatility: float | None
    sharpe_ratio: float | None
    number_of_trades: int
    win_rate: float | None
    average_gain: str | None
    average_loss: str | None
    turnover: float
    fees_paid: str
    time_in_market: float
    comparison_with_buy_and_hold: float
    benchmark_comparison: BenchmarkComparison


class ExperimentListRow(BaseModel):
    experiment_id: str
    variant_id: str
    registered_model_id: str
    algorithm: str
    portfolio_id: str
    symbol: str
    evaluation_dates: EvaluationDateRange
    ending_equity: str
    total_return: float
    maximum_drawdown: float
    sharpe_ratio: float | None
    number_of_trades: int


class ExperimentsResponse(RunContextResponse):
    total: int
    rows: list[ExperimentListRow]


class ExperimentDetailResponse(RunContextResponse):
    variant_id: str
    algorithm: str
    configuration: ExperimentConfiguration
    result: ExperimentResult
    actual_class_usage: str


class TradeRow(BaseModel):
    schema_version: str
    trade_id: str
    portfolio_id: str
    signal_date: str
    execution_date: str
    action: str
    quantity: int
    reference_price: str
    execution_price: str
    fee: str
    cash_after_trade: str
    shares_after_trade: int


class SnapshotRow(BaseModel):
    schema_version: str
    portfolio_id: str
    date: str
    cash: str
    shares: int
    market_price: str
    position_value: str
    total_equity: str
    realized_profit_loss: str
    unrealized_profit_loss: str


class TradesResponse(RunContextResponse):
    experiment_id: str
    actual_class_usage: str
    total: int
    limit: int
    offset: int
    rows: list[TradeRow]


class SnapshotsResponse(RunContextResponse):
    experiment_id: str
    total: int
    limit: int
    offset: int
    rows: list[SnapshotRow]


class BenchmarkSnapshotSeries(BaseModel):
    portfolio_id: str
    ending_equity: str
    total_return: float
    total: int
    limit: int
    offset: int
    rows: list[SnapshotRow]


class BenchmarkTradeSeries(BaseModel):
    portfolio_id: str
    total: int
    limit: int
    offset: int
    rows: list[TradeRow]


class BenchmarksResponse(RunContextResponse):
    experiment_id: str
    cash_only: BenchmarkSnapshotSeries
    buy_and_hold: BenchmarkSnapshotSeries
    buy_and_hold_trades: BenchmarkTradeSeries


class SourceDetails(BaseModel):
    source: str
    acquisition_mode: str
    is_real_data: bool
    live_fetch_succeeded: bool
    live_acquisition_error_present: bool
    original_provider: str
    legacy_metadata_preserved: bool
    producer_component: str | None
    row_count: int
    input_prices_sha256: str
    frozen_prices_sha256: str
    provenance_sha256: str | None
    migration_strategy: str | None
    provenance_strength: str | None


class PipelineEvidence(BaseModel):
    schema_version: str | None
    producer: dict[str, Any] | str | None
    generated_at: str | None
    provider: str | None
    symbols_requested: list[str]
    prices_artifact: dict[str, Any] | None


class MigrationEvidence(BaseModel):
    schema_version: str | None
    status: str | None
    producer: dict[str, Any] | str | None
    migration_strategy: str | None
    provenance_strength: str | None
    limitations: list[str]


class SourceResponse(RunContextResponse):
    source: SourceDetails
    pipeline_evidence: PipelineEvidence | None
    migration_evidence: MigrationEvidence | None
