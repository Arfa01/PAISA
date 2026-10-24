"""Explicit response schemas for PAISA HTTP APIs."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

from paisa.api.classification_metrics import validate_confusion_derived_metrics


_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
_Rate = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
_NullableRate = _Rate | None
_Correlation = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
_NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0)]
_UnifiedClassificationCount = Annotated[int, Field(ge=0, le=10_000_000)]
_PositiveUnifiedClassificationCount = Annotated[
    int, Field(gt=0, le=10_000_000)
]
_NonNegativeMoney = Annotated[
    str,
    Field(pattern=r"^(?:0|[1-9][0-9]*)\.[0-9]{2}$"),
]
_ConfusionRow = Annotated[
    list[_UnifiedClassificationCount], Field(min_length=2, max_length=2)
]
_ConfusionMatrix = Annotated[list[_ConfusionRow], Field(min_length=2, max_length=2)]


class DateRange(BaseModel):
    start: str
    end: str


class HealthResponse(BaseModel):
    status: str
    data_dir: str
    model_dir: str
    prices_ready: bool
    dataset_ready: bool
    model_ready: bool
    price_rows: int | None = None
    stocks: list[str] | None = None


class ErrorResponse(BaseModel):
    detail: str


class StockPriceRow(BaseModel):
    date: str
    open: str
    high: str | None
    low: str | None
    close: str
    volume: int
    source: str


class StockPricesResponse(BaseModel):
    schema_version: str
    run_id: str
    ticker: str
    source_mode: str
    data_date_range: DateRange
    total: int
    limit: int
    offset: int
    rows: list[StockPriceRow]
    warnings: list[str]
    disclaimer: str


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


class UnifiedComparisonContext(BaseModel):
    schema_version: str
    comparison_id: str
    status: str
    symbol: str
    holdout_term: str
    outcomes_are_not_combined: bool
    limitations: list[str]
    disclaimer: str


class UnifiedComparisonListItem(BaseModel):
    schema_version: str
    comparison_id: str
    status: str
    symbol: str
    model_count: int
    validation_count: int
    shared_holdout_count: int
    shared_holdout_target_start: str
    shared_holdout_target_end: str
    holdout_term: str
    is_default: bool


class UnifiedComparisonListResponse(BaseModel):
    schema_version: str
    total: int
    limit: int
    offset: int
    default_comparison_id: str | None
    rows: list[UnifiedComparisonListItem]
    disclaimer: str


class UnifiedModelIdentity(BaseModel):
    variant_id: str
    registered_model_id: str
    algorithm: str
    configuration_family: str


class UnifiedSelectionPolicy(BaseModel):
    primary: str
    tie_breakers: list[str]
    test_metrics_used_for_selection: bool


class UnifiedPresentation(BaseModel):
    format: str
    plots_included: bool
    data_sources: list[str]


class UnifiedComparisonDetailResponse(UnifiedComparisonContext):
    models: list[UnifiedModelIdentity]
    source_frequency: str
    validation_count: int
    shared_holdout_count: int
    selection_policy: UnifiedSelectionPolicy
    presentation: UnifiedPresentation


class UnifiedSourceReference(BaseModel):
    role: str
    artifact_type: str
    run_id: str
    integrity_root_sha256: str
    payload_file_count: int
    integrity_root_derivation: str


class UnifiedSourceRelationships(BaseModel):
    vertical_slice_uses_training_run: bool
    lstm_uses_training_run: bool
    lstm_uses_vertical_slice: bool


class UnifiedSourcesResponse(UnifiedComparisonContext):
    sources: list[UnifiedSourceReference]
    relationships: UnifiedSourceRelationships


class UnifiedHoldoutContext(BaseModel):
    term: str
    common_count: int
    target_start: str
    target_end: str
    previously_exposed_through: list[str]


class UnifiedPredictionOutcome(BaseModel):
    variant_id: str
    registered_model_id: str
    criterion: str
    scope: str
    accuracy: _Rate
    balanced_accuracy: _NullableRate
    f1: _Rate


class UnifiedPortfolioOutcome(BaseModel):
    variant_id: str
    registered_model_id: str
    portfolio_id: str
    criterion: str
    ending_equity: str
    total_return: _FiniteFloat
    maximum_drawdown: _Rate
    fees_paid: _NonNegativeMoney


class UnifiedBenchmarkOutcome(BaseModel):
    initial_equity: str
    ending_equity: str
    total_return: _FiniteFloat
    maximum_drawdown: _Rate
    volatility: _NonNegativeFloat | None
    sharpe_ratio: _FiniteFloat | None
    number_of_trades: _NonNegativeInt
    fees_paid: _NonNegativeMoney


class UnifiedBeatBuyAndHoldOutcome(BaseModel):
    value: bool
    best_model_variant_id: str
    best_model_minus_buy_and_hold: _FiniteFloat
    comparison_metric: str


class UnifiedReportedOutcomes(BaseModel):
    validation_selected_model: UnifiedPredictionOutcome
    lstm_validation_preferred_model: UnifiedPredictionOutcome
    best_shared_holdout_balanced_accuracy: UnifiedPredictionOutcome
    best_shared_holdout_f1: UnifiedPredictionOutcome
    best_simulated_model_portfolio: UnifiedPortfolioOutcome
    lowest_drawdown_model_portfolio: UnifiedPortfolioOutcome
    lowest_fee_model_portfolio: UnifiedPortfolioOutcome
    cash_only_benchmark: UnifiedBenchmarkOutcome
    buy_and_hold_result: UnifiedBenchmarkOutcome
    any_model_beat_buy_and_hold: UnifiedBeatBuyAndHoldOutcome


class UnifiedSummaryResponse(UnifiedComparisonContext):
    holdout_context: UnifiedHoldoutContext
    reported_outcomes: UnifiedReportedOutcomes


class UnifiedPredictionMetricRow(BaseModel):
    competitor_id: str
    competitor_type: str
    registered_model_id: str | None
    algorithm: str | None
    configuration_family: str | None
    evaluation_scope: str
    row_count: _PositiveUnifiedClassificationCount
    support: _PositiveUnifiedClassificationCount
    accuracy: _Rate
    balanced_accuracy: _NullableRate
    precision: _Rate
    recall: _Rate
    f1: _Rate
    matthews_corrcoef: _Correlation | None
    roc_auc: _NullableRate
    brier_score: _Rate
    log_loss: _NonNegativeFloat | None
    expected_calibration_error: _Rate
    selection_eligible: bool
    metric_warnings: list[str]
    class_counts: dict[str, _UnifiedClassificationCount]
    predicted_class_counts: dict[str, _UnifiedClassificationCount]
    confusion_matrix: _ConfusionMatrix

    @model_validator(mode="after")
    def reconcile_confusion_counts(self) -> "UnifiedPredictionMetricRow":
        if set(self.class_counts) != {"0", "1"} or set(
            self.predicted_class_counts
        ) != {"0", "1"}:
            raise ValueError("class count maps must contain exactly classes 0 and 1")
        if self.row_count != self.support:
            raise ValueError("row_count and support must agree")
        if sum(self.class_counts.values()) != self.support:
            raise ValueError("class_counts must reconcile with support")
        if sum(self.predicted_class_counts.values()) != self.support:
            raise ValueError("predicted_class_counts must reconcile with support")
        if sum(map(sum, self.confusion_matrix)) != self.support:
            raise ValueError("confusion_matrix must reconcile with support")
        if [sum(row) for row in self.confusion_matrix] != [
            self.class_counts["0"],
            self.class_counts["1"],
        ]:
            raise ValueError("confusion_matrix rows must reconcile with class_counts")
        if [
            self.confusion_matrix[0][column] + self.confusion_matrix[1][column]
            for column in range(2)
        ] != [
            self.predicted_class_counts["0"],
            self.predicted_class_counts["1"],
        ]:
            raise ValueError(
                "confusion_matrix columns must reconcile with predicted_class_counts"
            )
        validate_confusion_derived_metrics(
            {
                "accuracy": self.accuracy,
                "balanced_accuracy": self.balanced_accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "matthews_corrcoef": self.matthews_corrcoef,
            },
            self.confusion_matrix,
            field="prediction metrics",
        )
        complete_class_coverage = all(
            self.class_counts[label] > 0 for label in ("0", "1")
        )
        if self.selection_eligible is not complete_class_coverage:
            raise ValueError(
                "selection_eligible must equal complete actual-class coverage"
            )
        if ("single_class_actuals" in self.metric_warnings) != (
            not complete_class_coverage
        ):
            raise ValueError(
                "single_class_actuals warning must match actual-class coverage"
            )
        if complete_class_coverage:
            if self.roc_auc is None or self.log_loss is None:
                raise ValueError(
                    "roc_auc and log_loss must be defined with both actual classes"
                )
        elif self.roc_auc is not None or self.log_loss is not None:
            raise ValueError(
                "roc_auc and log_loss must be null for single-class actuals"
            )
        return self


class UnifiedPredictionMetricsResponse(UnifiedComparisonContext):
    models: list[UnifiedPredictionMetricRow]
    reference_baselines: list[UnifiedPredictionMetricRow]


class UnifiedValidationMetricsResponse(UnifiedPredictionMetricsResponse):
    evaluation_scope: str


class UnifiedModelPortfolioMetric(BaseModel):
    variant_id: str
    registered_model_id: str
    experiment_id: str
    portfolio_id: str
    initial_equity: str
    ending_equity: str
    total_return: _FiniteFloat
    maximum_drawdown: _Rate
    volatility: _NonNegativeFloat | None
    sharpe_ratio: _FiniteFloat | None
    number_of_trades: _NonNegativeInt
    win_rate: _NullableRate
    average_gain: str | None
    average_loss: str | None
    turnover: _NonNegativeFloat
    fees_paid: _NonNegativeMoney
    time_in_market: _Rate
    excess_return_vs_cash_only: _FiniteFloat
    excess_return_vs_buy_and_hold: _FiniteFloat


class UnifiedBenchmarkPortfolioMetric(BaseModel):
    benchmark_id: str
    initial_equity: str
    ending_equity: str
    total_return: _FiniteFloat
    maximum_drawdown: _Rate
    volatility: _NonNegativeFloat | None
    sharpe_ratio: _FiniteFloat | None
    number_of_trades: _NonNegativeInt
    win_rate: _NullableRate
    average_gain: str | None
    average_loss: str | None
    turnover: _NonNegativeFloat
    fees_paid: _NonNegativeMoney
    time_in_market: _Rate
    ledger_count: _PositiveInt


class UnifiedPortfolioMetricsResponse(UnifiedComparisonContext):
    models: list[UnifiedModelPortfolioMetric]
    benchmarks: list[UnifiedBenchmarkPortfolioMetric]


class UnifiedPredictionRankingRow(BaseModel):
    rank: _PositiveInt
    competitor_id: str
    registered_model_id: str
    accuracy: _Rate
    balanced_accuracy: _NullableRate
    f1: _Rate


class UnifiedPredictionRankingsResponse(UnifiedComparisonContext):
    validation_model_selection: list[UnifiedPredictionRankingRow]
    lstm_validation_preference: list[UnifiedPredictionRankingRow]
    shared_project_holdout_balanced_accuracy: list[UnifiedPredictionRankingRow]
    shared_project_holdout_f1: list[UnifiedPredictionRankingRow]


class UnifiedPortfolioRankingRow(BaseModel):
    rank: _PositiveInt
    competitor_id: str
    registered_model_id: str
    total_return: _FiniteFloat | None
    maximum_drawdown: _NullableRate
    sharpe_ratio: _FiniteFloat | None
    fees_paid: _NonNegativeMoney | None
    number_of_trades: _NonNegativeInt | None


class UnifiedPortfolioRankingsResponse(UnifiedComparisonContext):
    simulated_model_portfolio_return: list[UnifiedPortfolioRankingRow]
    simulated_model_portfolio_drawdown: list[UnifiedPortfolioRankingRow]
    simulated_model_portfolio_fees: list[UnifiedPortfolioRankingRow]


class UnifiedCommonDateKey(BaseModel):
    ticker: str
    feature_date: str
    target_date: str
    actual_class: int


class UnifiedDateScope(BaseModel):
    terminology: str
    common_count: int
    feature_start: str
    feature_end: str
    target_start: str
    target_end: str
    original_counts: dict[str, int]
    common_counts: dict[str, int]
    dropped_target_dates: dict[str, list[str]]
    common_keys: list[UnifiedCommonDateKey]


class UnifiedHoldoutDateScope(UnifiedDateScope):
    previously_exposed_through: list[str]


class UnifiedCommonDateResponse(UnifiedComparisonContext):
    alignment_key: list[str]
    validation: UnifiedDateScope
    shared_project_holdout: UnifiedHoldoutDateScope


class UnifiedConfusionMatrixRow(BaseModel):
    competitor_id: str
    competitor_type: str
    registered_model_id: str | None
    evaluation_scope: str
    class_labels: Annotated[list[int], Field(min_length=2, max_length=2)]
    confusion_matrix: _ConfusionMatrix
    support: _PositiveUnifiedClassificationCount

    @model_validator(mode="after")
    def reconcile_confusion_support(self) -> "UnifiedConfusionMatrixRow":
        if self.class_labels != [0, 1]:
            raise ValueError("class_labels must equal [0, 1]")
        if sum(map(sum, self.confusion_matrix)) != self.support:
            raise ValueError("confusion_matrix must reconcile with support")
        return self


class UnifiedConfusionMatricesResponse(UnifiedComparisonContext):
    validation_models: list[UnifiedConfusionMatrixRow]
    validation_reference_baselines: list[UnifiedConfusionMatrixRow]
    shared_project_holdout_models: list[UnifiedConfusionMatrixRow]
    shared_project_holdout_reference_baselines: list[UnifiedConfusionMatrixRow]
