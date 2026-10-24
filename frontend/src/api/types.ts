export interface DateRange {
  start: string;
  end: string;
}

export interface EvaluationDateRange {
  feature_start: string;
  feature_end: string;
  target_start: string;
  target_end: string;
}

export interface RunContext {
  schema_version: string;
  run_id: string;
  source_mode: string;
  data_date_range: DateRange;
  warnings: string[];
  limitations: string[];
  disclaimer: string;
}

export interface VerticalSliceListItem {
  schema_version: string;
  run_id: string;
  status: string;
  symbol: string;
  completed_at: string;
  source_mode: string;
  data_date_range: DateRange;
  model_training_run_id: string;
  model_count: number;
  experiment_count: number;
}

export interface VerticalSliceListResponse {
  schema_version: string;
  total: number;
  limit: number;
  offset: number;
  rows: VerticalSliceListItem[];
  disclaimer: string;
}

export interface ExperimentAssumptions {
  initial_cash_pkr: string;
  buy_probability_up_gte: number;
  sell_probability_up_lte: number;
  maximum_allocation: number;
  long_only: boolean;
  whole_shares: boolean;
  transaction_cost_bps: number;
  slippage_bps: number;
  signal_timing: string;
  execution: string;
  valuation: string;
  ending_position: string;
}

export interface VerticalSliceDetailResponse extends RunContext {
  status: string;
  symbol: string;
  created_at: string;
  evaluation_dates: EvaluationDateRange;
  model_training_run_id: string;
  model_variants: string[];
  registered_model_ids: Record<string, string>;
  experiment_ids: string[];
  shared_experiment_assumptions: ExperimentAssumptions;
}

export interface BestValidationModel {
  model_id: string;
  registered_model_id: string;
  algorithm: string;
  validation_balanced_accuracy: number | null;
  validation_f1: number;
  selection_basis: string;
}

export interface BestPortfolio {
  model_id: string;
  portfolio_id: string;
  ending_portfolio_value: string;
  total_return: number;
  sharpe_ratio: number | null;
  selection_basis: string;
}

export interface LowestDrawdownPortfolio {
  model_id: string;
  portfolio_id: string;
  maximum_drawdown: number;
  total_return: number;
}

export interface BuyAndHoldSummary {
  portfolio_id: string;
  ending_portfolio_value: string;
  total_return: number;
  maximum_drawdown: number;
  fees_paid: string;
}

export interface VerticalSliceSummaryResponse extends RunContext {
  status: string;
  symbol: string;
  completed_at: string;
  model_training_run_id: string;
  evaluation_dates: EvaluationDateRange;
  best_validation_prediction_model: BestValidationModel;
  best_test_period_simulated_portfolio: BestPortfolio;
  lowest_drawdown_portfolio: LowestDrawdownPortfolio;
  buy_and_hold_result: BuyAndHoldSummary;
  any_model_portfolio_beat_buy_and_hold: boolean;
  selection_policy: string;
  target_interpretation: string;
  portfolio_return_interpretation: string;
  training_data: {
    provider: string;
    actual_feature_start: string;
    actual_feature_end: string;
    dataset_sha256: string;
  };
}

export interface ModelMetricRow {
  model_id: string;
  registered_model_id: string;
  algorithm: string;
  validation_accuracy: number;
  validation_balanced_accuracy: number | null;
  validation_f1: number;
  test_accuracy: number;
  test_balanced_accuracy: number | null;
  test_f1: number;
  train_count: number;
  validation_count: number;
  test_count: number;
  train_class_0: number;
  train_class_1: number;
  validation_class_0: number;
  validation_class_1: number;
  test_class_0: number;
  test_class_1: number;
}

export interface ModelsResponse extends RunContext {
  total: number;
  rows: ModelMetricRow[];
}

export interface PortfolioMetricRow {
  portfolio_type: string;
  model_id: string;
  portfolio_id: string;
  ending_portfolio_value: string;
  total_return: number;
  maximum_drawdown: number;
  volatility: number | null;
  sharpe_ratio: number | null;
  number_of_trades: number;
  fees_paid: string;
  result_versus_buy_and_hold: number;
}

export interface PortfoliosResponse extends RunContext {
  total: number;
  rows: PortfolioMetricRow[];
}

export interface LeaderboardRow extends Omit<ModelMetricRow, "validation_accuracy" | "train_count" | "validation_count" | "test_count" | "train_class_0" | "train_class_1" | "validation_class_0" | "validation_class_1" | "test_class_0" | "test_class_1"> {
  portfolio_id: string;
  ending_portfolio_value: string;
  total_return: number;
  maximum_drawdown: number;
  volatility: number | null;
  sharpe_ratio: number | null;
  number_of_trades: number;
  fees_paid: string;
  result_versus_buy_and_hold: number;
}

export interface LeaderboardResponse extends RunContext {
  total: number;
  rows: LeaderboardRow[];
}

export interface EquityCurveRow {
  date: string;
  series_id: string;
  portfolio_type: string;
  model_id: string | null;
  portfolio_id: string;
  cash: string;
  shares: number;
  market_price: string;
  total_equity: string;
  drawdown: number;
}

export interface DrawdownRow {
  date: string;
  series_id: string;
  portfolio_type: string;
  model_id: string | null;
  portfolio_id: string;
  drawdown: number;
}

export interface PageResponse<T> extends RunContext {
  total: number;
  limit: number;
  offset: number;
  rows: T[];
}

export type EquityCurvesResponse = PageResponse<EquityCurveRow>;
export type DrawdownsResponse = PageResponse<DrawdownRow>;

export interface ExperimentListRow {
  experiment_id: string;
  variant_id: string;
  registered_model_id: string;
  algorithm: string;
  portfolio_id: string;
  symbol: string;
  evaluation_dates: EvaluationDateRange;
  ending_equity: string;
  total_return: number;
  maximum_drawdown: number;
  sharpe_ratio: number | null;
  number_of_trades: number;
}

export interface ExperimentsResponse extends RunContext {
  total: number;
  rows: ExperimentListRow[];
}

export interface ExperimentConfiguration {
  schema_version: string;
  experiment_id: string;
  model_id: string;
  portfolio_id: string;
  benchmark_portfolio_ids: { cash_only: string; buy_and_hold: string };
  symbol: string;
  initial_cash: string;
  thresholds: { buy_probability_up_gte: number; sell_probability_up_lte: number };
  maximum_allocation: number;
  transaction_cost: { type: string; value: number };
  evaluation_dates: EvaluationDateRange;
  execution_policy: {
    signal_timing: string;
    execution_session: string;
    reference_price_field: string;
    slippage_bps: number;
    missing_execution_price: string;
    allocation_mode: string;
    whole_share_rounding: string;
    valuation_price_field: string;
    missing_valuation_price: string;
    ending_position: string;
  };
  metric_assumptions: { annualization_factor: number; risk_free_rate_annual: number };
  created_at: string;
}

export interface ExperimentResult {
  schema_version: string;
  experiment_id: string;
  model_id: string;
  portfolio_id: string;
  prediction_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    support: number;
    positive_label: number;
    average: string;
    zero_division: number;
  };
  initial_equity: string;
  ending_equity: string;
  total_return: number;
  maximum_drawdown: number;
  volatility: number | null;
  sharpe_ratio: number | null;
  number_of_trades: number;
  win_rate: number | null;
  average_gain: string | null;
  average_loss: string | null;
  turnover: number;
  fees_paid: string;
  time_in_market: number;
  comparison_with_buy_and_hold: number;
  benchmark_comparison: {
    cash_only: { portfolio_id: string; ending_equity: string; total_return: number };
    buy_and_hold: { portfolio_id: string; ending_equity: string; total_return: number };
    excess_return_vs_cash_only: number;
    excess_return_vs_buy_and_hold: number;
  };
}

export interface ExperimentDetailResponse extends RunContext {
  variant_id: string;
  algorithm: string;
  configuration: ExperimentConfiguration;
  result: ExperimentResult;
  actual_class_usage: string;
}

export interface TradeRow {
  schema_version: string;
  trade_id: string;
  portfolio_id: string;
  signal_date: string;
  execution_date: string;
  action: string;
  quantity: number;
  reference_price: string;
  execution_price: string;
  fee: string;
  cash_after_trade: string;
  shares_after_trade: number;
}

export interface SnapshotRow {
  schema_version: string;
  portfolio_id: string;
  date: string;
  cash: string;
  shares: number;
  market_price: string;
  position_value: string;
  total_equity: string;
  realized_profit_loss: string;
  unrealized_profit_loss: string;
}

export interface TradesResponse extends PageResponse<TradeRow> {
  experiment_id: string;
  actual_class_usage: string;
}

export interface SnapshotsResponse extends PageResponse<SnapshotRow> {
  experiment_id: string;
}

export interface BenchmarkSnapshotSeries {
  portfolio_id: string;
  ending_equity: string;
  total_return: number;
  total: number;
  limit: number;
  offset: number;
  rows: SnapshotRow[];
}

export interface BenchmarksResponse extends RunContext {
  experiment_id: string;
  cash_only: BenchmarkSnapshotSeries;
  buy_and_hold: BenchmarkSnapshotSeries;
  buy_and_hold_trades: {
    portfolio_id: string;
    total: number;
    limit: number;
    offset: number;
    rows: TradeRow[];
  };
}

export interface SourceResponse extends RunContext {
  source: {
    source: string;
    acquisition_mode: string;
    is_real_data: boolean;
    live_fetch_succeeded: boolean;
    live_acquisition_error_present: boolean;
    original_provider: string;
    legacy_metadata_preserved: boolean;
    producer_component: string | null;
    row_count: number;
    input_prices_sha256: string;
    frozen_prices_sha256: string;
    provenance_sha256: string | null;
    migration_strategy: string | null;
    provenance_strength: string | null;
  };
  pipeline_evidence: {
    schema_version: string | null;
    producer: Record<string, unknown> | string | null;
    generated_at: string | null;
    provider: string | null;
    symbols_requested: string[];
    prices_artifact: Record<string, unknown> | null;
  } | null;
  migration_evidence: {
    schema_version: string | null;
    status: string | null;
    producer: Record<string, unknown> | string | null;
    migration_strategy: string | null;
    provenance_strength: string | null;
    limitations: string[];
  } | null;
}

export interface HistoricalPriceRow {
  date: string;
  open: string;
  high: string | null;
  low: string | null;
  close: string;
  volume: number;
  source: string;
}

export interface HistoricalPricesResponse {
  schema_version: string;
  run_id: string;
  ticker: string;
  source_mode: string;
  data_date_range: DateRange;
  total: number;
  rows: HistoricalPriceRow[];
  warnings: string[];
  disclaimer: string;
}

export interface HistoricalPricesPageResponse extends HistoricalPricesResponse {
  limit: number;
  offset: number;
}

export interface CoreRunData {
  detail: VerticalSliceDetailResponse;
  summary: VerticalSliceSummaryResponse;
  leaderboard: LeaderboardResponse;
  models: ModelsResponse;
  portfolios: PortfoliosResponse;
  source: SourceResponse;
  experiments: ExperimentsResponse;
}

export type UnifiedModelVariant =
  | "M-001"
  | "M-002"
  | "M-003"
  | "M-004"
  | "M-005"
  | "M-006";

export interface UnifiedComparisonContext {
  schema_version: string;
  comparison_id: string;
  status: "PASS";
  symbol: "OGDC";
  holdout_term: "shared project holdout";
  outcomes_are_not_combined: true;
  limitations: string[];
  disclaimer: string;
}

export interface UnifiedComparisonListItem {
  schema_version: string;
  comparison_id: string;
  status: "PASS";
  symbol: "OGDC";
  model_count: 6;
  validation_count: number;
  shared_holdout_count: number;
  shared_holdout_target_start: string;
  shared_holdout_target_end: string;
  holdout_term: "shared project holdout";
  is_default: boolean;
}

export interface UnifiedComparisonListResponse {
  schema_version: string;
  total: number;
  limit: number;
  offset: number;
  default_comparison_id: string | null;
  rows: UnifiedComparisonListItem[];
  disclaimer: string;
}

export interface UnifiedModelIdentity {
  variant_id: UnifiedModelVariant;
  registered_model_id: string;
  algorithm: string;
  configuration_family: string;
}

export interface UnifiedComparisonDetailResponse
  extends UnifiedComparisonContext {
  models: UnifiedModelIdentity[];
  source_frequency: string;
  validation_count: number;
  shared_holdout_count: number;
  selection_policy: {
    primary: string;
    tie_breakers: string[];
    test_metrics_used_for_selection: false;
  };
  presentation: {
    format: string;
    plots_included: false;
    data_sources: string[];
  };
}

export interface UnifiedSourcesResponse extends UnifiedComparisonContext {
  sources: [
    {
      role: "baseline_training";
      artifact_type: "model_training_run";
      run_id: string;
      integrity_root_sha256: string;
      payload_file_count: number;
      integrity_root_derivation: "byte_copy";
    },
    {
      role: "baseline_vertical_slice";
      artifact_type: "vertical_slice";
      run_id: string;
      integrity_root_sha256: string;
      payload_file_count: number;
      integrity_root_derivation: "deterministic_redacted_projection";
    },
    {
      role: "lstm";
      artifact_type: "lstm_price_only_experiment";
      run_id: string;
      integrity_root_sha256: string;
      payload_file_count: number;
      integrity_root_derivation: "byte_copy";
    },
  ];
  relationships: {
    vertical_slice_uses_training_run: true;
    lstm_uses_training_run: true;
    lstm_uses_vertical_slice: true;
  };
}

export interface UnifiedPredictionOutcome {
  variant_id: UnifiedModelVariant;
  registered_model_id: string;
  criterion: string;
  scope: string;
  accuracy: number;
  balanced_accuracy: number | null;
  f1: number;
}

export interface UnifiedPortfolioOutcome {
  variant_id: UnifiedModelVariant;
  registered_model_id: string;
  portfolio_id: string;
  criterion: string;
  ending_equity: string;
  total_return: number;
  maximum_drawdown: number;
  fees_paid: string;
}

export interface UnifiedBenchmarkOutcome {
  initial_equity: string;
  ending_equity: string;
  total_return: number;
  maximum_drawdown: number;
  volatility: number | null;
  sharpe_ratio: number | null;
  number_of_trades: number;
  fees_paid: string;
}

export interface UnifiedComparisonSummaryResponse
  extends UnifiedComparisonContext {
  holdout_context: {
    term: "shared project holdout";
    common_count: number;
    target_start: string;
    target_end: string;
    previously_exposed_through: UnifiedModelVariant[];
  };
  reported_outcomes: {
    validation_selected_model: UnifiedPredictionOutcome;
    lstm_validation_preferred_model: UnifiedPredictionOutcome;
    best_shared_holdout_balanced_accuracy: UnifiedPredictionOutcome;
    best_shared_holdout_f1: UnifiedPredictionOutcome;
    best_simulated_model_portfolio: UnifiedPortfolioOutcome;
    lowest_drawdown_model_portfolio: UnifiedPortfolioOutcome;
    lowest_fee_model_portfolio: UnifiedPortfolioOutcome;
    cash_only_benchmark: UnifiedBenchmarkOutcome;
    buy_and_hold_result: UnifiedBenchmarkOutcome;
    any_model_beat_buy_and_hold: {
      value: boolean;
      best_model_variant_id: UnifiedModelVariant;
      best_model_minus_buy_and_hold: number;
      comparison_metric: string;
    };
  };
}

export interface UnifiedPredictionMetricRow {
  competitor_id: string;
  competitor_type: "model" | "reference_baseline";
  registered_model_id: string | null;
  algorithm: string | null;
  configuration_family: string | null;
  evaluation_scope: "validation" | "shared project holdout";
  row_count: number;
  support: number;
  accuracy: number;
  balanced_accuracy: number | null;
  precision: number;
  recall: number;
  f1: number;
  matthews_corrcoef: number | null;
  roc_auc: number | null;
  brier_score: number;
  log_loss: number | null;
  expected_calibration_error: number;
  selection_eligible: boolean;
  metric_warnings: string[];
  class_counts: Record<string, number>;
  predicted_class_counts: Record<string, number>;
  confusion_matrix: [[number, number], [number, number]];
}

export interface UnifiedPredictionMetricsResponse
  extends UnifiedComparisonContext {
  models: UnifiedPredictionMetricRow[];
  reference_baselines: UnifiedPredictionMetricRow[];
}

export interface UnifiedValidationMetricsResponse
  extends UnifiedPredictionMetricsResponse {
  evaluation_scope: "validation";
}

export interface UnifiedModelPortfolioMetric {
  variant_id: UnifiedModelVariant;
  registered_model_id: string;
  experiment_id: string;
  portfolio_id: string;
  initial_equity: string;
  ending_equity: string;
  total_return: number;
  maximum_drawdown: number;
  volatility: number | null;
  sharpe_ratio: number | null;
  number_of_trades: number;
  win_rate: number | null;
  average_gain: string | null;
  average_loss: string | null;
  turnover: number;
  fees_paid: string;
  time_in_market: number;
  excess_return_vs_cash_only: number;
  excess_return_vs_buy_and_hold: number;
}

export interface UnifiedBenchmarkPortfolioMetric {
  benchmark_id: "cash-only" | "buy-and-hold";
  initial_equity: string;
  ending_equity: string;
  total_return: number;
  maximum_drawdown: number;
  volatility: number | null;
  sharpe_ratio: number | null;
  number_of_trades: number;
  win_rate: number | null;
  average_gain: string | null;
  average_loss: string | null;
  turnover: number;
  fees_paid: string;
  time_in_market: number;
  ledger_count: 6;
}

export interface UnifiedPortfolioMetricsResponse
  extends UnifiedComparisonContext {
  models: UnifiedModelPortfolioMetric[];
  benchmarks: UnifiedBenchmarkPortfolioMetric[];
}

export interface UnifiedPredictionRankingRow {
  rank: number;
  competitor_id: UnifiedModelVariant;
  registered_model_id: string;
  accuracy: number;
  balanced_accuracy: number | null;
  f1: number;
}

export interface UnifiedPredictionRankingsResponse
  extends UnifiedComparisonContext {
  validation_model_selection: UnifiedPredictionRankingRow[];
  lstm_validation_preference: UnifiedPredictionRankingRow[];
  shared_project_holdout_balanced_accuracy: UnifiedPredictionRankingRow[];
  shared_project_holdout_f1: UnifiedPredictionRankingRow[];
}

export interface UnifiedPortfolioRankingRow {
  rank: number;
  competitor_id: UnifiedModelVariant;
  registered_model_id: string;
  total_return: number | null;
  maximum_drawdown: number | null;
  sharpe_ratio: number | null;
  fees_paid: string | null;
  number_of_trades: number | null;
}

export interface UnifiedPortfolioRankingsResponse
  extends UnifiedComparisonContext {
  simulated_model_portfolio_return: UnifiedPortfolioRankingRow[];
  simulated_model_portfolio_drawdown: UnifiedPortfolioRankingRow[];
  simulated_model_portfolio_fees: UnifiedPortfolioRankingRow[];
}

export interface UnifiedCommonDateKey {
  ticker: string;
  feature_date: string;
  target_date: string;
  actual_class: 0 | 1;
}

export interface UnifiedDateScope {
  terminology: string;
  common_count: number;
  feature_start: string;
  feature_end: string;
  target_start: string;
  target_end: string;
  original_counts: Record<string, number>;
  common_counts: Record<string, number>;
  dropped_target_dates: Record<string, string[]>;
  common_keys: UnifiedCommonDateKey[];
}

export interface UnifiedCommonDateResponse extends UnifiedComparisonContext {
  alignment_key: string[];
  validation: UnifiedDateScope;
  shared_project_holdout: UnifiedDateScope & {
    previously_exposed_through: UnifiedModelVariant[];
  };
}

export interface UnifiedConfusionMatrixRow {
  competitor_id: string;
  competitor_type: "model" | "reference_baseline";
  registered_model_id: string | null;
  evaluation_scope: "validation" | "shared project holdout";
  class_labels: [0, 1];
  confusion_matrix: [[number, number], [number, number]];
  support: number;
}

export interface UnifiedConfusionMatricesResponse
  extends UnifiedComparisonContext {
  validation_models: UnifiedConfusionMatrixRow[];
  validation_reference_baselines: UnifiedConfusionMatrixRow[];
  shared_project_holdout_models: UnifiedConfusionMatrixRow[];
  shared_project_holdout_reference_baselines: UnifiedConfusionMatrixRow[];
}

export interface UnifiedComparisonData {
  detail: UnifiedComparisonDetailResponse;
  sources: UnifiedSourcesResponse;
  summary: UnifiedComparisonSummaryResponse;
  predictionMetrics: UnifiedPredictionMetricsResponse;
  validationMetrics: UnifiedValidationMetricsResponse;
  portfolioMetrics: UnifiedPortfolioMetricsResponse;
  predictionRankings: UnifiedPredictionRankingsResponse;
  portfolioRankings: UnifiedPortfolioRankingsResponse;
  commonDateContract: UnifiedCommonDateResponse;
  confusionMatrices: UnifiedConfusionMatricesResponse;
}
