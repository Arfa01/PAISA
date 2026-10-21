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
  ticker: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number;
  volume: number | null;
  source: string | null;
}

export interface HistoricalPricesResponse {
  ticker: string;
  count: number;
  rows: HistoricalPriceRow[];
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
