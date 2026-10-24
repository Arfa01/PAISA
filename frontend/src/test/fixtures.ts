import { vi } from "vitest";

import type { PaisaApiClient } from "../api/client";
import type {
  BenchmarksResponse,
  CoreRunData,
  DrawdownRow,
  EquityCurveRow,
  ExperimentDetailResponse,
  ExperimentsResponse,
  HistoricalPricesResponse,
  LeaderboardRow,
  ModelMetricRow,
  PageResponse,
  PortfolioMetricRow,
  RunContext,
  SnapshotRow,
  SnapshotsResponse,
  SourceResponse,
  TradeRow,
  TradesResponse,
  UnifiedCommonDateKey,
  UnifiedComparisonContext,
  UnifiedComparisonData,
  UnifiedComparisonListResponse,
  UnifiedModelVariant,
  UnifiedPredictionMetricRow,
  VerticalSliceListItem,
  VerticalSliceListResponse,
} from "../api/types";

export const RUN_ID = "VSL-TEST-OGDC-01";
export const OTHER_RUN_ID = "VSL-TEST-OGDC-02";
export const EXPERIMENT_ONE = `${RUN_ID}-m-001`;
export const EXPERIMENT_TWO = `${RUN_ID}-m-002`;
export const DISCLAIMER = "Academic simulation; not financial advice.";
export const COMPARISON_ID = "CMP-TEST-OGDC-price-only-01";
export const OTHER_COMPARISON_ID = "CMP-TEST-OGDC-price-only-02";
export const UNIFIED_MODELS: UnifiedModelVariant[] = [
  "M-001",
  "M-002",
  "M-003",
  "M-004",
  "M-005",
  "M-006",
];

export function runContext(runId = RUN_ID): RunContext {
  return {
    schema_version: "1.0",
    run_id: runId,
    source_mode: "verified_legacy_psx_cache",
    data_date_range: { start: "2021-09-16", end: "2026-09-15" },
    warnings: [
      "Prices are not adjusted for corporate actions and dividends are not included.",
      "The prediction target is close-to-close; portfolio returns begin at the next-session open.",
    ],
    limitations: [
      "actual_class is evaluation-only and is never used for trading decisions.",
    ],
    disclaimer: DISCLAIMER,
  };
}
export function makeRunListItem(
  runId = RUN_ID,
  symbol = "OGDC",
): VerticalSliceListItem {
  return {
    schema_version: "1.0",
    run_id: runId,
    status: "PASS",
    symbol,
    completed_at: "2026-10-20T20:00:00Z",
    source_mode: "verified_legacy_psx_cache",
    data_date_range: { start: "2021-09-16", end: "2026-09-15" },
    model_training_run_id: `TRN-${runId}`,
    model_count: 4,
    experiment_count: 4,
  };
}

export function makeRunList(
  rows: VerticalSliceListItem[] = [makeRunListItem()],
): VerticalSliceListResponse {
  return {
    schema_version: "1.0",
    total: rows.length,
    limit: 1000,
    offset: 0,
    rows,
    disclaimer: DISCLAIMER,
  };
}

const modelRows: ModelMetricRow[] = [
  {
    model_id: "M-001",
    registered_model_id: "registered-m-001",
    algorithm: "random_forest",
    validation_accuracy: 0.54,
    validation_balanced_accuracy: 0.53,
    validation_f1: 0.46,
    test_accuracy: 0.52,
    test_balanced_accuracy: 0.51,
    test_f1: 0.23,
    train_count: 700,
    validation_count: 220,
    test_count: 184,
    train_class_0: 350,
    train_class_1: 350,
    validation_class_0: 110,
    validation_class_1: 110,
    test_class_0: 92,
    test_class_1: 92,
  },
  {
    model_id: "M-002",
    registered_model_id: "registered-m-002",
    algorithm: "random_forest",
    validation_accuracy: 0.51,
    validation_balanced_accuracy: 0.5,
    validation_f1: 0.42,
    test_accuracy: 0.55,
    test_balanced_accuracy: 0.54,
    test_f1: 0.49,
    train_count: 700,
    validation_count: 220,
    test_count: 184,
    train_class_0: 350,
    train_class_1: 350,
    validation_class_0: 110,
    validation_class_1: 110,
    test_class_0: 92,
    test_class_1: 92,
  },
  {
    model_id: "M-003",
    registered_model_id: "registered-m-003",
    algorithm: "random_forest",
    validation_accuracy: 0.5,
    validation_balanced_accuracy: 0.49,
    validation_f1: 0.4,
    test_accuracy: 0.51,
    test_balanced_accuracy: 0.5,
    test_f1: 0.45,
    train_count: 700,
    validation_count: 220,
    test_count: 184,
    train_class_0: 350,
    train_class_1: 350,
    validation_class_0: 110,
    validation_class_1: 110,
    test_class_0: 92,
    test_class_1: 92,
  },
  {
    model_id: "M-004",
    registered_model_id: "registered-m-004",
    algorithm: "xgboost",
    validation_accuracy: 0.48,
    validation_balanced_accuracy: 0.47,
    validation_f1: 0.38,
    test_accuracy: 0.49,
    test_balanced_accuracy: 0.48,
    test_f1: 0.43,
    train_count: 700,
    validation_count: 220,
    test_count: 184,
    train_class_0: 350,
    train_class_1: 350,
    validation_class_0: 110,
    validation_class_1: 110,
    test_class_0: 92,
    test_class_1: 92,
  },
];

const portfolioRows: PortfolioMetricRow[] = [
  {
    portfolio_type: "model",
    model_id: "M-001",
    portfolio_id: "portfolio-m-001",
    ending_portfolio_value: "99298.24",
    total_return: -0.0070176,
    maximum_drawdown: 0.0219,
    volatility: 0.0287,
    sharpe_ratio: -0.32,
    number_of_trades: 2,
    fees_paid: "99.06",
    result_versus_buy_and_hold: -0.0694,
  },
  {
    portfolio_type: "model",
    model_id: "M-002",
    portfolio_id: "portfolio-m-002",
    ending_portfolio_value: "101250.75",
    total_return: 0.0125075,
    maximum_drawdown: 0.018,
    volatility: 0.026,
    sharpe_ratio: 0.41,
    number_of_trades: 4,
    fees_paid: "145.20",
    result_versus_buy_and_hold: -0.0499,
  },
  {
    portfolio_type: "model",
    model_id: "M-003",
    portfolio_id: "portfolio-m-003",
    ending_portfolio_value: "100420.00",
    total_return: 0.0042,
    maximum_drawdown: 0.01,
    volatility: 0.019,
    sharpe_ratio: 0.2,
    number_of_trades: 3,
    fees_paid: "120.00",
    result_versus_buy_and_hold: -0.0582,
  },
  {
    portfolio_type: "model",
    model_id: "M-004",
    portfolio_id: "portfolio-m-004",
    ending_portfolio_value: "97500.00",
    total_return: -0.025,
    maximum_drawdown: 0.04,
    volatility: 0.04,
    sharpe_ratio: -0.7,
    number_of_trades: 38,
    fees_paid: "1800.00",
    result_versus_buy_and_hold: -0.0874,
  },
  {
    portfolio_type: "cash_only",
    model_id: "CASH-ONLY",
    portfolio_id: "portfolio-cash",
    ending_portfolio_value: "100000.00",
    total_return: 0,
    maximum_drawdown: 0,
    volatility: 0,
    sharpe_ratio: null,
    number_of_trades: 0,
    fees_paid: "0.00",
    result_versus_buy_and_hold: -0.0624,
  },
  {
    portfolio_type: "buy_and_hold",
    model_id: "BUY-AND-HOLD",
    portfolio_id: "portfolio-buy-hold",
    ending_portfolio_value: "106240.50",
    total_return: 0.062405,
    maximum_drawdown: 0.03,
    volatility: 0.12,
    sharpe_ratio: 0.55,
    number_of_trades: 2,
    fees_paid: "100.00",
    result_versus_buy_and_hold: 0,
  },
];

function makeExperiments(runId: string): ExperimentsResponse {
  const context = runContext(runId);
  const rows = ["M-001", "M-002"].map((variantId, index) => ({
    experiment_id: `${runId}-m-00${index + 1}`,
    variant_id: variantId,
    registered_model_id: `registered-m-00${index + 1}`,
    algorithm: "random_forest",
    portfolio_id: `portfolio-m-00${index + 1}`,
    symbol: "OGDC",
    evaluation_dates: {
      feature_start: "2025-12-14",
      feature_end: "2026-09-14",
      target_start: "2025-12-15",
      target_end: "2026-09-15",
    },
    ending_equity: index === 0 ? "99298.24" : "101250.75",
    total_return: index === 0 ? -0.0070176 : 0.0125075,
    maximum_drawdown: index === 0 ? 0.0219 : 0.018,
    sharpe_ratio: index === 0 ? -0.32 : 0.41,
    number_of_trades: index === 0 ? 2 : 4,
  }));
  return { ...context, total: rows.length, rows };
}

export function makeCoreRunData(runId = RUN_ID): CoreRunData {
  const context = runContext(runId);
  const models = modelRows.map((row) => ({ ...row }));
  const portfolios = portfolioRows.map((row) => ({ ...row }));
  const leaderboardRows: LeaderboardRow[] = models.map((model, index) => {
    const portfolio = portfolios[index] ?? portfolios[0]!;
    return {
      model_id: model.model_id,
      registered_model_id: model.registered_model_id,
      algorithm: model.algorithm,
      validation_balanced_accuracy: model.validation_balanced_accuracy,
      validation_f1: model.validation_f1,
      test_accuracy: model.test_accuracy,
      test_balanced_accuracy: model.test_balanced_accuracy,
      test_f1: model.test_f1,
      portfolio_id: portfolio.portfolio_id,
      ending_portfolio_value: portfolio.ending_portfolio_value,
      total_return: portfolio.total_return,
      maximum_drawdown: portfolio.maximum_drawdown,
      volatility: portfolio.volatility,
      sharpe_ratio: portfolio.sharpe_ratio,
      number_of_trades: portfolio.number_of_trades,
      fees_paid: portfolio.fees_paid,
      result_versus_buy_and_hold: portfolio.result_versus_buy_and_hold,
    };
  });
  const source: SourceResponse = {
    ...context,
    source: {
      source: "legacy_psx_cache",
      acquisition_mode: "verified_legacy_psx_cache",
      is_real_data: true,
      live_fetch_succeeded: false,
      live_acquisition_error_present: true,
      original_provider: "psx_dps",
      legacy_metadata_preserved: true,
      producer_component: "paisa.legacy_cache",
      row_count: 1234,
      input_prices_sha256: "a".repeat(64),
      frozen_prices_sha256: "b".repeat(64),
      provenance_sha256: "c".repeat(64),
      migration_strategy: "copy_and_seal",
      provenance_strength: "checksum_bound",
    },
    pipeline_evidence: {
      schema_version: "1.0",
      producer: "paisa.pipeline",
      generated_at: "2026-10-20T18:00:00Z",
      provider: "psx_dps",
      symbols_requested: ["OGDC"],
      prices_artifact: null,
    },
    migration_evidence: {
      schema_version: "1.0",
      status: "PASS",
      producer: "paisa.legacy_cache",
      migration_strategy: "copy_and_seal",
      provenance_strength: "checksum_bound",
      limitations: [
        "Historical prices do not independently establish corporate-action adjustment or dividends.",
      ],
    },
  };
  const experiments = makeExperiments(runId);

  return {
    detail: {
      ...context,
      status: "PASS",
      symbol: "OGDC",
      created_at: "2026-10-20T19:00:00Z",
      evaluation_dates: {
        feature_start: "2025-12-14",
        feature_end: "2026-09-14",
        target_start: "2025-12-15",
        target_end: "2026-09-15",
      },
      model_training_run_id: `TRN-${runId}`,
      model_variants: ["M-001", "M-002", "M-003", "M-004"],
      registered_model_ids: Object.fromEntries(
        models.map((row) => [row.model_id, row.registered_model_id]),
      ),
      experiment_ids: experiments.rows.map((row) => row.experiment_id),
      shared_experiment_assumptions: {
        initial_cash_pkr: "100000.00",
        buy_probability_up_gte: 0.6,
        sell_probability_up_lte: 0.4,
        maximum_allocation: 1,
        long_only: true,
        whole_shares: true,
        transaction_cost_bps: 10,
        slippage_bps: 0,
        signal_timing: "after_close",
        execution: "next_session_open",
        valuation: "session_close",
        ending_position: "mark_to_market",
      },
    },
    summary: {
      ...context,
      status: "PASS",
      symbol: "OGDC",
      completed_at: "2026-10-20T20:00:00Z",
      model_training_run_id: `TRN-${runId}`,
      evaluation_dates: {
        feature_start: "2025-12-14",
        feature_end: "2026-09-14",
        target_start: "2025-12-15",
        target_end: "2026-09-15",
      },
      best_validation_prediction_model: {
        model_id: "M-001",
        registered_model_id: "registered-m-001",
        algorithm: "random_forest",
        validation_balanced_accuracy: 0.53,
        validation_f1: 0.46,
        selection_basis: "validation_balanced_accuracy_then_f1",
      },
      best_test_period_simulated_portfolio: {
        model_id: "M-002",
        portfolio_id: "portfolio-m-002",
        ending_portfolio_value: "101250.75",
        total_return: 0.0125075,
        sharpe_ratio: 0.41,
        selection_basis: "highest_ending_portfolio_value",
      },
      lowest_drawdown_portfolio: {
        model_id: "M-003",
        portfolio_id: "portfolio-m-003",
        maximum_drawdown: 0.01,
        total_return: 0.0042,
      },
      buy_and_hold_result: {
        portfolio_id: "portfolio-buy-hold",
        ending_portfolio_value: "106240.50",
        total_return: 0.062405,
        maximum_drawdown: 0.03,
        fees_paid: "100.00",
      },
      any_model_portfolio_beat_buy_and_hold: false,
      selection_policy:
        "The prediction champion is selected on validation balanced accuracy.",
      target_interpretation: "Next-session close-to-close direction.",
      portfolio_return_interpretation:
        "Trades begin no earlier than the next-session open.",
      training_data: {
        provider: "psx_dps",
        actual_feature_start: "2021-09-16",
        actual_feature_end: "2026-09-14",
        dataset_sha256: "d".repeat(64),
      },
    },
    leaderboard: {
      ...context,
      total: leaderboardRows.length,
      rows: leaderboardRows,
    },
    models: { ...context, total: models.length, rows: models },
    portfolios: { ...context, total: portfolios.length, rows: portfolios },
    source,
    experiments,
  };
}

export function makeExperimentDetail(
  runId = RUN_ID,
  experimentId = EXPERIMENT_ONE,
  variantId = "M-001",
): ExperimentDetailResponse {
  const portfolioId = `portfolio-${variantId.toLowerCase()}`;
  return {
    ...runContext(runId),
    variant_id: variantId,
    algorithm: "random_forest",
    configuration: {
      schema_version: "1.0",
      experiment_id: experimentId,
      model_id: `registered-${variantId.toLowerCase()}`,
      portfolio_id: portfolioId,
      benchmark_portfolio_ids: {
        cash_only: "portfolio-cash",
        buy_and_hold: "portfolio-buy-hold",
      },
      symbol: "OGDC",
      initial_cash: "100000.00",
      thresholds: {
        buy_probability_up_gte: 0.6,
        sell_probability_up_lte: 0.4,
      },
      maximum_allocation: 1,
      transaction_cost: { type: "basis_points", value: 10 },
      evaluation_dates: {
        feature_start: "2025-12-14",
        feature_end: "2026-09-14",
        target_start: "2025-12-15",
        target_end: "2026-09-15",
      },
      execution_policy: {
        signal_timing: "after_close",
        execution_session: "next_session",
        reference_price_field: "open",
        slippage_bps: 0,
        missing_execution_price: "skip_trade",
        allocation_mode: "maximum_allocation",
        whole_share_rounding: "floor",
        valuation_price_field: "close",
        missing_valuation_price: "fail",
        ending_position: "mark_to_market",
      },
      metric_assumptions: { annualization_factor: 252, risk_free_rate_annual: 0 },
      created_at: "2026-10-20T19:00:00Z",
    },
    result: {
      schema_version: "1.0",
      experiment_id: experimentId,
      model_id: `registered-${variantId.toLowerCase()}`,
      portfolio_id: portfolioId,
      prediction_metrics: {
        accuracy: 0.52,
        precision: 0.5,
        recall: 0.4,
        f1: 0.44,
        support: 184,
        positive_label: 1,
        average: "binary",
        zero_division: 0,
      },
      initial_equity: "100000.00",
      ending_equity: variantId === "M-002" ? "101250.75" : "99298.24",
      total_return: variantId === "M-002" ? 0.0125075 : -0.0070176,
      maximum_drawdown: 0.02,
      volatility: 0.03,
      sharpe_ratio: 0.2,
      number_of_trades: 2,
      win_rate: 0.5,
      average_gain: "200.00",
      average_loss: "-100.00",
      turnover: 1.1,
      fees_paid: "99.06",
      time_in_market: 0.5,
      comparison_with_buy_and_hold: -0.06,
      benchmark_comparison: {
        cash_only: {
          portfolio_id: "portfolio-cash",
          ending_equity: "100000.00",
          total_return: 0,
        },
        buy_and_hold: {
          portfolio_id: "portfolio-buy-hold",
          ending_equity: "106240.50",
          total_return: 0.062405,
        },
        excess_return_vs_cash_only: -0.0070176,
        excess_return_vs_buy_and_hold: -0.0694,
      },
    },
    actual_class_usage:
      "Actual classes are used only for evaluation, never for execution.",
  };
}

export function makeTrade(offset = 0, experimentId = EXPERIMENT_ONE): TradeRow {
  return {
    schema_version: "1.0",
    trade_id: `${experimentId}-trade-${offset + 1}`,
    portfolio_id: `portfolio-${experimentId}`,
    signal_date: offset === 0 ? "2026-01-02" : "2026-02-02",
    execution_date: offset === 0 ? "2026-01-05" : "2026-02-03",
    action: offset === 0 ? "BUY" : "SELL",
    quantity: 10,
    reference_price: "100.00",
    execution_price: "101.00",
    fee: "10.10",
    cash_after_trade: "98979.90",
    shares_after_trade: offset === 0 ? 10 : 0,
  };
}

export function makeTradesResponse(
  offset = 0,
  experimentId = EXPERIMENT_ONE,
  runId = RUN_ID,
): TradesResponse {
  return {
    ...runContext(runId),
    experiment_id: experimentId,
    actual_class_usage: "Evaluation only.",
    total: 12,
    limit: 10,
    offset,
    rows: [makeTrade(offset, experimentId)],
  };
}

const snapshot: SnapshotRow = {
  schema_version: "1.0",
  portfolio_id: "portfolio-m-001",
  date: "2026-01-05",
  cash: "98979.90",
  shares: 10,
  market_price: "102.00",
  position_value: "1020.00",
  total_equity: "99999.90",
  realized_profit_loss: "0.00",
  unrealized_profit_loss: "10.00",
};

export function makeSnapshotsResponse(
  experimentId = EXPERIMENT_ONE,
  runId = RUN_ID,
): SnapshotsResponse {
  return {
    ...runContext(runId),
    experiment_id: experimentId,
    total: 1,
    limit: 10,
    offset: 0,
    rows: [{ ...snapshot }],
  };
}

function benchmarkPage<T>(rows: T[]): {
  portfolio_id: string;
  ending_equity: string;
  total_return: number;
  total: number;
  limit: number;
  offset: number;
  rows: T[];
} {
  return {
    portfolio_id: "portfolio-benchmark",
    ending_equity: "100000.00",
    total_return: 0,
    total: rows.length,
    limit: 10,
    offset: 0,
    rows,
  };
}
export function makeBenchmarksResponse(
  experimentId = EXPERIMENT_ONE,
  runId = RUN_ID,
): BenchmarksResponse {
  return {
    ...runContext(runId),
    experiment_id: experimentId,
    cash_only: benchmarkPage([{ ...snapshot, portfolio_id: "portfolio-cash" }]),
    buy_and_hold: benchmarkPage([
      { ...snapshot, portfolio_id: "portfolio-buy-hold" },
    ]),
    buy_and_hold_trades: {
      portfolio_id: "portfolio-buy-hold",
      total: 1,
      limit: 10,
      offset: 0,
      rows: [makeTrade(0, experimentId)],
    },
  };
}

export const equityRows: EquityCurveRow[] = portfolioRows.map((portfolio, index) => ({
  date: "2026-01-05",
  series_id: portfolio.model_id,
  portfolio_type: portfolio.portfolio_type,
  model_id: portfolio.portfolio_type === "model" ? portfolio.model_id : null,
  portfolio_id: portfolio.portfolio_id,
  cash: portfolio.portfolio_type === "buy_and_hold" ? "0.00" : "98979.90",
  shares: portfolio.portfolio_type === "cash_only" ? 0 : 10 + index,
  market_price: "102.00",
  total_equity: portfolio.ending_portfolio_value,
  drawdown: portfolio.maximum_drawdown,
}));

export const drawdownRows: DrawdownRow[] = equityRows.map((row) => ({
  date: row.date,
  series_id: row.series_id,
  portfolio_type: row.portfolio_type,
  model_id: row.model_id,
  portfolio_id: row.portfolio_id,
  drawdown: row.drawdown,
}));

export function makeHistoricalPrices(runId = RUN_ID): HistoricalPricesResponse {
  return {
    schema_version: "1.0",
    run_id: runId,
    ticker: "OGDC",
    source_mode: "verified_legacy_psx_cache",
    data_date_range: { start: "2021-09-16", end: "2026-09-15" },
    total: 2,
    rows: [
      {
        date: "2021-09-16",
        open: "88.09",
        high: null,
        low: null,
        close: "89.17",
        volume: 4_201_784,
        source: "psx_dps",
      },
      {
        date: "2026-09-15",
        open: "212.04",
        high: null,
        low: null,
        close: "213.58",
        volume: 500_000,
        source: "psx_dps",
      },
    ],
    warnings: [
      "Historical prices are shown as immutable source observations.",
    ],
    disclaimer: DISCLAIMER,
  };
}

function unifiedContext(
  comparisonId = COMPARISON_ID,
): UnifiedComparisonContext {
  return {
    schema_version: "1.0",
    comparison_id: comparisonId,
    status: "PASS",
    symbol: "OGDC",
    holdout_term: "shared project holdout",
    outcomes_are_not_combined: true,
    limitations: [
      "One stock and one historical period do not establish general superiority.",
      "The shared project holdout was exposed previously.",
    ],
    disclaimer: DISCLAIMER,
  };
}

const configurationFamilies: Record<UnifiedModelVariant, string> = {
  "M-001": "PAISA v0 baseline",
  "M-002": "conservative",
  "M-003": "flexible",
  "M-004": "CPU histogram",
  "M-005": "20-session baseline recurrent",
  "M-006": "20-session regularized recurrent",
};

function algorithmFor(model: UnifiedModelVariant): string {
  if (model === "M-004") return "xgboost";
  if (model === "M-005" || model === "M-006") return "lstm";
  return "random_forest";
}

function registeredModelId(model: UnifiedModelVariant): string {
  return `registered-${model.toLowerCase()}`;
}

function predictionMetric(
  model: UnifiedModelVariant,
  scope: "validation" | "shared project holdout",
  index: number,
): UnifiedPredictionMetricRow {
  const rowCount = scope === "validation" ? 182 : 184;
  const classZero = scope === "validation" ? 93 : 93;
  const classOne = rowCount - classZero;
  const trueNegative = 50;
  const falsePositive = classZero - trueNegative;
  const falseNegative = 40;
  const truePositive = classOne - falseNegative;
  return {
    competitor_id: model,
    competitor_type: "model",
    registered_model_id: registeredModelId(model),
    algorithm: algorithmFor(model),
    configuration_family: configurationFamilies[model],
    evaluation_scope: scope,
    row_count: rowCount,
    support: rowCount,
    accuracy: 0.49 + index * 0.01,
    balanced_accuracy: model === "M-003" ? null : 0.48 + index * 0.01,
    precision: 0.45 + index * 0.01,
    recall: 0.44 + index * 0.01,
    f1: 0.43 + index * 0.02,
    matthews_corrcoef: model === "M-003" ? null : 0.02 + index * 0.01,
    roc_auc: model === "M-003" ? null : 0.5 + index * 0.01,
    brier_score: 0.25,
    log_loss: 0.7,
    expected_calibration_error: 0.05,
    selection_eligible: true,
    metric_warnings: [],
    class_counts: { "0": classZero, "1": classOne },
    predicted_class_counts: {
      "0": trueNegative + falseNegative,
      "1": falsePositive + truePositive,
    },
    confusion_matrix: [
      [trueNegative, falsePositive],
      [falseNegative, truePositive],
    ],
  };
}

function referenceMetric(
  competitorId: "majority-class-baseline" | "previous-direction-baseline",
  scope: "validation" | "shared project holdout",
): UnifiedPredictionMetricRow {
  const base = predictionMetric("M-001", scope, 0);
  return {
    ...base,
    competitor_id: competitorId,
    competitor_type: "reference_baseline",
    registered_model_id: null,
    algorithm: null,
    configuration_family: null,
    selection_eligible: false,
  };
}

function commonDateKeys(count: number, start: string): UnifiedCommonDateKey[] {
  const base = new Date(`${start}T00:00:00.000Z`);
  return Array.from({ length: count }, (_, index) => {
    const target = new Date(base);
    target.setUTCDate(target.getUTCDate() + index);
    const feature = new Date(target);
    feature.setUTCDate(feature.getUTCDate() - 1);
    return {
      ticker: "OGDC",
      feature_date: feature.toISOString().slice(0, 10),
      target_date: target.toISOString().slice(0, 10),
      actual_class: index % 2 === 0 ? 0 : 1,
    };
  });
}

export function makeUnifiedComparisonList(
  rows = [
    {
      schema_version: "1.0",
      comparison_id: COMPARISON_ID,
      status: "PASS" as const,
      symbol: "OGDC" as const,
      model_count: 6 as const,
      validation_count: 182,
      shared_holdout_count: 184,
      shared_holdout_target_start: "2025-12-16",
      shared_holdout_target_end: "2026-09-15",
      holdout_term: "shared project holdout" as const,
      is_default: true,
    },
  ],
  defaultComparisonId: string | null = rows[0]?.comparison_id ?? null,
): UnifiedComparisonListResponse {
  return {
    schema_version: "1.0",
    total: rows.length,
    limit: 1000,
    offset: 0,
    default_comparison_id: defaultComparisonId,
    rows: rows.map((row) => ({
      ...row,
      is_default: row.comparison_id === defaultComparisonId,
    })),
    disclaimer: DISCLAIMER,
  };
}

export function makeUnifiedComparisonData(
  comparisonId = COMPARISON_ID,
): UnifiedComparisonData {
  const context = unifiedContext(comparisonId);
  const validationModels = UNIFIED_MODELS.map((model, index) =>
    predictionMetric(model, "validation", index),
  );
  const predictionModels = UNIFIED_MODELS.map((model, index) =>
    predictionMetric(model, "shared project holdout", index),
  );
  const validationBaselines = [
    referenceMetric("majority-class-baseline", "validation"),
    referenceMetric("previous-direction-baseline", "validation"),
  ];
  const predictionBaselines = [
    referenceMetric("majority-class-baseline", "shared project holdout"),
    referenceMetric("previous-direction-baseline", "shared project holdout"),
  ];
  const modelPortfolios = UNIFIED_MODELS.map((model, index) => ({
    variant_id: model,
    registered_model_id: registeredModelId(model),
    experiment_id: `experiment-${model.toLowerCase()}`,
    portfolio_id: `portfolio-${model.toLowerCase()}`,
    initial_equity: "100000.00",
    ending_equity: [
      "99298.24",
      "105987.04",
      "99166.07",
      "100190.05",
      "96359.29",
      "96110.91",
    ][index]!,
    total_return: [-0.0070176, 0.0598704, -0.0083393, 0.0019005, -0.0364071, -0.0388909][index]!,
    maximum_drawdown: [0.0219965, 0.0566995, 0.0118467, 0.0712783, 0.1465748, 0.1546586][index]!,
    volatility: index === 2 ? null : 0.03 + index * 0.01,
    sharpe_ratio: index === 2 ? null : -0.3 + index * 0.2,
    number_of_trades: [2, 3, 2, 34, 3, 3][index]!,
    win_rate: index === 2 ? null : 0.5,
    average_gain: index === 2 ? null : "100.00",
    average_loss: index === 2 ? null : "-50.00",
    turnover: 1 + index * 0.1,
    fees_paid: ["99.06", "160.95", "98.93", "1751.06", "146.08", "146.72"][index]!,
    time_in_market: 0.1 + index * 0.1,
    excess_return_vs_cash_only: [-0.0070176, 0.0598704, -0.0083393, 0.0019005, -0.0364071, -0.0388909][index]!,
    excess_return_vs_buy_and_hold: [-0.0694965, -0.0026085, -0.0708182, -0.0605784, -0.098886, -0.1013698][index]!,
  }));
  const benchmarks = [
    {
      benchmark_id: "cash-only" as const,
      initial_equity: "100000.00",
      ending_equity: "100000.00",
      total_return: 0,
      maximum_drawdown: 0,
      volatility: 0,
      sharpe_ratio: null,
      number_of_trades: 0,
      win_rate: null,
      average_gain: null,
      average_loss: null,
      turnover: 0,
      fees_paid: "0.00",
      time_in_market: 0,
      ledger_count: 6 as const,
    },
    {
      benchmark_id: "buy-and-hold" as const,
      initial_equity: "100000.00",
      ending_equity: "106247.89",
      total_return: 0.0624789,
      maximum_drawdown: 0.1279045,
      volatility: 0.1786761,
      sharpe_ratio: 0.553179,
      number_of_trades: 1,
      win_rate: null,
      average_gain: null,
      average_loss: null,
      turnover: 0.473,
      fees_paid: "49.75",
      time_in_market: 1,
      ledger_count: 6 as const,
    },
  ];
  const predictionRanking = UNIFIED_MODELS.map((model, index) => ({
    rank: index + 1,
    competitor_id: model,
    registered_model_id: registeredModelId(model),
    accuracy: predictionModels[index]!.accuracy,
    balanced_accuracy: predictionModels[index]!.balanced_accuracy,
    f1: predictionModels[index]!.f1,
  }));
  const portfolioRanking = UNIFIED_MODELS.map((model, index) => ({
    rank: index + 1,
    competitor_id: model,
    registered_model_id: registeredModelId(model),
    total_return: modelPortfolios[index]!.total_return,
    maximum_drawdown: modelPortfolios[index]!.maximum_drawdown,
    sharpe_ratio: modelPortfolios[index]!.sharpe_ratio,
    fees_paid: modelPortfolios[index]!.fees_paid,
    number_of_trades: modelPortfolios[index]!.number_of_trades,
  }));
  const counts = Object.fromEntries(UNIFIED_MODELS.map((model) => [model, 182]));
  const holdoutCounts = Object.fromEntries(UNIFIED_MODELS.map((model) => [model, 184]));
  const dropped = Object.fromEntries(UNIFIED_MODELS.map((model) => [model, []]));
  const predictionOutcome = (
    variant_id: UnifiedModelVariant,
    criterion: string,
    balanced_accuracy: number,
    f1: number,
  ) => ({
    variant_id,
    registered_model_id: registeredModelId(variant_id),
    criterion,
    scope: "all_six_models",
    accuracy: 0.53,
    balanced_accuracy,
    f1,
  });
  const portfolioOutcome = (
    variant_id: UnifiedModelVariant,
    criterion: string,
  ) => {
    const row = modelPortfolios[UNIFIED_MODELS.indexOf(variant_id)]!;
    return {
      variant_id,
      registered_model_id: registeredModelId(variant_id),
      portfolio_id: row.portfolio_id,
      criterion,
      ending_equity: row.ending_equity,
      total_return: row.total_return,
      maximum_drawdown: row.maximum_drawdown,
      fees_paid: row.fees_paid,
    };
  };

  return {
    detail: {
      ...context,
      models: UNIFIED_MODELS.map((variant_id) => ({
        variant_id,
        registered_model_id: registeredModelId(variant_id),
        algorithm: algorithmFor(variant_id),
        configuration_family: configurationFamilies[variant_id],
      })),
      source_frequency: "daily",
      validation_count: 182,
      shared_holdout_count: 184,
      selection_policy: {
        primary: "balanced_accuracy",
        tie_breakers: ["f1", "accuracy", "model_id_ascending"],
        test_metrics_used_for_selection: false,
      },
      presentation: {
        format: "tabular_report_only",
        plots_included: false,
        data_sources: [
          "prediction-metrics",
          "validation-metrics",
          "portfolio-metrics",
          "prediction-rankings",
          "portfolio-rankings",
          "common-date-contract",
          "confusion-matrices",
        ],
      },
    },
    sources: {
      ...context,
      sources: [
        {
          role: "baseline_training",
          artifact_type: "model_training_run",
          run_id: "TRN-TEST-01",
          integrity_root_sha256: "a".repeat(64),
          payload_file_count: 46,
          integrity_root_derivation: "byte_copy",
        },
        {
          role: "baseline_vertical_slice",
          artifact_type: "vertical_slice",
          run_id: "VSL-TEST-01",
          integrity_root_sha256: "b".repeat(64),
          payload_file_count: 65,
          integrity_root_derivation: "deterministic_redacted_projection",
        },
        {
          role: "lstm",
          artifact_type: "lstm_price_only_experiment",
          run_id: "LSTM-TEST-01",
          integrity_root_sha256: "c".repeat(64),
          payload_file_count: 147,
          integrity_root_derivation: "byte_copy",
        },
      ],
      relationships: {
        vertical_slice_uses_training_run: true,
        lstm_uses_training_run: true,
        lstm_uses_vertical_slice: true,
      },
    },
    summary: {
      ...context,
      holdout_context: {
        term: "shared project holdout",
        common_count: 184,
        target_start: "2025-12-16",
        target_end: "2026-09-15",
        previously_exposed_through: ["M-001", "M-002", "M-003", "M-004"],
      },
      reported_outcomes: {
        validation_selected_model: predictionOutcome(
          "M-001",
          "validation balanced accuracy, then F1, accuracy, and model ID",
          0.530204,
          0.458599,
        ),
        lstm_validation_preferred_model: predictionOutcome(
          "M-005",
          "validation balanced accuracy, then F1, accuracy, and model ID",
          0.47928,
          0.59322,
        ),
        best_shared_holdout_balanced_accuracy: predictionOutcome(
          "M-006",
          "highest shared project holdout balanced accuracy",
          0.544251,
          0.571429,
        ),
        best_shared_holdout_f1: predictionOutcome(
          "M-005",
          "highest shared project holdout F1",
          0.520087,
          0.639676,
        ),
        best_simulated_model_portfolio: portfolioOutcome(
          "M-002",
          "highest simulated total return",
        ),
        lowest_drawdown_model_portfolio: portfolioOutcome(
          "M-003",
          "lowest simulated maximum drawdown",
        ),
        lowest_fee_model_portfolio: portfolioOutcome(
          "M-003",
          "lowest simulated fees paid",
        ),
        cash_only_benchmark: { ...benchmarks[0]! },
        buy_and_hold_result: { ...benchmarks[1]! },
        any_model_beat_buy_and_hold: {
          value: false,
          best_model_variant_id: "M-002",
          best_model_minus_buy_and_hold: -0.0026085,
          comparison_metric: "simulated total return",
        },
      },
    },
    predictionMetrics: {
      ...context,
      models: predictionModels,
      reference_baselines: predictionBaselines,
    },
    validationMetrics: {
      ...context,
      evaluation_scope: "validation",
      models: validationModels,
      reference_baselines: validationBaselines,
    },
    portfolioMetrics: {
      ...context,
      models: modelPortfolios,
      benchmarks,
    },
    predictionRankings: {
      ...context,
      validation_model_selection: predictionRanking,
      lstm_validation_preference: predictionRanking.filter((row) =>
        row.competitor_id === "M-005" || row.competitor_id === "M-006",
      ).map((row, index) => ({ ...row, rank: index + 1 })),
      shared_project_holdout_balanced_accuracy: predictionRanking,
      shared_project_holdout_f1: predictionRanking,
    },
    portfolioRankings: {
      ...context,
      simulated_model_portfolio_return: portfolioRanking,
      simulated_model_portfolio_drawdown: portfolioRanking,
      simulated_model_portfolio_fees: portfolioRanking,
    },
    commonDateContract: {
      ...context,
      alignment_key: ["ticker", "feature_date", "target_date"],
      validation: {
        terminology: "validation",
        common_count: 182,
        feature_start: "2025-03-21",
        feature_end: "2025-12-12",
        target_start: "2025-03-24",
        target_end: "2025-12-15",
        original_counts: counts,
        common_counts: counts,
        dropped_target_dates: dropped,
        common_keys: commonDateKeys(182, "2025-03-24"),
      },
      shared_project_holdout: {
        terminology: "shared project holdout",
        common_count: 184,
        feature_start: "2025-12-15",
        feature_end: "2026-09-14",
        target_start: "2025-12-16",
        target_end: "2026-09-15",
        original_counts: holdoutCounts,
        common_counts: holdoutCounts,
        dropped_target_dates: dropped,
        common_keys: commonDateKeys(184, "2025-12-16"),
        previously_exposed_through: ["M-001", "M-002", "M-003", "M-004"],
      },
    },
    confusionMatrices: {
      ...context,
      validation_models: validationModels.map((row) => ({
        competitor_id: row.competitor_id,
        competitor_type: row.competitor_type,
        registered_model_id: row.registered_model_id,
        evaluation_scope: row.evaluation_scope,
        class_labels: [0, 1],
        confusion_matrix: row.confusion_matrix,
        support: row.support,
      })),
      validation_reference_baselines: validationBaselines.map((row) => ({
        competitor_id: row.competitor_id,
        competitor_type: row.competitor_type,
        registered_model_id: row.registered_model_id,
        evaluation_scope: row.evaluation_scope,
        class_labels: [0, 1],
        confusion_matrix: row.confusion_matrix,
        support: row.support,
      })),
      shared_project_holdout_models: predictionModels.map((row) => ({
        competitor_id: row.competitor_id,
        competitor_type: row.competitor_type,
        registered_model_id: row.registered_model_id,
        evaluation_scope: row.evaluation_scope,
        class_labels: [0, 1],
        confusion_matrix: row.confusion_matrix,
        support: row.support,
      })),
      shared_project_holdout_reference_baselines: predictionBaselines.map((row) => ({
        competitor_id: row.competitor_id,
        competitor_type: row.competitor_type,
        registered_model_id: row.registered_model_id,
        evaluation_scope: row.evaluation_scope,
        class_labels: [0, 1],
        confusion_matrix: row.confusion_matrix,
        support: row.support,
      })),
    },
  };
}

export function makeMockClient(
  options: {
    core?: CoreRunData;
    runList?: VerticalSliceListResponse;
  } = {},
): PaisaApiClient {
  const core = options.core ?? makeCoreRunData();
  const runList = options.runList ?? makeRunList();
  return {
    baseUrl: "http://127.0.0.1:8000",
    listRuns: vi.fn().mockResolvedValue(runList),
    getCoreRunData: vi.fn().mockResolvedValue(core),
    getAllEquityCurves: vi.fn().mockResolvedValue(equityRows),
    getAllDrawdowns: vi.fn().mockResolvedValue(drawdownRows),
    getHistoricalPrices: vi.fn(
      async (runId: string) => makeHistoricalPrices(runId),
    ),
    listUnifiedComparisons: vi.fn().mockResolvedValue(makeUnifiedComparisonList()),
    getUnifiedComparisonData: vi.fn(
      async (comparisonId: string) => makeUnifiedComparisonData(comparisonId),
    ),
    getExperiment: vi.fn(
      async (runId: string, experimentId: string) =>
        makeExperimentDetail(
          runId,
          experimentId,
          experimentId.endsWith("m-002") ? "M-002" : "M-001",
        ),
    ),
    getTrades: vi.fn(
      async (
        runId: string,
        experimentId: string,
        _limit: number,
        offset: number,
      ) => makeTradesResponse(offset, experimentId, runId),
    ),
    getSnapshots: vi.fn(
      async (runId: string, experimentId: string) =>
        makeSnapshotsResponse(experimentId, runId),
    ),
    getBenchmarks: vi.fn(
      async (runId: string, experimentId: string) =>
        makeBenchmarksResponse(experimentId, runId),
    ),
  } as unknown as PaisaApiClient;
}

export function pageResponse<T>(rows: T[]): PageResponse<T> {
  return {
    ...runContext(),
    total: rows.length,
    limit: 1000,
    offset: 0,
    rows,
  };
}
