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
  VerticalSliceListItem,
  VerticalSliceListResponse,
} from "../api/types";

export const RUN_ID = "VSL-TEST-OGDC-01";
export const OTHER_RUN_ID = "VSL-TEST-OGDC-02";
export const EXPERIMENT_ONE = `${RUN_ID}-m-001`;
export const EXPERIMENT_TWO = `${RUN_ID}-m-002`;
export const DISCLAIMER = "Academic simulation; not financial advice.";

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
