import { z } from "zod";

import { containsProhibitedPublicArtifactText } from "./publicText";

const finite = z.number().finite();
const nullableFinite = finite.nullable();
const probability = finite.min(0).max(1);
const nullableProbability = probability.nullable();
const signedUnit = finite.min(-1).max(1);
const nullableSignedUnit = signedUnit.nullable();
const nonNegativeFinite = finite.nonnegative();
const nullableNonNegativeFinite = nonNegativeFinite.nullable();
const integer = z.number().int();
const nonNegativeInteger = integer.nonnegative();
const text = z.string().refine(
  (value) => !containsProhibitedPublicArtifactText(value),
  { message: "public text must not expose local paths or executable model artifacts" },
);
const MAX_TABLE_PAGE_SIZE = 1_000;
const MAX_PRICE_PAGE_SIZE = 5_000;
const MAX_UNPAGINATED_COLLECTION_SIZE = 1_000;
const MAX_CLASSIFICATION_COUNT = 10_000_000;
const CLASSIFICATION_METRIC_TOLERANCE = 1e-9;
const money = z.string().regex(/^-?\d+\.\d{2}$/);
const nonNegativeMoney = z.string().regex(/^\d+\.\d{2}$/);
const isoDate = z.string().regex(/^\d{4}-\d{2}-\d{2}$/).refine((value) => {
  const parsed = new Date(`${value}T00:00:00.000Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
});
const positivePrice = z
  .string()
  .regex(/^\d{1,100}\.\d{2,18}$/)
  .refine((value) => /[1-9]/.test(value), {
    message: "price must be strictly positive",
  });
const invalidPriceSources = new Set([
  "nan",
  "+nan",
  "-nan",
  "inf",
  "+inf",
  "-inf",
  "infinity",
  "+infinity",
  "-infinity",
  "none",
  "null",
]);
const priceSource = z
  .string()
  .trim()
  .min(1)
  .refine((value) => !invalidPriceSources.has(value.toLowerCase()), {
    message: "source must identify a real provider",
  });
export const ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice.";
const academicDisclaimer = z.literal(ACADEMIC_DISCLAIMER);

function requireExactRowCount<T extends z.ZodRawShape>(schema: z.ZodObject<T>) {
  return schema.refine(
    (value) => {
      const collection = value as { total: number; rows: unknown[] };
      return collection.total === collection.rows.length;
    },
    { message: "total must equal the number of rows" },
  );
}

function boundedRecord<T extends z.ZodType>(valueSchema: T) {
  return z.preprocess((value, context) => {
    if (
      value !== null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      Object.keys(value).length > MAX_UNPAGINATED_COLLECTION_SIZE
    ) {
      context.addIssue({
        code: "too_big",
        origin: "record",
        maximum: MAX_UNPAGINATED_COLLECTION_SIZE,
        inclusive: true,
        message: `record must contain at most ${MAX_UNPAGINATED_COLLECTION_SIZE} entries`,
      });
      return z.NEVER;
    }
    return value;
  }, z.record(z.string(), valueSchema));
}

export const dateRangeSchema = z.object({ start: text, end: text });
export const evaluationDateRangeSchema = z.object({
  feature_start: text,
  feature_end: text,
  target_start: text,
  target_end: text,
});

const runContextShape = {
  schema_version: text,
  run_id: text,
  source_mode: text,
  data_date_range: dateRangeSchema,
  warnings: z.array(text).max(MAX_UNPAGINATED_COLLECTION_SIZE),
  limitations: z.array(text).max(MAX_UNPAGINATED_COLLECTION_SIZE),
  disclaimer: academicDisclaimer,
};

export const runContextSchema = z.object(runContextShape);

export const verticalSliceListSchema = z.object({
  schema_version: text,
  total: nonNegativeInteger,
  limit: integer.positive().max(MAX_TABLE_PAGE_SIZE),
  offset: nonNegativeInteger,
  rows: z.array(
    z.object({
      schema_version: text,
      run_id: text,
      status: text,
      symbol: text,
      completed_at: text,
      source_mode: text,
      data_date_range: dateRangeSchema,
      model_training_run_id: text,
      model_count: nonNegativeInteger,
      experiment_count: nonNegativeInteger,
    }),
  ).max(MAX_TABLE_PAGE_SIZE),
  disclaimer: academicDisclaimer,
}).superRefine((value, context) => {
  if (value.rows.length > value.limit) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "rows must not exceed limit",
      path: ["rows"],
    });
  }
});

export const verticalSliceDetailSchema = z.object({
  ...runContextShape,
  status: text,
  symbol: text,
  created_at: text,
  evaluation_dates: evaluationDateRangeSchema,
  model_training_run_id: text,
  model_variants: z.array(text).max(MAX_UNPAGINATED_COLLECTION_SIZE),
  registered_model_ids: boundedRecord(text),
  experiment_ids: z.array(text).max(MAX_UNPAGINATED_COLLECTION_SIZE),
  shared_experiment_assumptions: z.object({
    initial_cash_pkr: money,
    buy_probability_up_gte: probability,
    sell_probability_up_lte: probability,
    maximum_allocation: finite,
    long_only: z.boolean(),
    whole_shares: z.boolean(),
    transaction_cost_bps: finite,
    slippage_bps: finite,
    signal_timing: text,
    execution: text,
    valuation: text,
    ending_position: text,
  }),
});

export const verticalSliceSummarySchema = z.object({
  ...runContextShape,
  status: text,
  symbol: text,
  completed_at: text,
  model_training_run_id: text,
  evaluation_dates: evaluationDateRangeSchema,
  best_validation_prediction_model: z.object({
    model_id: text,
    registered_model_id: text,
    algorithm: text,
    validation_balanced_accuracy: nullableFinite,
    validation_f1: finite,
    selection_basis: text,
  }),
  best_test_period_simulated_portfolio: z.object({
    model_id: text,
    portfolio_id: text,
    ending_portfolio_value: money,
    total_return: finite,
    sharpe_ratio: nullableFinite,
    selection_basis: text,
  }),
  lowest_drawdown_portfolio: z.object({
    model_id: text,
    portfolio_id: text,
    maximum_drawdown: finite,
    total_return: finite,
  }),
  buy_and_hold_result: z.object({
    portfolio_id: text,
    ending_portfolio_value: money,
    total_return: finite,
    maximum_drawdown: finite,
    fees_paid: money,
  }),
  any_model_portfolio_beat_buy_and_hold: z.boolean(),
  selection_policy: text,
  target_interpretation: text,
  portfolio_return_interpretation: text,
  training_data: z.object({
    provider: text,
    actual_feature_start: text,
    actual_feature_end: text,
    dataset_sha256: text,
  }),
});

export const modelMetricRowSchema = z.object({
  model_id: text,
  registered_model_id: text,
  algorithm: text,
  validation_accuracy: finite,
  validation_balanced_accuracy: nullableFinite,
  validation_f1: finite,
  test_accuracy: finite,
  test_balanced_accuracy: nullableFinite,
  test_f1: finite,
  train_count: nonNegativeInteger,
  validation_count: nonNegativeInteger,
  test_count: nonNegativeInteger,
  train_class_0: nonNegativeInteger,
  train_class_1: nonNegativeInteger,
  validation_class_0: nonNegativeInteger,
  validation_class_1: nonNegativeInteger,
  test_class_0: nonNegativeInteger,
  test_class_1: nonNegativeInteger,
});

export const modelsSchema = requireExactRowCount(z.object({
  ...runContextShape,
  total: nonNegativeInteger,
  rows: z.array(modelMetricRowSchema).max(MAX_UNPAGINATED_COLLECTION_SIZE),
}));

export const portfolioMetricRowSchema = z.object({
  portfolio_type: text,
  model_id: text,
  portfolio_id: text,
  ending_portfolio_value: money,
  total_return: finite,
  maximum_drawdown: finite,
  volatility: nullableFinite,
  sharpe_ratio: nullableFinite,
  number_of_trades: nonNegativeInteger,
  fees_paid: money,
  result_versus_buy_and_hold: finite,
});

export const portfoliosSchema = requireExactRowCount(z.object({
  ...runContextShape,
  total: nonNegativeInteger,
  rows: z.array(portfolioMetricRowSchema).max(MAX_UNPAGINATED_COLLECTION_SIZE),
}));

export const leaderboardSchema = requireExactRowCount(z.object({
  ...runContextShape,
  total: nonNegativeInteger,
  rows: z.array(
    z.object({
      model_id: text,
      registered_model_id: text,
      algorithm: text,
      validation_balanced_accuracy: nullableFinite,
      validation_f1: finite,
      test_accuracy: finite,
      test_balanced_accuracy: nullableFinite,
      test_f1: finite,
      portfolio_id: text,
      ending_portfolio_value: money,
      total_return: finite,
      maximum_drawdown: finite,
      volatility: nullableFinite,
      sharpe_ratio: nullableFinite,
      number_of_trades: nonNegativeInteger,
      fees_paid: money,
      result_versus_buy_and_hold: finite,
    }),
  ).max(MAX_UNPAGINATED_COLLECTION_SIZE),
}));

export const equityCurveRowSchema = z.object({
  date: text,
  series_id: text,
  portfolio_type: text,
  model_id: text.nullable(),
  portfolio_id: text,
  cash: money,
  shares: nonNegativeInteger,
  market_price: money,
  total_equity: money,
  drawdown: finite,
});

export const drawdownRowSchema = z.object({
  date: text,
  series_id: text,
  portfolio_type: text,
  model_id: text.nullable(),
  portfolio_id: text,
  drawdown: finite,
});

function pageSchema<T extends z.ZodTypeAny>(rowSchema: T) {
  return z.object({
    ...runContextShape,
    total: nonNegativeInteger,
    limit: integer.positive().max(MAX_TABLE_PAGE_SIZE),
    offset: nonNegativeInteger,
    rows: z.array(rowSchema).max(MAX_TABLE_PAGE_SIZE),
  });
}

function validatePageSize(
  value: { rows: unknown[]; limit: number },
  context: z.RefinementCtx,
): void {
  if (value.rows.length > value.limit) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "rows must not exceed limit",
      path: ["rows"],
    });
  }
}

export const equityCurvesSchema = pageSchema(equityCurveRowSchema).superRefine(validatePageSize);
export const drawdownsSchema = pageSchema(drawdownRowSchema).superRefine(validatePageSize);

export const experimentsSchema = requireExactRowCount(z.object({
  ...runContextShape,
  total: nonNegativeInteger,
  rows: z.array(
    z.object({
      experiment_id: text,
      variant_id: text,
      registered_model_id: text,
      algorithm: text,
      portfolio_id: text,
      symbol: text,
      evaluation_dates: evaluationDateRangeSchema,
      ending_equity: money,
      total_return: finite,
      maximum_drawdown: finite,
      sharpe_ratio: nullableFinite,
      number_of_trades: nonNegativeInteger,
    }),
  ).max(MAX_UNPAGINATED_COLLECTION_SIZE),
}));

const experimentConfigurationSchema = z.object({
  schema_version: text,
  experiment_id: text,
  model_id: text,
  portfolio_id: text,
  benchmark_portfolio_ids: z.object({ cash_only: text, buy_and_hold: text }),
  symbol: text,
  initial_cash: money,
  thresholds: z.object({
    buy_probability_up_gte: probability,
    sell_probability_up_lte: probability,
  }),
  maximum_allocation: finite,
  transaction_cost: z.object({ type: text, value: finite }),
  evaluation_dates: evaluationDateRangeSchema,
  execution_policy: z.object({
    signal_timing: text,
    execution_session: text,
    reference_price_field: text,
    slippage_bps: finite,
    missing_execution_price: text,
    allocation_mode: text,
    whole_share_rounding: text,
    valuation_price_field: text,
    missing_valuation_price: text,
    ending_position: text,
  }),
  metric_assumptions: z.object({
    annualization_factor: integer.positive(),
    risk_free_rate_annual: finite,
  }),
  created_at: text,
});

const benchmarkResultSchema = z.object({
  portfolio_id: text,
  ending_equity: money,
  total_return: finite,
});

const experimentResultSchema = z.object({
  schema_version: text,
  experiment_id: text,
  model_id: text,
  portfolio_id: text,
  prediction_metrics: z.object({
    accuracy: finite,
    precision: finite,
    recall: finite,
    f1: finite,
    support: nonNegativeInteger,
    positive_label: integer,
    average: text,
    zero_division: integer,
  }),
  initial_equity: money,
  ending_equity: money,
  total_return: finite,
  maximum_drawdown: finite,
  volatility: nullableFinite,
  sharpe_ratio: nullableFinite,
  number_of_trades: nonNegativeInteger,
  win_rate: nullableFinite,
  average_gain: money.nullable(),
  average_loss: money.nullable(),
  turnover: finite,
  fees_paid: money,
  time_in_market: finite,
  comparison_with_buy_and_hold: finite,
  benchmark_comparison: z.object({
    cash_only: benchmarkResultSchema,
    buy_and_hold: benchmarkResultSchema,
    excess_return_vs_cash_only: finite,
    excess_return_vs_buy_and_hold: finite,
  }),
});

export const experimentDetailSchema = z.object({
  ...runContextShape,
  variant_id: text,
  algorithm: text,
  configuration: experimentConfigurationSchema,
  result: experimentResultSchema,
  actual_class_usage: text,
}).superRefine((value, context) => {
  if (value.configuration.experiment_id !== value.result.experiment_id) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "experiment identifiers must agree",
      path: ["result", "experiment_id"],
    });
  }
});

export const tradeRowSchema = z.object({
  schema_version: text,
  trade_id: text,
  portfolio_id: text,
  signal_date: text,
  execution_date: text,
  action: text,
  quantity: nonNegativeInteger,
  reference_price: money,
  execution_price: money,
  fee: money,
  cash_after_trade: money,
  shares_after_trade: nonNegativeInteger,
});

export const snapshotRowSchema = z.object({
  schema_version: text,
  portfolio_id: text,
  date: text,
  cash: money,
  shares: nonNegativeInteger,
  market_price: money,
  position_value: money,
  total_equity: money,
  realized_profit_loss: money,
  unrealized_profit_loss: money,
});

export const tradesSchema = pageSchema(tradeRowSchema).extend({
  experiment_id: text,
  actual_class_usage: text,
}).superRefine(validatePageSize);

export const snapshotsSchema = pageSchema(snapshotRowSchema).extend({
  experiment_id: text,
}).superRefine(validatePageSize);

const benchmarkSnapshotSeriesSchema = z.object({
  portfolio_id: text,
  ending_equity: money,
  total_return: finite,
  total: nonNegativeInteger,
  limit: integer.positive().max(MAX_TABLE_PAGE_SIZE),
  offset: nonNegativeInteger,
  rows: z.array(snapshotRowSchema).max(MAX_TABLE_PAGE_SIZE),
}).superRefine((value, context) => {
  if (value.rows.length > value.limit) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "rows must not exceed limit",
      path: ["rows"],
    });
  }
});

export const benchmarksSchema = z.object({
  ...runContextShape,
  experiment_id: text,
  cash_only: benchmarkSnapshotSeriesSchema,
  buy_and_hold: benchmarkSnapshotSeriesSchema,
  buy_and_hold_trades: z.object({
    portfolio_id: text,
    total: nonNegativeInteger,
    limit: integer.positive().max(MAX_TABLE_PAGE_SIZE),
    offset: nonNegativeInteger,
    rows: z.array(tradeRowSchema).max(MAX_TABLE_PAGE_SIZE),
  }).superRefine((value, context) => {
    if (value.rows.length > value.limit) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "rows must not exceed limit",
        path: ["rows"],
      });
    }
  }),
});

export const sourceSchema = z.object({
  ...runContextShape,
  source: z.object({
    source: text,
    acquisition_mode: text,
    is_real_data: z.boolean(),
    live_fetch_succeeded: z.boolean(),
    live_acquisition_error_present: z.boolean(),
    original_provider: text,
    legacy_metadata_preserved: z.boolean(),
    producer_component: text.nullable(),
    row_count: nonNegativeInteger,
    input_prices_sha256: text,
    frozen_prices_sha256: text,
    provenance_sha256: text.nullable(),
    migration_strategy: text.nullable(),
    provenance_strength: text.nullable(),
  }),
  pipeline_evidence: z
    .object({
      schema_version: text.nullable(),
      producer: z.union([boundedRecord(z.unknown()), text]).nullable(),
      generated_at: text.nullable(),
      provider: text.nullable(),
      symbols_requested: z.array(text).max(MAX_UNPAGINATED_COLLECTION_SIZE),
      prices_artifact: boundedRecord(z.unknown()).nullable(),
    })
    .nullable(),
  migration_evidence: z
    .object({
      schema_version: text.nullable(),
      status: text.nullable(),
      producer: z.union([boundedRecord(z.unknown()), text]).nullable(),
      migration_strategy: text.nullable(),
      provenance_strength: text.nullable(),
      limitations: z.array(text).max(MAX_UNPAGINATED_COLLECTION_SIZE),
    })
    .nullable(),
});

export const historicalPricesSchema = z.object({
  schema_version: text,
  run_id: text,
  ticker: text,
  source_mode: text,
  data_date_range: z.object({ start: isoDate, end: isoDate }),
  total: nonNegativeInteger.safe(),
  limit: integer.positive().max(MAX_PRICE_PAGE_SIZE),
  offset: nonNegativeInteger.safe(),
  rows: z.array(
    z.object({
      date: isoDate,
      open: positivePrice,
      high: positivePrice.nullable(),
      low: positivePrice.nullable(),
      close: positivePrice,
      volume: nonNegativeInteger.safe(),
      source: priceSource,
    }),
  ).max(MAX_PRICE_PAGE_SIZE),
  warnings: z.array(text).max(MAX_UNPAGINATED_COLLECTION_SIZE),
  disclaimer: academicDisclaimer,
}).superRefine((value, context) => {
  if (value.data_date_range.start > value.data_date_range.end) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "data date range must be chronological",
      path: ["data_date_range"],
    });
  }
  if (
    value.rows.length > value.limit ||
    value.offset > value.total ||
    value.offset + value.rows.length > value.total
  ) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "price page metadata is inconsistent",
      path: ["rows"],
    });
  }
});

export const UNIFIED_MODEL_VARIANTS = [
  "M-001",
  "M-002",
  "M-003",
  "M-004",
  "M-005",
  "M-006",
] as const;
const UNIFIED_REFERENCE_BASELINES = [
  "majority-class-baseline",
  "previous-direction-baseline",
] as const;
const UNIFIED_BENCHMARKS = ["cash-only", "buy-and-hold"] as const;
const unifiedVariant = z.enum(UNIFIED_MODEL_VARIANTS);
const unifiedReferenceBaseline = z.enum(UNIFIED_REFERENCE_BASELINES);
const unifiedBenchmark = z.enum(UNIFIED_BENCHMARKS);
const unifiedHoldoutTerm = z.literal("shared project holdout");
const sha256 = z.string().regex(/^[a-f0-9]{64}$/);

const unifiedContextShape = {
  schema_version: text,
  comparison_id: text,
  status: z.literal("PASS"),
  symbol: z.literal("OGDC"),
  holdout_term: unifiedHoldoutTerm,
  outcomes_are_not_combined: z.literal(true),
  limitations: z.array(text).max(MAX_UNPAGINATED_COLLECTION_SIZE),
  disclaimer: academicDisclaimer,
};

function requireExactIdentifiers(
  values: readonly string[],
  expected: readonly string[],
  context: z.RefinementCtx,
  path: PropertyKey[],
): void {
  if (
    values.length !== expected.length ||
    values.some((value, index) => value !== expected[index])
  ) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: `identifiers must be exactly ${expected.join(", ")} in canonical order`,
      path,
    });
  }
}

function requireRankedIdentifiers(
  values: ReadonlyArray<{ rank: number; competitor_id: string }>,
  expected: readonly string[],
  context: z.RefinementCtx,
  path: PropertyKey[],
): void {
  const identities = values.map((row) => row.competitor_id);
  const ranks = values.map((row) => row.rank).sort((left, right) => left - right);
  if (
    identities.length !== expected.length ||
    new Set(identities).size !== expected.length ||
    identities.some((identity) => !expected.includes(identity)) ||
    ranks.some((rank, index) => rank !== index + 1)
  ) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "ranking must contain every expected competitor exactly once with contiguous ranks",
      path,
    });
  }
}

export const unifiedComparisonListSchema = z.object({
  schema_version: text,
  total: nonNegativeInteger.safe(),
  limit: integer.positive().max(MAX_TABLE_PAGE_SIZE),
  offset: nonNegativeInteger.safe(),
  default_comparison_id: text.nullable(),
  rows: z.array(z.object({
    schema_version: text,
    comparison_id: text,
    status: z.literal("PASS"),
    symbol: z.literal("OGDC"),
    model_count: z.literal(6),
    validation_count: integer.positive(),
    shared_holdout_count: integer.positive(),
    shared_holdout_target_start: isoDate,
    shared_holdout_target_end: isoDate,
    holdout_term: unifiedHoldoutTerm,
    is_default: z.boolean(),
  })).max(MAX_TABLE_PAGE_SIZE),
  disclaimer: academicDisclaimer,
}).superRefine((value, context) => {
  if (
    value.rows.length > value.limit ||
    value.offset > value.total ||
    value.offset + value.rows.length > value.total
  ) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "comparison page metadata is inconsistent",
      path: ["rows"],
    });
  }
  if (
    value.rows.some(
      (row) => row.is_default !== (row.comparison_id === value.default_comparison_id),
    )
  ) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "default comparison markers are inconsistent",
      path: ["rows"],
    });
  }
});

const unifiedModelIdentitySchema = z.object({
  variant_id: unifiedVariant,
  registered_model_id: text,
  algorithm: text,
  configuration_family: text,
});

export const unifiedComparisonDetailSchema = z.object({
  ...unifiedContextShape,
  models: z.array(unifiedModelIdentitySchema).max(UNIFIED_MODEL_VARIANTS.length),
  source_frequency: text,
  validation_count: integer.positive(),
  shared_holdout_count: integer.positive(),
  selection_policy: z.object({
    primary: text,
    tie_breakers: z.array(text).max(10),
    test_metrics_used_for_selection: z.literal(false),
  }),
  presentation: z.object({
    format: z.literal("tabular_report_only"),
    plots_included: z.literal(false),
    data_sources: z.array(text).max(20),
  }),
}).superRefine((value, context) => {
  requireExactIdentifiers(
    value.models.map((row) => row.variant_id),
    UNIFIED_MODEL_VARIANTS,
    context,
    ["models"],
  );
});

export const unifiedSourcesSchema = z.object({
  ...unifiedContextShape,
  sources: z.tuple([
    z.object({
      role: z.literal("baseline_training"),
      artifact_type: z.literal("model_training_run"),
      run_id: text.regex(/^TRN-[A-Za-z0-9][A-Za-z0-9._-]*$/),
      integrity_root_sha256: sha256,
      payload_file_count: integer.positive(),
      integrity_root_derivation: z.literal("byte_copy"),
    }),
    z.object({
      role: z.literal("baseline_vertical_slice"),
      artifact_type: z.literal("vertical_slice"),
      run_id: text.regex(/^VSL-[A-Za-z0-9][A-Za-z0-9._-]*$/),
      integrity_root_sha256: sha256,
      payload_file_count: integer.positive(),
      integrity_root_derivation: z.literal("deterministic_redacted_projection"),
    }),
    z.object({
      role: z.literal("lstm"),
      artifact_type: z.literal("lstm_price_only_experiment"),
      run_id: text.regex(/^LSTM-[A-Za-z0-9][A-Za-z0-9._-]*$/),
      integrity_root_sha256: sha256,
      payload_file_count: integer.positive(),
      integrity_root_derivation: z.literal("byte_copy"),
    }),
  ]),
  relationships: z.object({
    vertical_slice_uses_training_run: z.literal(true),
    lstm_uses_training_run: z.literal(true),
    lstm_uses_vertical_slice: z.literal(true),
  }),
});

const unifiedPredictionOutcomeSchema = z.object({
  variant_id: unifiedVariant,
  registered_model_id: text,
  criterion: text,
  scope: text,
  accuracy: probability,
  balanced_accuracy: nullableProbability,
  f1: probability,
});

const unifiedPortfolioOutcomeSchema = z.object({
  variant_id: unifiedVariant,
  registered_model_id: text,
  portfolio_id: text,
  criterion: text,
  ending_equity: money,
  total_return: finite,
  maximum_drawdown: probability,
  fees_paid: nonNegativeMoney,
});

const unifiedBenchmarkOutcomeSchema = z.object({
  initial_equity: money,
  ending_equity: money,
  total_return: finite,
  maximum_drawdown: probability,
  volatility: nullableNonNegativeFinite,
  sharpe_ratio: nullableFinite,
  number_of_trades: nonNegativeInteger,
  fees_paid: nonNegativeMoney,
});

export const unifiedSummarySchema = z.object({
  ...unifiedContextShape,
  holdout_context: z.object({
    term: unifiedHoldoutTerm,
    common_count: integer.positive(),
    target_start: isoDate,
    target_end: isoDate,
    previously_exposed_through: z.array(unifiedVariant).max(6),
  }),
  reported_outcomes: z.object({
    validation_selected_model: unifiedPredictionOutcomeSchema,
    lstm_validation_preferred_model: unifiedPredictionOutcomeSchema,
    best_shared_holdout_balanced_accuracy: unifiedPredictionOutcomeSchema,
    best_shared_holdout_f1: unifiedPredictionOutcomeSchema,
    best_simulated_model_portfolio: unifiedPortfolioOutcomeSchema,
    lowest_drawdown_model_portfolio: unifiedPortfolioOutcomeSchema,
    lowest_fee_model_portfolio: unifiedPortfolioOutcomeSchema,
    cash_only_benchmark: unifiedBenchmarkOutcomeSchema,
    buy_and_hold_result: unifiedBenchmarkOutcomeSchema,
    any_model_beat_buy_and_hold: z.object({
      value: z.boolean(),
      best_model_variant_id: unifiedVariant,
      best_model_minus_buy_and_hold: finite,
      comparison_metric: text,
    }),
  }),
});

const classificationCount = nonNegativeInteger.safe().max(MAX_CLASSIFICATION_COUNT);
const positiveClassificationCount = integer
  .positive()
  .safe()
  .max(MAX_CLASSIFICATION_COUNT);
const countPair = z.object({
  "0": classificationCount,
  "1": classificationCount,
});
const confusionMatrix = z.tuple([
  z.tuple([classificationCount, classificationCount]),
  z.tuple([classificationCount, classificationCount]),
]);
const unifiedPredictionMetricShape = {
  evaluation_scope: z.union([z.literal("validation"), unifiedHoldoutTerm]),
  row_count: positiveClassificationCount,
  support: positiveClassificationCount,
  accuracy: probability,
  balanced_accuracy: nullableProbability,
  precision: probability,
  recall: probability,
  f1: probability,
  matthews_corrcoef: nullableSignedUnit,
  roc_auc: nullableProbability,
  brier_score: probability,
  log_loss: nullableNonNegativeFinite,
  expected_calibration_error: probability,
  selection_eligible: z.boolean(),
  metric_warnings: z.array(text).max(MAX_UNPAGINATED_COLLECTION_SIZE),
  class_counts: countPair,
  predicted_class_counts: countPair,
  confusion_matrix: confusionMatrix,
};
type ReconciledPredictionMetric = {
  row_count: number;
  support: number;
  accuracy: number;
  balanced_accuracy: number | null;
  precision: number;
  recall: number;
  f1: number;
  matthews_corrcoef: number | null;
  roc_auc: number | null;
  log_loss: number | null;
  selection_eligible: boolean;
  metric_warnings: string[];
  class_counts: { "0": number; "1": number };
  predicted_class_counts: { "0": number; "1": number };
  confusion_matrix: [[number, number], [number, number]];
};

function metricsAreClose(left: number, right: number): boolean {
  return Math.abs(left - right) <= CLASSIFICATION_METRIC_TOLERANCE * Math.max(
    1,
    Math.abs(left),
    Math.abs(right),
  );
}

function reconcilePredictionMetrics(
  value: ReconciledPredictionMetric,
  context: z.RefinementCtx,
): void {
  const [[trueNegative, falsePositive], [falseNegative, truePositive]] =
    value.confusion_matrix;
  if (
    value.support !== value.row_count ||
    value.class_counts["0"] + value.class_counts["1"] !== value.row_count ||
    value.predicted_class_counts["0"] + value.predicted_class_counts["1"] !== value.row_count ||
    trueNegative + falsePositive + falseNegative + truePositive !== value.row_count ||
    value.class_counts["0"] !== trueNegative + falsePositive ||
    value.class_counts["1"] !== falseNegative + truePositive ||
    value.predicted_class_counts["0"] !== trueNegative + falseNegative ||
    value.predicted_class_counts["1"] !== falsePositive + truePositive
  ) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "prediction metric counts must reconcile",
    });
    return;
  }

  const actualNegative = trueNegative + falsePositive;
  const actualPositive = falseNegative + truePositive;
  const predictedPositive = falsePositive + truePositive;
  const f1Denominator = 2 * truePositive + falsePositive + falseNegative;
  const completeClassCoverage = actualNegative > 0 && actualPositive > 0;
  const matthewsDenominator = Math.sqrt(
    (truePositive + falsePositive) *
      (truePositive + falseNegative) *
      (trueNegative + falsePositive) *
      (trueNegative + falseNegative),
  );
  const expected: Record<
    "accuracy" | "balanced_accuracy" | "precision" | "recall" | "f1" | "matthews_corrcoef",
    number | null
  > = {
    accuracy: (trueNegative + truePositive) / value.support,
    balanced_accuracy: completeClassCoverage
      ? (trueNegative / actualNegative + truePositive / actualPositive) / 2
      : null,
    precision: predictedPositive > 0 ? truePositive / predictedPositive : 0,
    recall: actualPositive > 0 ? truePositive / actualPositive : 0,
    f1: f1Denominator > 0 ? (2 * truePositive) / f1Denominator : 0,
    matthews_corrcoef: completeClassCoverage
      ? matthewsDenominator > 0
        ? (truePositive * trueNegative - falsePositive * falseNegative) /
          matthewsDenominator
        : 0
      : null,
  };

  for (const metricName of Object.keys(expected) as Array<keyof typeof expected>) {
    const reported = value[metricName];
    const derived = expected[metricName];
    const agrees = derived === null
      ? reported === null
      : reported !== null && metricsAreClose(reported, derived);
    if (!agrees) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: `${metricName} must reconcile with the confusion matrix`,
        path: [metricName],
      });
    }
  }

  if (value.selection_eligible !== completeClassCoverage) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "selection_eligible must equal complete actual-class coverage",
      path: ["selection_eligible"],
    });
  }
  if (
    value.metric_warnings.includes("single_class_actuals") !==
    !completeClassCoverage
  ) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "single_class_actuals warning must match actual-class coverage",
      path: ["metric_warnings"],
    });
  }
  if (
    completeClassCoverage
      ? value.roc_auc === null || value.log_loss === null
      : value.roc_auc !== null || value.log_loss !== null
  ) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: completeClassCoverage
        ? "roc_auc and log_loss must be defined with both actual classes"
        : "roc_auc and log_loss must be null for single-class actuals",
      path: ["roc_auc"],
    });
  }
}
const unifiedModelPredictionMetricSchema = z.object({
  competitor_id: unifiedVariant,
  competitor_type: z.literal("model"),
  registered_model_id: text,
  algorithm: text,
  configuration_family: text,
  ...unifiedPredictionMetricShape,
}).superRefine(reconcilePredictionMetrics);
const unifiedReferencePredictionMetricSchema = z.object({
  competitor_id: unifiedReferenceBaseline,
  competitor_type: z.literal("reference_baseline"),
  registered_model_id: z.null(),
  algorithm: z.null(),
  configuration_family: z.null(),
  ...unifiedPredictionMetricShape,
}).superRefine(reconcilePredictionMetrics);

function unifiedMetricsSchema(expectedScope: "validation" | "shared project holdout") {
  return z.object({
    ...unifiedContextShape,
    models: z.array(unifiedModelPredictionMetricSchema).max(6),
    reference_baselines: z.array(unifiedReferencePredictionMetricSchema).max(2),
  }).superRefine((value, context) => {
    requireExactIdentifiers(
      value.models.map((row) => row.competitor_id),
      UNIFIED_MODEL_VARIANTS,
      context,
      ["models"],
    );
    requireExactIdentifiers(
      value.reference_baselines.map((row) => row.competitor_id),
      UNIFIED_REFERENCE_BASELINES,
      context,
      ["reference_baselines"],
    );
    if (
      [...value.models, ...value.reference_baselines].some(
        (row) => row.evaluation_scope !== expectedScope,
      )
    ) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: `all metric rows must use ${expectedScope}`,
        path: ["models"],
      });
    }
  });
}

export const unifiedPredictionMetricsSchema = unifiedMetricsSchema(
  "shared project holdout",
);
export const unifiedValidationMetricsSchema = unifiedMetricsSchema("validation")
  .and(z.object({ evaluation_scope: z.literal("validation") }));

const unifiedModelPortfolioMetricSchema = z.object({
  variant_id: unifiedVariant,
  registered_model_id: text,
  experiment_id: text,
  portfolio_id: text,
  initial_equity: money,
  ending_equity: money,
  total_return: finite,
  maximum_drawdown: probability,
  volatility: nullableNonNegativeFinite,
  sharpe_ratio: nullableFinite,
  number_of_trades: nonNegativeInteger,
  win_rate: nullableProbability,
  average_gain: money.nullable(),
  average_loss: money.nullable(),
  turnover: nonNegativeFinite,
  fees_paid: nonNegativeMoney,
  time_in_market: probability,
  excess_return_vs_cash_only: finite,
  excess_return_vs_buy_and_hold: finite,
});
const unifiedBenchmarkPortfolioMetricSchema = z.object({
  benchmark_id: unifiedBenchmark,
  initial_equity: money,
  ending_equity: money,
  total_return: finite,
  maximum_drawdown: probability,
  volatility: nullableNonNegativeFinite,
  sharpe_ratio: nullableFinite,
  number_of_trades: nonNegativeInteger,
  win_rate: nullableProbability,
  average_gain: money.nullable(),
  average_loss: money.nullable(),
  turnover: nonNegativeFinite,
  fees_paid: nonNegativeMoney,
  time_in_market: probability,
  ledger_count: z.literal(6),
});
export const unifiedPortfolioMetricsSchema = z.object({
  ...unifiedContextShape,
  models: z.array(unifiedModelPortfolioMetricSchema).max(6),
  benchmarks: z.array(unifiedBenchmarkPortfolioMetricSchema).max(2),
}).superRefine((value, context) => {
  requireExactIdentifiers(
    value.models.map((row) => row.variant_id),
    UNIFIED_MODEL_VARIANTS,
    context,
    ["models"],
  );
  requireExactIdentifiers(
    value.benchmarks.map((row) => row.benchmark_id),
    UNIFIED_BENCHMARKS,
    context,
    ["benchmarks"],
  );
  if (new Set(value.models.map((row) => row.portfolio_id)).size !== 6) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "model portfolio identifiers must be unique",
      path: ["models"],
    });
  }
});

const unifiedPredictionRankingRowSchema = z.object({
  rank: integer.positive(),
  competitor_id: unifiedVariant,
  registered_model_id: text,
  accuracy: probability,
  balanced_accuracy: nullableProbability,
  f1: probability,
});
export const unifiedPredictionRankingsSchema = z.object({
  ...unifiedContextShape,
  validation_model_selection: z.array(unifiedPredictionRankingRowSchema).max(6),
  lstm_validation_preference: z.array(unifiedPredictionRankingRowSchema).max(2),
  shared_project_holdout_balanced_accuracy: z.array(unifiedPredictionRankingRowSchema).max(6),
  shared_project_holdout_f1: z.array(unifiedPredictionRankingRowSchema).max(6),
}).superRefine((value, context) => {
  requireRankedIdentifiers(value.validation_model_selection, UNIFIED_MODEL_VARIANTS, context, ["validation_model_selection"]);
  requireRankedIdentifiers(value.lstm_validation_preference, ["M-005", "M-006"], context, ["lstm_validation_preference"]);
  requireRankedIdentifiers(value.shared_project_holdout_balanced_accuracy, UNIFIED_MODEL_VARIANTS, context, ["shared_project_holdout_balanced_accuracy"]);
  requireRankedIdentifiers(value.shared_project_holdout_f1, UNIFIED_MODEL_VARIANTS, context, ["shared_project_holdout_f1"]);
});

const unifiedPortfolioRankingRowSchema = z.object({
  rank: integer.positive(),
  competitor_id: unifiedVariant,
  registered_model_id: text,
  total_return: nullableFinite,
  maximum_drawdown: nullableProbability,
  sharpe_ratio: nullableFinite,
  fees_paid: nonNegativeMoney.nullable(),
  number_of_trades: nonNegativeInteger.nullable(),
});
export const unifiedPortfolioRankingsSchema = z.object({
  ...unifiedContextShape,
  simulated_model_portfolio_return: z.array(unifiedPortfolioRankingRowSchema).max(6),
  simulated_model_portfolio_drawdown: z.array(unifiedPortfolioRankingRowSchema).max(6),
  simulated_model_portfolio_fees: z.array(unifiedPortfolioRankingRowSchema).max(6),
}).superRefine((value, context) => {
  requireRankedIdentifiers(value.simulated_model_portfolio_return, UNIFIED_MODEL_VARIANTS, context, ["simulated_model_portfolio_return"]);
  requireRankedIdentifiers(value.simulated_model_portfolio_drawdown, UNIFIED_MODEL_VARIANTS, context, ["simulated_model_portfolio_drawdown"]);
  requireRankedIdentifiers(value.simulated_model_portfolio_fees, UNIFIED_MODEL_VARIANTS, context, ["simulated_model_portfolio_fees"]);
});

const unifiedCountRecord = z.object(Object.fromEntries(
  UNIFIED_MODEL_VARIANTS.map((variant) => [variant, nonNegativeInteger]),
) as Record<(typeof UNIFIED_MODEL_VARIANTS)[number], typeof nonNegativeInteger>);
const unifiedDroppedDatesRecord = z.object(Object.fromEntries(
  UNIFIED_MODEL_VARIANTS.map((variant) => [variant, z.array(isoDate).max(MAX_UNPAGINATED_COLLECTION_SIZE)]),
) as Record<(typeof UNIFIED_MODEL_VARIANTS)[number], z.ZodArray<typeof isoDate>>);
const unifiedCommonDateKeySchema = z.object({
  ticker: z.literal("OGDC"),
  feature_date: isoDate,
  target_date: isoDate,
  actual_class: z.union([z.literal(0), z.literal(1)]),
});
function unifiedDateScopeSchema(expectedTerm: "validation" | "shared project holdout") {
  return z.object({
    terminology: z.literal(expectedTerm),
    common_count: integer.positive(),
    feature_start: isoDate,
    feature_end: isoDate,
    target_start: isoDate,
    target_end: isoDate,
    original_counts: unifiedCountRecord,
    common_counts: unifiedCountRecord,
    dropped_target_dates: unifiedDroppedDatesRecord,
    common_keys: z.array(unifiedCommonDateKeySchema).max(MAX_UNPAGINATED_COLLECTION_SIZE),
  }).superRefine((value, context) => {
    if (
      value.common_keys.length !== value.common_count ||
      UNIFIED_MODEL_VARIANTS.some(
        (variant) => value.common_counts[variant] !== value.common_count,
      )
    ) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "common-date counts must reconcile for all six models",
      });
    }
  });
}
export const unifiedCommonDateSchema = z.object({
  ...unifiedContextShape,
  alignment_key: z.tuple([
    z.literal("ticker"),
    z.literal("feature_date"),
    z.literal("target_date"),
  ]),
  validation: unifiedDateScopeSchema("validation"),
  shared_project_holdout: unifiedDateScopeSchema("shared project holdout").and(
    z.object({
      previously_exposed_through: z.array(unifiedVariant).max(6),
    }),
  ),
});

const unifiedModelConfusionRowSchema = z.object({
  competitor_id: unifiedVariant,
  competitor_type: z.literal("model"),
  registered_model_id: text,
  evaluation_scope: z.union([z.literal("validation"), unifiedHoldoutTerm]),
  class_labels: z.tuple([z.literal(0), z.literal(1)]),
  confusion_matrix: confusionMatrix,
  support: positiveClassificationCount,
});
const unifiedReferenceConfusionRowSchema = z.object({
  competitor_id: unifiedReferenceBaseline,
  competitor_type: z.literal("reference_baseline"),
  registered_model_id: z.null(),
  evaluation_scope: z.union([z.literal("validation"), unifiedHoldoutTerm]),
  class_labels: z.tuple([z.literal(0), z.literal(1)]),
  confusion_matrix: confusionMatrix,
  support: positiveClassificationCount,
});
export const unifiedConfusionMatricesSchema = z.object({
  ...unifiedContextShape,
  validation_models: z.array(unifiedModelConfusionRowSchema).max(6),
  validation_reference_baselines: z.array(unifiedReferenceConfusionRowSchema).max(2),
  shared_project_holdout_models: z.array(unifiedModelConfusionRowSchema).max(6),
  shared_project_holdout_reference_baselines: z.array(unifiedReferenceConfusionRowSchema).max(2),
}).superRefine((value, context) => {
  requireExactIdentifiers(value.validation_models.map((row) => row.competitor_id), UNIFIED_MODEL_VARIANTS, context, ["validation_models"]);
  requireExactIdentifiers(value.validation_reference_baselines.map((row) => row.competitor_id), UNIFIED_REFERENCE_BASELINES, context, ["validation_reference_baselines"]);
  requireExactIdentifiers(value.shared_project_holdout_models.map((row) => row.competitor_id), UNIFIED_MODEL_VARIANTS, context, ["shared_project_holdout_models"]);
  requireExactIdentifiers(value.shared_project_holdout_reference_baselines.map((row) => row.competitor_id), UNIFIED_REFERENCE_BASELINES, context, ["shared_project_holdout_reference_baselines"]);
  if (
    value.validation_models.some((row) => row.evaluation_scope !== "validation") ||
    value.validation_reference_baselines.some((row) => row.evaluation_scope !== "validation") ||
    value.shared_project_holdout_models.some((row) => row.evaluation_scope !== "shared project holdout") ||
    value.shared_project_holdout_reference_baselines.some((row) => row.evaluation_scope !== "shared project holdout")
  ) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "confusion matrix scopes are inconsistent",
    });
  }
  for (const row of [
    ...value.validation_models,
    ...value.validation_reference_baselines,
    ...value.shared_project_holdout_models,
    ...value.shared_project_holdout_reference_baselines,
  ]) {
    if (row.confusion_matrix.flat().reduce((total, cell) => total + cell, 0) !== row.support) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "confusion matrix cells must sum to support",
      });
      break;
    }
  }
});
