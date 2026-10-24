import { describe, expect, it } from "vitest";
import type { ZodSafeParseResult } from "zod";

import {
  benchmarksSchema,
  drawdownsSchema,
  equityCurvesSchema,
  experimentDetailSchema,
  experimentsSchema,
  historicalPricesSchema,
  leaderboardSchema,
  modelsSchema,
  portfoliosSchema,
  runContextSchema,
  snapshotsSchema,
  sourceSchema,
  tradesSchema,
  unifiedCommonDateSchema,
  unifiedComparisonDetailSchema,
  unifiedComparisonListSchema,
  unifiedConfusionMatricesSchema,
  unifiedPortfolioMetricsSchema,
  unifiedPortfolioRankingsSchema,
  unifiedPredictionMetricsSchema,
  unifiedPredictionRankingsSchema,
  unifiedSourcesSchema,
  unifiedSummarySchema,
  unifiedValidationMetricsSchema,
  verticalSliceDetailSchema,
  verticalSliceListSchema,
  verticalSliceSummarySchema,
} from "./schemas";
import {
  drawdownRows,
  equityRows,
  makeBenchmarksResponse,
  makeCoreRunData,
  makeExperimentDetail,
  makeHistoricalPrices,
  makeUnifiedComparisonData,
  makeUnifiedComparisonList,
  makeRunList,
  makeRunListItem,
  makeSnapshotsResponse,
  makeTradesResponse,
  RUN_ID,
  runContext,
} from "../test/fixtures";

const OVERSIZED_COLLECTION_LENGTH = 10_001;

function oversizedInvalidArray(): null[] {
  return Array.from({ length: OVERSIZED_COLLECTION_LENGTH }, () => null);
}

function oversizedInvalidRecord(): Record<string, null> {
  return Object.fromEntries(
    Array.from({ length: OVERSIZED_COLLECTION_LENGTH }, (_, index) => [
      `invalid-${index}`,
      null,
    ]),
  );
}

function expectSingleCollectionIssue(
  result: ZodSafeParseResult<unknown>,
  path: PropertyKey[],
  code?: string,
): void {
  expect(result.success).toBe(false);
  if (result.success) return;

  expect(result.error.issues).toHaveLength(1);
  expect(result.error.issues[0]).toEqual(expect.objectContaining({ path }));
  if (code) {
    expect(result.error.issues[0]?.code).toBe(code);
  }
}

function makeHistoricalPricePage() {
  const history = makeHistoricalPrices();
  return { ...history, limit: 5_000, offset: 0 };
}

function makeSixModelResponses() {
  const core = makeCoreRunData();
  const modelIds = ["M-001", "M-002", "M-003", "M-004", "M-005", "M-006"];
  const models = modelIds.map((modelId, index) => ({
    ...core.models.rows[index % core.models.rows.length]!,
    model_id: modelId,
    registered_model_id: `registered-${modelId.toLowerCase()}`,
    algorithm: index < 4 ? "random_forest" : "lstm",
  }));
  const leaderboard = modelIds.map((modelId, index) => ({
    ...core.leaderboard.rows[index % core.leaderboard.rows.length]!,
    model_id: modelId,
    registered_model_id: `registered-${modelId.toLowerCase()}`,
    portfolio_id: `portfolio-${modelId.toLowerCase()}`,
    algorithm: index < 4 ? "random_forest" : "lstm",
  }));
  const modelPortfolio = core.portfolios.rows.find(
    (row) => row.portfolio_type === "model",
  )!;
  const modelPortfolios = modelIds.map((modelId) => ({
    ...modelPortfolio,
    model_id: modelId,
    portfolio_id: `portfolio-${modelId.toLowerCase()}`,
  }));
  const benchmarkPortfolios = core.portfolios.rows.filter(
    (row) => row.portfolio_type !== "model",
  );
  const baseExperiment = core.experiments.rows[0]!;
  const experiments = modelIds.map((modelId, index) => ({
    ...baseExperiment,
    experiment_id: `${RUN_ID}-${modelId.toLowerCase()}`,
    variant_id: modelId,
    registered_model_id: `registered-${modelId.toLowerCase()}`,
    portfolio_id: `portfolio-${modelId.toLowerCase()}`,
    algorithm: index < 4 ? "random_forest" : "lstm",
  }));

  return {
    detail: {
      ...core.detail,
      model_variants: modelIds,
      registered_model_ids: Object.fromEntries(
        modelIds.map((modelId) => [modelId, `registered-${modelId.toLowerCase()}`]),
      ),
      experiment_ids: experiments.map((row) => row.experiment_id),
    },
    models: { ...core.models, total: models.length, rows: models },
    leaderboard: {
      ...core.leaderboard,
      total: leaderboard.length,
      rows: leaderboard,
    },
    portfolios: {
      ...core.portfolios,
      total: modelPortfolios.length + benchmarkPortfolios.length,
      rows: [...modelPortfolios, ...benchmarkPortfolios],
    },
    experiments: {
      ...core.experiments,
      total: experiments.length,
      rows: experiments,
    },
  };
}

describe("Zod API response contracts", () => {
  it("parses every valid current core response", () => {
    const core = makeCoreRunData();

    expect(verticalSliceDetailSchema.safeParse(core.detail).success).toBe(true);
    expect(verticalSliceSummarySchema.safeParse(core.summary).success).toBe(true);
    expect(leaderboardSchema.safeParse(core.leaderboard).success).toBe(true);
    expect(modelsSchema.safeParse(core.models).success).toBe(true);
    expect(portfoliosSchema.safeParse(core.portfolios).success).toBe(true);
    expect(sourceSchema.safeParse(core.source).success).toBe(true);
    expect(experimentsSchema.safeParse(core.experiments).success).toBe(true);
    expect(experimentDetailSchema.safeParse(makeExperimentDetail()).success).toBe(true);
    expect(verticalSliceListSchema.safeParse(makeRunList()).success).toBe(true);
    expect(historicalPricesSchema.safeParse(makeHistoricalPricePage()).success).toBe(true);
    expect(
      equityCurvesSchema.safeParse({
        ...runContext(),
        total: equityRows.length,
        limit: 1_000,
        offset: 0,
        rows: equityRows,
      }).success,
    ).toBe(true);
    expect(
      drawdownsSchema.safeParse({
        ...runContext(),
        total: drawdownRows.length,
        limit: 1_000,
        offset: 0,
        rows: drawdownRows,
      }).success,
    ).toBe(true);
    expect(tradesSchema.safeParse(makeTradesResponse()).success).toBe(true);
    expect(snapshotsSchema.safeParse(makeSnapshotsResponse()).success).toBe(true);
    expect(benchmarksSchema.safeParse(makeBenchmarksResponse()).success).toBe(true);
  });

  it("keeps M-001 through M-006 dashboard comparison collections valid", () => {
    const responses = makeSixModelResponses();

    expect(verticalSliceDetailSchema.safeParse(responses.detail).success).toBe(true);
    expect(modelsSchema.safeParse(responses.models).success).toBe(true);
    expect(leaderboardSchema.safeParse(responses.leaderboard).success).toBe(true);
    expect(portfoliosSchema.safeParse(responses.portfolios).success).toBe(true);
    expect(experimentsSchema.safeParse(responses.experiments).success).toBe(true);
  });

  it("rejects malformed leaderboard and six-model rows", () => {
    const core = makeCoreRunData();
    const malformedLeaderboard = {
      ...core.leaderboard,
      rows: core.leaderboard.rows.map((row, index) =>
        index === 0 ? { ...row, ending_portfolio_value: "100000" } : row,
      ),
    };
    const sixModels = makeSixModelResponses();
    const malformedSixModels = {
      ...sixModels.models,
      rows: sixModels.models.rows.map((row) =>
        row.model_id === "M-006" ? { ...row, test_f1: "invalid" } : row,
      ),
    };

    expect(leaderboardSchema.safeParse(malformedLeaderboard).success).toBe(false);
    expect(modelsSchema.safeParse(malformedSixModels).success).toBe(false);
  });

  it.each([-0.01, 1.01, Number.NaN, Number.POSITIVE_INFINITY])(
    "rejects invalid probability thresholds (%s)",
    (invalidProbability) => {
      const detail = makeCoreRunData().detail;
      const invalid = {
        ...detail,
        shared_experiment_assumptions: {
          ...detail.shared_experiment_assumptions,
          buy_probability_up_gte: invalidProbability,
        },
      };
      const experiment = makeExperimentDetail();
      const invalidExperiment = {
        ...experiment,
        configuration: {
          ...experiment.configuration,
          thresholds: {
            ...experiment.configuration.thresholds,
            sell_probability_up_lte: invalidProbability,
          },
        },
      };

      expect(verticalSliceDetailSchema.safeParse(invalid).success).toBe(false);
      expect(experimentDetailSchema.safeParse(invalidExperiment).success).toBe(false);
    },
  );

  it("rejects NaN, Infinity, and unexpected null while retaining declared nullability", () => {
    const models = makeCoreRunData().models;
    const withNaN = {
      ...models,
      rows: models.rows.map((row, index) =>
        index === 0 ? { ...row, validation_accuracy: Number.NaN } : row,
      ),
    };
    const withInfinity = {
      ...models,
      rows: models.rows.map((row, index) =>
        index === 0 ? { ...row, test_f1: Number.POSITIVE_INFINITY } : row,
      ),
    };
    const withUnexpectedNull = {
      ...models,
      rows: models.rows.map((row, index) =>
        index === 0 ? { ...row, test_f1: null } : row,
      ),
    };
    const withAllowedNull = {
      ...models,
      rows: models.rows.map((row, index) =>
        index === 0 ? { ...row, test_balanced_accuracy: null } : row,
      ),
    };

    expect(modelsSchema.safeParse(withNaN).success).toBe(false);
    expect(modelsSchema.safeParse(withInfinity).success).toBe(false);
    expect(modelsSchema.safeParse(withUnexpectedNull).success).toBe(false);
    expect(modelsSchema.safeParse(withAllowedNull).success).toBe(true);
  });

  it("enforces documented table and historical-price page limits", () => {
    const run = makeRunListItem();
    const tooManyRuns = {
      ...makeRunList(),
      total: 1_001,
      rows: Array.from({ length: 1_001 }, () => ({ ...run })),
    };
    const history = makeHistoricalPricePage();
    const tooManyPrices = {
      ...history,
      total: 5_001,
      rows: Array.from({ length: 5_001 }, () => ({ ...history.rows[0]! })),
    };

    expect(verticalSliceListSchema.safeParse(tooManyRuns).success).toBe(false);
    expect(historicalPricesSchema.safeParse(tooManyPrices).success).toBe(false);
    expect(
      verticalSliceListSchema.safeParse({ ...makeRunList(), limit: 1_001 }).success,
    ).toBe(false);
  });

  it("fails safely on a bounded oversized invalid collection", () => {
    const oversized = {
      ...makeRunList(),
      total: 10_001,
      rows: Array.from({ length: 10_001 }, () => null),
    };
    const result = verticalSliceListSchema.safeParse(oversized);

    expect(result.success).toBe(false);
    if (result.success) return;
    expect(result.error.issues).toEqual([
      expect.objectContaining({ code: "too_big", path: ["rows"] }),
    ]);
  });

  it("bounds every shared-context and detail collection before item validation", () => {
    const context = runContext();
    const detail = makeCoreRunData().detail;
    const oversized = oversizedInvalidArray();

    for (const field of ["warnings", "limitations"] as const) {
      expectSingleCollectionIssue(
        runContextSchema.safeParse({ ...context, [field]: oversized }),
        [field],
        "too_big",
      );
    }

    for (const field of ["model_variants", "experiment_ids"] as const) {
      expectSingleCollectionIssue(
        verticalSliceDetailSchema.safeParse({ ...detail, [field]: oversized }),
        [field],
        "too_big",
      );
    }
  });

  it("rejects an oversized registered-model record before validating its values", () => {
    const detail = makeCoreRunData().detail;
    const result = verticalSliceDetailSchema.safeParse({
      ...detail,
      registered_model_ids: oversizedInvalidRecord(),
    });

    expectSingleCollectionIssue(result, ["registered_model_ids"], "too_big");
  });

  it("bounds every unpaginated result table before row validation", () => {
    const core = makeCoreRunData();
    const rows = oversizedInvalidArray();
    const cases = [
      { schema: modelsSchema, response: core.models },
      { schema: portfoliosSchema, response: core.portfolios },
      { schema: leaderboardSchema, response: core.leaderboard },
      { schema: experimentsSchema, response: core.experiments },
    ];

    for (const { schema, response } of cases) {
      const result = schema.safeParse({
        ...response,
        total: OVERSIZED_COLLECTION_LENGTH,
        rows,
      });
      expectSingleCollectionIssue(result, ["rows"], "too_big");
    }
  });

  it("bounds source-evidence arrays and records before nested validation", () => {
    const source = makeCoreRunData().source;
    const oversizedArray = oversizedInvalidArray();
    const oversizedRecord = oversizedInvalidRecord();
    const cases = [
      {
        response: {
          ...source,
          pipeline_evidence: {
            ...source.pipeline_evidence!,
            symbols_requested: oversizedArray,
          },
        },
        path: ["pipeline_evidence", "symbols_requested"],
        code: "too_big",
      },
      {
        response: {
          ...source,
          pipeline_evidence: {
            ...source.pipeline_evidence!,
            producer: oversizedRecord,
          },
        },
        path: ["pipeline_evidence", "producer"],
      },
      {
        response: {
          ...source,
          pipeline_evidence: {
            ...source.pipeline_evidence!,
            prices_artifact: oversizedRecord,
          },
        },
        path: ["pipeline_evidence", "prices_artifact"],
        code: "too_big",
      },
      {
        response: {
          ...source,
          migration_evidence: {
            ...source.migration_evidence!,
            producer: oversizedRecord,
          },
        },
        path: ["migration_evidence", "producer"],
      },
      {
        response: {
          ...source,
          migration_evidence: {
            ...source.migration_evidence!,
            limitations: oversizedArray,
          },
        },
        path: ["migration_evidence", "limitations"],
        code: "too_big",
      },
    ];

    for (const { response, path, code } of cases) {
      expectSingleCollectionIssue(sourceSchema.safeParse(response), path, code);
    }
  });

  it("bounds historical-price warnings independently from the price-page limit", () => {
    const result = historicalPricesSchema.safeParse({
      ...makeHistoricalPricePage(),
      warnings: oversizedInvalidArray(),
    });

    expectSingleCollectionIssue(result, ["warnings"], "too_big");
  });

  it("strips unknown object fields while preserving intentionally open evidence records", () => {
    const models = makeCoreRunData().models;
    const parsedModels = modelsSchema.parse({
      ...models,
      unexpected_security_field: "not part of the response contract",
      rows: models.rows.map((row) => ({ ...row, unexpected_row_field: "strip me" })),
    });
    const source = makeCoreRunData().source;
    const parsedSource = sourceSchema.parse({
      ...source,
      pipeline_evidence: {
        ...source.pipeline_evidence!,
        prices_artifact: { relative_reference: "frozen_input/OGDC_prices.csv" },
      },
    });

    expect(parsedModels).not.toHaveProperty("unexpected_security_field");
    expect(parsedModels.rows[0]).not.toHaveProperty("unexpected_row_field");
    expect(parsedSource.pipeline_evidence?.prices_artifact).toEqual({
      relative_reference: "frozen_input/OGDC_prices.csv",
    });
  });

  it("parses every valid unified-comparison response contract", () => {
    const data = makeUnifiedComparisonData();

    expect(unifiedComparisonListSchema.safeParse(makeUnifiedComparisonList()).success).toBe(true);
    expect(unifiedComparisonDetailSchema.safeParse(data.detail).success).toBe(true);
    expect(unifiedSourcesSchema.safeParse(data.sources).success).toBe(true);
    expect(unifiedSummarySchema.safeParse(data.summary).success).toBe(true);
    expect(unifiedPredictionMetricsSchema.safeParse(data.predictionMetrics).success).toBe(true);
    expect(unifiedValidationMetricsSchema.safeParse(data.validationMetrics).success).toBe(true);
    expect(unifiedPortfolioMetricsSchema.safeParse(data.portfolioMetrics).success).toBe(true);
    expect(unifiedPredictionRankingsSchema.safeParse(data.predictionRankings).success).toBe(true);
    expect(unifiedPortfolioRankingsSchema.safeParse(data.portfolioRankings).success).toBe(true);
    expect(unifiedCommonDateSchema.safeParse(data.commonDateContract).success).toBe(true);
    expect(unifiedConfusionMatricesSchema.safeParse(data.confusionMatrices).success).toBe(true);
  });

  it("requires M-001 through M-006 exactly once in every six-model collection", () => {
    const data = makeUnifiedComparisonData();
    const duplicateMetrics = structuredClone(data.predictionMetrics);
    duplicateMetrics.models[5] = { ...duplicateMetrics.models[0]! };
    const missingPortfolio = structuredClone(data.portfolioMetrics);
    missingPortfolio.models.pop();
    const duplicateConfusion = structuredClone(data.confusionMatrices);
    duplicateConfusion.shared_project_holdout_models[5] = {
      ...duplicateConfusion.shared_project_holdout_models[0]!,
    };

    expect(unifiedPredictionMetricsSchema.safeParse(duplicateMetrics).success).toBe(false);
    expect(unifiedPortfolioMetricsSchema.safeParse(missingPortfolio).success).toBe(false);
    expect(unifiedConfusionMatricesSchema.safeParse(duplicateConfusion).success).toBe(false);
  });

  it("requires the shared project holdout term and separate-outcome contract", () => {
    const data = makeUnifiedComparisonData();

    expect(
      unifiedComparisonDetailSchema.safeParse({
        ...data.detail,
        holdout_term: "test set",
      }).success,
    ).toBe(false);
    expect(
      unifiedSummarySchema.safeParse({
        ...data.summary,
        outcomes_are_not_combined: false,
      }).success,
    ).toBe(false);
    expect(
      unifiedSummarySchema.parse({
        ...data.summary,
        universal_winner: "M-006",
      }),
    ).not.toHaveProperty("universal_winner");
  });

  it("requires canonical source roles, artifact types, run-ID families, and relationships", () => {
    const data = makeUnifiedComparisonData();
    const duplicateRole = structuredClone(data.sources);
    Object.assign(duplicateRole.sources[1], {
      role: "baseline_training",
      artifact_type: "model_training_run",
    });
    const wrongRunFamily = structuredClone(data.sources);
    Object.assign(wrongRunFamily.sources[2], { run_id: "TRN-wrong-family" });
    const falseRelationship = structuredClone(data.sources);
    Object.assign(falseRelationship.relationships, {
      lstm_uses_vertical_slice: false,
    });

    expect(unifiedSourcesSchema.safeParse(duplicateRole).success).toBe(false);
    expect(unifiedSourcesSchema.safeParse(wrongRunFamily).success).toBe(false);
    expect(unifiedSourcesSchema.safeParse(falseRelationship).success).toBe(false);
  });

  it("accepts declared null metrics and rejects NaN or Infinity", () => {
    const data = makeUnifiedComparisonData();
    expect(unifiedPredictionMetricsSchema.safeParse(data.predictionMetrics).success).toBe(true);
    expect(unifiedPortfolioMetricsSchema.safeParse(data.portfolioMetrics).success).toBe(true);

    const singleClass = structuredClone(data.predictionMetrics);
    Object.assign(singleClass.models[2]!, {
      accuracy: 1,
      balanced_accuracy: null,
      precision: 0,
      recall: 0,
      f1: 0,
      matthews_corrcoef: null,
      roc_auc: null,
      log_loss: null,
      selection_eligible: false,
      metric_warnings: ["single_class_actuals"],
      class_counts: { "0": 184, "1": 0 },
      predicted_class_counts: { "0": 184, "1": 0 },
      confusion_matrix: [[184, 0], [0, 0]],
    });
    expect(singleClass.models[2]?.balanced_accuracy).toBeNull();
    expect(unifiedPredictionMetricsSchema.safeParse(singleClass).success).toBe(true);

    const nanMetrics = structuredClone(data.predictionMetrics);
    nanMetrics.models[0]!.accuracy = Number.NaN;
    const infinitePortfolio = structuredClone(data.portfolioMetrics);
    infinitePortfolio.models[0]!.total_return = Number.POSITIVE_INFINITY;
    expect(unifiedPredictionMetricsSchema.safeParse(nanMetrics).success).toBe(false);
    expect(unifiedPortfolioMetricsSchema.safeParse(infinitePortfolio).success).toBe(false);
  });

  it("enforces metric domains and accepts their inclusive boundaries", () => {
    const data = makeUnifiedComparisonData();
    const outOfRangeCases: Array<[unknown, { safeParse: (value: unknown) => { success: boolean } }]> = [];

    const accuracy = structuredClone(data.predictionMetrics);
    accuracy.models[0]!.accuracy = 1.5;
    outOfRangeCases.push([accuracy, unifiedPredictionMetricsSchema]);

    const f1 = structuredClone(data.summary);
    f1.reported_outcomes.best_shared_holdout_f1.f1 = 99;
    outOfRangeCases.push([f1, unifiedSummarySchema]);

    const mcc = structuredClone(data.validationMetrics);
    mcc.models[0]!.matthews_corrcoef = -1.1;
    outOfRangeCases.push([mcc, unifiedValidationMetricsSchema]);

    const loss = structuredClone(data.predictionMetrics);
    loss.models[0]!.log_loss = -0.01;
    outOfRangeCases.push([loss, unifiedPredictionMetricsSchema]);

    const drawdown = structuredClone(data.portfolioMetrics);
    drawdown.models[0]!.maximum_drawdown = -0.2;
    outOfRangeCases.push([drawdown, unifiedPortfolioMetricsSchema]);

    const oversizedDrawdown = structuredClone(data.portfolioRankings);
    oversizedDrawdown.simulated_model_portfolio_drawdown[0]!.maximum_drawdown = 1.01;
    outOfRangeCases.push([oversizedDrawdown, unifiedPortfolioRankingsSchema]);

    const volatility = structuredClone(data.portfolioMetrics);
    volatility.benchmarks[1]!.volatility = -0.1;
    outOfRangeCases.push([volatility, unifiedPortfolioMetricsSchema]);

    const turnover = structuredClone(data.portfolioMetrics);
    turnover.models[0]!.turnover = -0.1;
    outOfRangeCases.push([turnover, unifiedPortfolioMetricsSchema]);

    const winRate = structuredClone(data.portfolioMetrics);
    winRate.models[0]!.win_rate = 5;
    outOfRangeCases.push([winRate, unifiedPortfolioMetricsSchema]);

    const timeInMarket = structuredClone(data.portfolioMetrics);
    timeInMarket.models[0]!.time_in_market = 5;
    outOfRangeCases.push([timeInMarket, unifiedPortfolioMetricsSchema]);

    const ranking = structuredClone(data.predictionRankings);
    ranking.validation_model_selection[0]!.balanced_accuracy = -0.1;
    outOfRangeCases.push([ranking, unifiedPredictionRankingsSchema]);

    for (const [payload, schema] of outOfRangeCases) {
      expect(schema.safeParse(payload).success).toBe(false);
    }

    const lowerBoundaries = structuredClone(data.predictionMetrics);
    Object.assign(lowerBoundaries.models[0]!, {
      accuracy: 0,
      balanced_accuracy: 0,
      precision: 0,
      recall: 0,
      f1: 0,
      matthews_corrcoef: -1,
      roc_auc: 0,
      brier_score: 0,
      log_loss: 0,
      expected_calibration_error: 0,
      predicted_class_counts: { "0": 91, "1": 93 },
      confusion_matrix: [[0, 93], [91, 0]],
    });
    expect(unifiedPredictionMetricsSchema.safeParse(lowerBoundaries).success).toBe(true);

    const upperBoundaries = structuredClone(data.predictionMetrics);
    Object.assign(upperBoundaries.models[0]!, {
      accuracy: 1,
      balanced_accuracy: 1,
      precision: 1,
      recall: 1,
      f1: 1,
      matthews_corrcoef: 1,
      roc_auc: 1,
      brier_score: 1,
      log_loss: 0,
      expected_calibration_error: 1,
      predicted_class_counts: { "0": 93, "1": 91 },
      confusion_matrix: [[93, 0], [0, 91]],
    });
    expect(unifiedPredictionMetricsSchema.safeParse(upperBoundaries).success).toBe(true);

    const portfolioBoundaries = structuredClone(data.portfolioMetrics);
    Object.assign(portfolioBoundaries.models[0]!, {
      maximum_drawdown: 1,
      volatility: 0,
      turnover: 0,
      win_rate: 1,
      time_in_market: 0,
    });
    expect(unifiedPortfolioMetricsSchema.safeParse(portfolioBoundaries).success).toBe(true);
  });

  it("reconciles class and predicted-class counts with confusion-matrix axes", () => {
    const data = makeUnifiedComparisonData();
    const badModelCounts = structuredClone(data.predictionMetrics);
    badModelCounts.models[0]!.class_counts = {
      "0": (badModelCounts.models[0]!.class_counts["0"] ?? 0) - 1,
      "1": (badModelCounts.models[0]!.class_counts["1"] ?? 0) + 1,
    };
    const badReferenceCounts = structuredClone(data.predictionMetrics);
    badReferenceCounts.reference_baselines[0]!.predicted_class_counts = {
      "0": (badReferenceCounts.reference_baselines[0]!.predicted_class_counts["0"] ?? 0) + 1,
      "1": (badReferenceCounts.reference_baselines[0]!.predicted_class_counts["1"] ?? 0) - 1,
    };

    expect(unifiedPredictionMetricsSchema.safeParse(badModelCounts).success).toBe(false);
    expect(unifiedPredictionMetricsSchema.safeParse(badReferenceCounts).success).toBe(false);
  });

  it("rejects within-range metrics that contradict model and reference matrices", () => {
    const data = makeUnifiedComparisonData();
    const badModelMetric = structuredClone(data.predictionMetrics);
    badModelMetric.models[0]!.accuracy = 0.25;
    const badReferenceMetric = structuredClone(data.predictionMetrics);
    badReferenceMetric.reference_baselines[0]!.f1 = 0.25;

    expect(unifiedPredictionMetricsSchema.safeParse(badModelMetric).success).toBe(false);
    expect(unifiedPredictionMetricsSchema.safeParse(badReferenceMetric).success).toBe(false);

    const serializerNoise = structuredClone(data.predictionMetrics);
    serializerNoise.models[0]!.accuracy += 5e-10;
    expect(unifiedPredictionMetricsSchema.safeParse(serializerNoise).success).toBe(true);
  });

  it("enforces producer class-coverage semantics and safe classification counts", () => {
    const data = makeUnifiedComparisonData();
    const completeButIneligible = structuredClone(data.predictionMetrics);
    completeButIneligible.models[0]!.selection_eligible = false;
    const completeButWarned = structuredClone(data.predictionMetrics);
    completeButWarned.models[0]!.metric_warnings = ["single_class_actuals"];
    const completeWithoutAuc = structuredClone(data.predictionMetrics);
    completeWithoutAuc.models[0]!.roc_auc = null;
    const completeWithoutLogLoss = structuredClone(data.predictionMetrics);
    completeWithoutLogLoss.models[0]!.log_loss = null;
    const undefinedBrier = structuredClone(data.predictionMetrics);
    Object.assign(undefinedBrier.models[0]!, { brier_score: null });
    const undefinedCalibrationError = structuredClone(data.predictionMetrics);
    Object.assign(undefinedCalibrationError.models[0]!, {
      expected_calibration_error: null,
    });
    const oversized = structuredClone(data.predictionMetrics);
    Object.assign(oversized.models[0]!, {
      row_count: 10_000_001,
      support: 10_000_001,
      accuracy: 1,
      balanced_accuracy: 1,
      precision: 1,
      recall: 1,
      f1: 1,
      matthews_corrcoef: 1,
      class_counts: { "0": 10_000_000, "1": 1 },
      predicted_class_counts: { "0": 10_000_000, "1": 1 },
      confusion_matrix: [[10_000_000, 0], [0, 1]],
    });

    for (const payload of [
      completeButIneligible,
      completeButWarned,
      completeWithoutAuc,
      completeWithoutLogLoss,
      undefinedBrier,
      undefinedCalibrationError,
      oversized,
    ]) {
      expect(unifiedPredictionMetricsSchema.safeParse(payload).success).toBe(false);
    }
  });

  it("strips unknown fields outside the unified API contract", () => {
    const detail = makeUnifiedComparisonData().detail;
    const parsed = unifiedComparisonDetailSchema.parse({
      ...detail,
      model_binary: "models/private.joblib",
      local_path: "/Users/private-owner/artifacts",
      models: detail.models.map((row) => ({
        ...row,
        artifact_path: `models/${row.variant_id}.joblib`,
      })),
    });

    expect(parsed).not.toHaveProperty("model_binary");
    expect(parsed).not.toHaveProperty("local_path");
    expect(parsed.models.every((row) => !("artifact_path" in row))).toBe(true);
  });

  it.each([
    "models/private.joblib",
    "models/private.joblib/metadata",
    "models\\private.joblib\\metadata",
    "models/private.joblib.",
    "models/private.pb",
    "//server/share/model.keras",
    "note,/Users/private/report.csv",
    "note;//server/share/report.csv",
    "source:file:C:\\private\\report.csv",
    "See https://example.test/report?source=/Users/private/report.csv",
    "See https://example.test/report?source=%2FUsers%2Fprivate%2Freport.csv",
    "See https://example.test/models/private%2Ejoblib",
  ])("rejects prohibited text in a schema-approved public field: %s", (unsafeText) => {
    const summary = makeUnifiedComparisonData().summary;
    summary.limitations = [unsafeText];

    expect(unifiedSummarySchema.safeParse(summary).success).toBe(false);
  });

  it("allows ordinary profile text in a schema-approved public field", () => {
    const summary = makeUnifiedComparisonData().summary;
    summary.limitations = [
      "profile:academic_research",
      "See https://models.example.test/usr/local/share/model-card.html",
      "See wss://models.example.test/usr/local/share/model-card.html",
      "See ftp://models.example.test/usr/local/share/model-card.html",
    ];

    expect(unifiedSummarySchema.safeParse(summary).success).toBe(true);
  });
});
