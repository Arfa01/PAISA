import type { ZodType } from "zod";

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
import type {
  BenchmarksResponse,
  CoreRunData,
  DrawdownRow,
  DrawdownsResponse,
  EquityCurveRow,
  EquityCurvesResponse,
  ExperimentDetailResponse,
  ExperimentsResponse,
  HistoricalPricesPageResponse,
  HistoricalPricesResponse,
  LeaderboardResponse,
  ModelsResponse,
  PortfoliosResponse,
  SnapshotsResponse,
  SourceResponse,
  TradesResponse,
  UnifiedCommonDateResponse,
  UnifiedComparisonData,
  UnifiedComparisonDetailResponse,
  UnifiedComparisonListItem,
  UnifiedComparisonListResponse,
  UnifiedComparisonSummaryResponse,
  UnifiedConfusionMatricesResponse,
  UnifiedPortfolioMetricsResponse,
  UnifiedPortfolioRankingsResponse,
  UnifiedPredictionMetricsResponse,
  UnifiedPredictionRankingsResponse,
  UnifiedSourcesResponse,
  UnifiedValidationMetricsResponse,
  VerticalSliceDetailResponse,
  VerticalSliceListItem,
  VerticalSliceListResponse,
  VerticalSliceSummaryResponse,
} from "./types";

export type ApiErrorKind =
  | "http"
  | "network"
  | "timeout"
  | "cancelled"
  | "malformed";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly kind: ApiErrorKind,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const NETWORK_URL_PATTERN = /\b(?:https?|wss?|ftp):\/\/[^\s"'<>]+/gi;
const LOCAL_PATH_PATTERNS = [
  /file:(?:\/\/)?[^\s"'<>]*/i,
  /(?:^|[\s(`'"=:>])\/(?!\/)[^\s"'<>),;]+/,
  /(?:^|[\s(`'"=:>])[A-Za-z]:[\\/][^\s"'<>)]*/,
  /(?:^|[\s(`'"=:>])\\\\[^\\\s"'<>]+\\[^\s"'<>)]*/,
];

function containsAbsoluteLocalPath(value: unknown): boolean {
  if (typeof value === "string") {
    const withoutWebUrls = value.replace(NETWORK_URL_PATTERN, "");
    return LOCAL_PATH_PATTERNS.some((pattern) => pattern.test(withoutWebUrls));
  }
  if (Array.isArray(value)) return value.some(containsAbsoluteLocalPath);
  if (value && typeof value === "object") {
    return Object.entries(value).some(
      ([key, nested]) => containsAbsoluteLocalPath(key) || containsAbsoluteLocalPath(nested),
    );
  }
  return false;
}

function safeHttpMessage(status: number): string {
  if (status === 404) return "The requested PAISA result was not found.";
  if (status === 409) return "The PAISA result is not ready or failed integrity checks.";
  if (status === 422) return "The PAISA API rejected the request parameters.";
  return `The PAISA API returned HTTP ${status}.`;
}

interface PageEnvelope<T> {
  total: number;
  limit: number;
  offset: number;
  rows: T[];
  run_id?: string;
}

const HISTORICAL_PRICE_PAGE_SIZE = 5_000;

function assertRunIdentity(response: { run_id: string }, expectedRunId: string): void {
  if (response.run_id !== expectedRunId) {
    throw new ApiError("The PAISA API returned inconsistent run identity.", "malformed");
  }
}

function assertExperimentIdentity(actualExperimentId: string, expectedExperimentId: string): void {
  if (actualExperimentId !== expectedExperimentId) {
    throw new ApiError(
      "The PAISA API returned inconsistent experiment identity.",
      "malformed",
    );
  }
}

function assertComparisonIdentity(
  response: { comparison_id: string },
  expectedComparisonId: string,
): void {
  if (response.comparison_id !== expectedComparisonId) {
    throw new ApiError(
      "The PAISA API returned inconsistent comparison identity.",
      "malformed",
    );
  }
}

function unifiedContextFingerprint(response: {
  schema_version: string;
  comparison_id: string;
  status: string;
  symbol: string;
  holdout_term: string;
  outcomes_are_not_combined: boolean;
  limitations: string[];
  disclaimer: string;
}): string {
  return JSON.stringify({
    schema_version: response.schema_version,
    comparison_id: response.comparison_id,
    status: response.status,
    symbol: response.symbol,
    holdout_term: response.holdout_term,
    outcomes_are_not_combined: response.outcomes_are_not_combined,
    limitations: response.limitations,
    disclaimer: response.disclaimer,
  });
}

function assertRequestedPage(
  response: PageEnvelope<unknown>,
  expectedLimit: number,
  expectedOffset: number,
): void {
  if (
    response.limit !== expectedLimit ||
    response.offset !== expectedOffset ||
    response.rows.length > response.limit ||
    response.offset + response.rows.length > response.total
  ) {
    throw new ApiError("The PAISA API returned inconsistent pagination metadata.", "malformed");
  }
}

function normalizeBaseUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, "");
  if (!/^https?:\/\//i.test(trimmed)) {
    throw new Error("VITE_PAISA_API_BASE_URL must be an HTTP(S) URL");
  }
  return trimmed;
}

export function configuredApiBaseUrl(): string {
  return normalizeBaseUrl(
    import.meta.env.VITE_PAISA_API_BASE_URL ?? "http://127.0.0.1:8000",
  );
}

export class PaisaApiClient {
  readonly baseUrl: string;

  constructor(
    baseUrl = configuredApiBaseUrl(),
    private readonly timeoutMs = 12_000,
  ) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
  }

  private async request<T>(
    path: string,
    schema: ZodType<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    const controller = new AbortController();
    let timedOut = false;
    const abortFromCaller = () => controller.abort();
    if (signal?.aborted) {
      controller.abort();
    } else {
      signal?.addEventListener("abort", abortFromCaller, { once: true });
    }
    const timer = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeoutMs);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        headers: { Accept: "application/json" },
        signal: controller.signal,
      });
      let payload: unknown;
      try {
        payload = await response.json();
      } catch {
        throw new ApiError(
          "The PAISA API returned a non-JSON response.",
          "malformed",
          response.status,
        );
      }
      if (!response.ok) {
        throw new ApiError(safeHttpMessage(response.status), "http", response.status);
      }
      if (containsAbsoluteLocalPath(payload)) {
        throw new ApiError(
          "The PAISA API response contained prohibited local-path data.",
          "malformed",
          response.status,
        );
      }
      const parsed = schema.safeParse(payload);
      if (!parsed.success) {
        throw new ApiError(
          "The PAISA API returned data in an unexpected format.",
          "malformed",
          response.status,
        );
      }
      return parsed.data;
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (controller.signal.aborted) {
        if (timedOut) {
          throw new ApiError("The PAISA API request timed out.", "timeout");
        }
        throw new ApiError("The PAISA API request was cancelled.", "cancelled");
      }
      throw new ApiError(
        "Could not reach the PAISA API. Check that the backend is running.",
        "network",
      );
    } finally {
      window.clearTimeout(timer);
      signal?.removeEventListener("abort", abortFromCaller);
    }
  }

  private async requestForRun<T extends { run_id: string }>(
    path: string,
    schema: ZodType<T>,
    runId: string,
    signal?: AbortSignal,
  ): Promise<T> {
    const response = await this.request(path, schema, signal);
    assertRunIdentity(response, runId);
    return response;
  }

  private async requestForComparison<T extends { comparison_id: string }>(
    path: string,
    schema: ZodType<T>,
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<T> {
    const response = await this.request(path, schema, signal);
    assertComparisonIdentity(response, comparisonId);
    return response;
  }

  async listUnifiedComparisons(
    signal?: AbortSignal,
  ): Promise<UnifiedComparisonListResponse> {
    let firstPage: UnifiedComparisonListResponse | undefined;
    let listFingerprint: string | undefined;
    const { rows, total } = await this.collectPages<UnifiedComparisonListItem>(
      async (limit, offset) => {
        const page = await this.request(
          `/api/v1/unified-comparisons?limit=${limit}&offset=${offset}`,
          unifiedComparisonListSchema,
          signal,
        );
        const fingerprint = JSON.stringify({
          schema_version: page.schema_version,
          default_comparison_id: page.default_comparison_id,
          disclaimer: page.disclaimer,
        });
        if (listFingerprint === undefined) {
          listFingerprint = fingerprint;
          firstPage = page;
        } else if (listFingerprint !== fingerprint) {
          throw new ApiError(
            "The PAISA API returned inconsistent comparison-list context.",
            "malformed",
          );
        }
        return page;
      },
      (row) => row.comparison_id,
    );
    if (!firstPage) {
      throw new ApiError(
        "The PAISA API returned an incomplete paginated result.",
        "malformed",
      );
    }
    return {
      ...firstPage,
      total,
      limit: Math.max(firstPage.limit, rows.length),
      offset: 0,
      rows,
    };
  }

  getUnifiedComparison(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedComparisonDetailResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}`,
      unifiedComparisonDetailSchema,
      comparisonId,
      signal,
    );
  }

  getUnifiedSources(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedSourcesResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}/sources`,
      unifiedSourcesSchema,
      comparisonId,
      signal,
    );
  }

  getUnifiedSummary(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedComparisonSummaryResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}/summary`,
      unifiedSummarySchema,
      comparisonId,
      signal,
    );
  }

  getUnifiedPredictionMetrics(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedPredictionMetricsResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}/prediction-metrics`,
      unifiedPredictionMetricsSchema,
      comparisonId,
      signal,
    );
  }

  getUnifiedValidationMetrics(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedValidationMetricsResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}/validation-metrics`,
      unifiedValidationMetricsSchema,
      comparisonId,
      signal,
    );
  }

  getUnifiedPortfolioMetrics(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedPortfolioMetricsResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}/portfolio-metrics`,
      unifiedPortfolioMetricsSchema,
      comparisonId,
      signal,
    );
  }

  getUnifiedPredictionRankings(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedPredictionRankingsResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}/prediction-rankings`,
      unifiedPredictionRankingsSchema,
      comparisonId,
      signal,
    );
  }

  getUnifiedPortfolioRankings(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedPortfolioRankingsResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}/portfolio-rankings`,
      unifiedPortfolioRankingsSchema,
      comparisonId,
      signal,
    );
  }

  getUnifiedCommonDateContract(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedCommonDateResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}/common-date-contract`,
      unifiedCommonDateSchema,
      comparisonId,
      signal,
    );
  }

  getUnifiedConfusionMatrices(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedConfusionMatricesResponse> {
    return this.requestForComparison(
      `/api/v1/unified-comparisons/${encodeURIComponent(comparisonId)}/confusion-matrices`,
      unifiedConfusionMatricesSchema,
      comparisonId,
      signal,
    );
  }

  async getUnifiedComparisonData(
    comparisonId: string,
    signal?: AbortSignal,
  ): Promise<UnifiedComparisonData> {
    const [
      detail,
      sources,
      summary,
      predictionMetrics,
      validationMetrics,
      portfolioMetrics,
      predictionRankings,
      portfolioRankings,
      commonDateContract,
      confusionMatrices,
    ] = await Promise.all([
      this.getUnifiedComparison(comparisonId, signal),
      this.getUnifiedSources(comparisonId, signal),
      this.getUnifiedSummary(comparisonId, signal),
      this.getUnifiedPredictionMetrics(comparisonId, signal),
      this.getUnifiedValidationMetrics(comparisonId, signal),
      this.getUnifiedPortfolioMetrics(comparisonId, signal),
      this.getUnifiedPredictionRankings(comparisonId, signal),
      this.getUnifiedPortfolioRankings(comparisonId, signal),
      this.getUnifiedCommonDateContract(comparisonId, signal),
      this.getUnifiedConfusionMatrices(comparisonId, signal),
    ]);
    const responses = [
      detail,
      sources,
      summary,
      predictionMetrics,
      validationMetrics,
      portfolioMetrics,
      predictionRankings,
      portfolioRankings,
      commonDateContract,
      confusionMatrices,
    ];
    const context = unifiedContextFingerprint(detail);
    if (responses.some((response) => unifiedContextFingerprint(response) !== context)) {
      throw new ApiError(
        "The PAISA API returned inconsistent unified-comparison context.",
        "malformed",
      );
    }
    if (
      detail.validation_count !== commonDateContract.validation.common_count ||
      detail.shared_holdout_count !==
        commonDateContract.shared_project_holdout.common_count ||
      summary.holdout_context.common_count !== detail.shared_holdout_count ||
      summary.holdout_context.target_start !==
        commonDateContract.shared_project_holdout.target_start ||
      summary.holdout_context.target_end !==
        commonDateContract.shared_project_holdout.target_end
    ) {
      throw new ApiError(
        "The PAISA API returned inconsistent common-date evidence.",
        "malformed",
      );
    }
    const registeredModels = new Map<string, string>(
      detail.models.map((model) => [model.variant_id, model.registered_model_id]),
    );
    const identityRows = [
      ...predictionMetrics.models,
      ...validationMetrics.models,
      ...portfolioMetrics.models.map((row) => ({
        competitor_id: row.variant_id,
        registered_model_id: row.registered_model_id,
      })),
    ];
    if (
      identityRows.some(
        (row) =>
          row.registered_model_id !== registeredModels.get(row.competitor_id),
      )
    ) {
      throw new ApiError(
        "The PAISA API returned inconsistent registered-model identity.",
        "malformed",
      );
    }
    return {
      detail,
      sources,
      summary,
      predictionMetrics,
      validationMetrics,
      portfolioMetrics,
      predictionRankings,
      portfolioRankings,
      commonDateContract,
      confusionMatrices,
    };
  }

  async listRuns(signal?: AbortSignal): Promise<VerticalSliceListResponse> {
    let firstPage: VerticalSliceListResponse | undefined;
    const { rows, total } = await this.collectPages<VerticalSliceListItem>(
      async (limit, offset) => {
        const page = await this.request(
          `/api/v1/vertical-slices?limit=${limit}&offset=${offset}`,
          verticalSliceListSchema,
          signal,
        );
        firstPage ??= page;
        return page;
      },
      (row) => row.run_id,
    );
    if (!firstPage) {
      throw new ApiError("The PAISA API returned an incomplete paginated result.", "malformed");
    }
    return {
      ...firstPage,
      total,
      limit: Math.max(firstPage.limit, rows.length),
      offset: 0,
      rows,
    };
  }

  getRun(runId: string, signal?: AbortSignal): Promise<VerticalSliceDetailResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}`,
      verticalSliceDetailSchema,
      runId,
      signal,
    );
  }

  getSummary(runId: string, signal?: AbortSignal): Promise<VerticalSliceSummaryResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/summary`,
      verticalSliceSummarySchema,
      runId,
      signal,
    );
  }

  getLeaderboard(runId: string, signal?: AbortSignal): Promise<LeaderboardResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/leaderboard`,
      leaderboardSchema,
      runId,
      signal,
    );
  }

  getModels(runId: string, signal?: AbortSignal): Promise<ModelsResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/models`,
      modelsSchema,
      runId,
      signal,
    );
  }

  getPortfolios(runId: string, signal?: AbortSignal): Promise<PortfoliosResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/portfolios`,
      portfoliosSchema,
      runId,
      signal,
    );
  }

  getSource(runId: string, signal?: AbortSignal): Promise<SourceResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/source`,
      sourceSchema,
      runId,
      signal,
    );
  }

  getExperiments(runId: string, signal?: AbortSignal): Promise<ExperimentsResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/experiments`,
      experimentsSchema,
      runId,
      signal,
    );
  }

  async getCoreRunData(runId: string, signal?: AbortSignal): Promise<CoreRunData> {
    const [detail, summary, leaderboard, models, portfolios, source, experiments] =
      await Promise.all([
        this.getRun(runId, signal),
        this.getSummary(runId, signal),
        this.getLeaderboard(runId, signal),
        this.getModels(runId, signal),
        this.getPortfolios(runId, signal),
        this.getSource(runId, signal),
        this.getExperiments(runId, signal),
      ]);
    [detail, summary, leaderboard, models, portfolios, source, experiments].forEach(
      (response) => assertRunIdentity(response, runId),
    );
    return { detail, summary, leaderboard, models, portfolios, source, experiments };
  }

  private getEquityPage(
    runId: string,
    limit: number,
    offset: number,
    signal?: AbortSignal,
  ): Promise<EquityCurvesResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/equity-curves?limit=${limit}&offset=${offset}`,
      equityCurvesSchema,
      runId,
      signal,
    );
  }

  private getDrawdownPage(
    runId: string,
    limit: number,
    offset: number,
    signal?: AbortSignal,
  ): Promise<DrawdownsResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/drawdowns?limit=${limit}&offset=${offset}`,
      drawdownsSchema,
      runId,
      signal,
    );
  }

  private async collectPages<T>(
    fetchPage: (limit: number, offset: number) => Promise<PageEnvelope<T>>,
    rowKey: (row: T) => string,
    expectedRunId?: string,
    pageSize = 1000,
  ): Promise<{ rows: T[]; total: number }> {
    const rows: T[] = [];
    const seenRowKeys = new Set<string>();
    let expectedTotal: number | undefined;
    let offset = 0;
    for (let page = 0; page < 100; page += 1) {
      const response = await fetchPage(pageSize, offset);
      if (expectedRunId !== undefined && response.run_id !== expectedRunId) {
        throw new ApiError("The PAISA API returned inconsistent run identity.", "malformed");
      }
      if (response.limit !== pageSize || response.offset !== offset) {
        throw new ApiError("The PAISA API returned inconsistent pagination metadata.", "malformed");
      }
      if (expectedTotal === undefined) {
        expectedTotal = response.total;
      } else if (response.total !== expectedTotal) {
        throw new ApiError("The PAISA API returned inconsistent pagination metadata.", "malformed");
      }
      if (
        response.rows.length > response.limit ||
        rows.length + response.rows.length > expectedTotal
      ) {
        throw new ApiError("The PAISA API returned inconsistent pagination metadata.", "malformed");
      }
      for (const row of response.rows) {
        const key = rowKey(row);
        if (seenRowKeys.has(key)) {
          throw new ApiError("The PAISA API returned duplicate paginated rows.", "malformed");
        }
        seenRowKeys.add(key);
        rows.push(row);
      }
      if (rows.length === expectedTotal) return { rows, total: expectedTotal };
      if (response.rows.length === 0) {
        throw new ApiError("The PAISA API returned an incomplete paginated result.", "malformed");
      }
      offset += response.rows.length;
    }
    throw new ApiError("The PAISA API result exceeded the safe pagination limit.", "malformed");
  }

  async getAllEquityCurves(runId: string, signal?: AbortSignal): Promise<EquityCurveRow[]> {
    const { rows } = await this.collectPages(
      (limit, offset) => this.getEquityPage(runId, limit, offset, signal),
      (row) => `${row.series_id}\u0000${row.date}`,
      runId,
    );
    return rows;
  }

  async getAllDrawdowns(runId: string, signal?: AbortSignal): Promise<DrawdownRow[]> {
    const { rows } = await this.collectPages(
      (limit, offset) => this.getDrawdownPage(runId, limit, offset, signal),
      (row) => `${row.series_id}\u0000${row.date}`,
      runId,
    );
    return rows;
  }

  private getHistoricalPricePage(
    runId: string,
    ticker: string,
    limit: number,
    offset: number,
    signal?: AbortSignal,
  ): Promise<HistoricalPricesPageResponse> {
    return this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/stocks/${encodeURIComponent(ticker)}/prices?limit=${limit}&offset=${offset}`,
      historicalPricesSchema,
      runId,
      signal,
    );
  }

  async getHistoricalPrices(
    runId: string,
    signal?: AbortSignal,
  ): Promise<HistoricalPricesResponse> {
    const ticker = "OGDC";
    let firstPage: HistoricalPricesPageResponse | undefined;
    let contextFingerprint: string | undefined;
    const { rows, total } = await this.collectPages(
      async (limit, offset) => {
        const response = await this.getHistoricalPricePage(
          runId,
          ticker,
          limit,
          offset,
          signal,
        );
        if (response.ticker !== ticker) {
          throw new ApiError("The PAISA API returned inconsistent ticker identity.", "malformed");
        }
        const fingerprint = JSON.stringify({
          schema_version: response.schema_version,
          run_id: response.run_id,
          ticker: response.ticker,
          source_mode: response.source_mode,
          data_date_range: response.data_date_range,
          warnings: response.warnings,
          disclaimer: response.disclaimer,
        });
        if (contextFingerprint === undefined) {
          contextFingerprint = fingerprint;
          firstPage = response;
        } else if (fingerprint !== contextFingerprint) {
          throw new ApiError(
            "The PAISA API returned inconsistent historical-price context.",
            "malformed",
          );
        }
        return response;
      },
      (row) => row.date,
      runId,
      HISTORICAL_PRICE_PAGE_SIZE,
    );
    if (!firstPage) {
      throw new ApiError("The PAISA API returned an incomplete paginated result.", "malformed");
    }
    for (let index = 1; index < rows.length; index += 1) {
      if (rows[index - 1]!.date >= rows[index]!.date) {
        throw new ApiError(
          "The PAISA API returned historical prices outside chronological order.",
          "malformed",
        );
      }
    }
    if (
      total > 0 &&
      (rows[0]?.date !== firstPage.data_date_range.start ||
        rows[rows.length - 1]?.date !== firstPage.data_date_range.end)
    ) {
      throw new ApiError(
        "The PAISA API returned inconsistent historical-price coverage.",
        "malformed",
      );
    }
    return {
      schema_version: firstPage.schema_version,
      run_id: firstPage.run_id,
      ticker: firstPage.ticker,
      source_mode: firstPage.source_mode,
      data_date_range: firstPage.data_date_range,
      total,
      rows,
      warnings: firstPage.warnings,
      disclaimer: firstPage.disclaimer,
    };
  }

  async getExperiment(
    runId: string,
    experimentId: string,
    signal?: AbortSignal,
  ): Promise<ExperimentDetailResponse> {
    const response = await this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/experiments/${encodeURIComponent(experimentId)}`,
      experimentDetailSchema,
      runId,
      signal,
    );
    assertExperimentIdentity(response.configuration.experiment_id, experimentId);
    assertExperimentIdentity(response.result.experiment_id, experimentId);
    return response;
  }

  async getTrades(
    runId: string,
    experimentId: string,
    limit: number,
    offset: number,
    signal?: AbortSignal,
  ): Promise<TradesResponse> {
    const response = await this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/experiments/${encodeURIComponent(experimentId)}/trades?limit=${limit}&offset=${offset}`,
      tradesSchema,
      runId,
      signal,
    );
    assertExperimentIdentity(response.experiment_id, experimentId);
    assertRequestedPage(response, limit, offset);
    return response;
  }

  async getSnapshots(
    runId: string,
    experimentId: string,
    limit: number,
    offset: number,
    signal?: AbortSignal,
  ): Promise<SnapshotsResponse> {
    const response = await this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/experiments/${encodeURIComponent(experimentId)}/snapshots?limit=${limit}&offset=${offset}`,
      snapshotsSchema,
      runId,
      signal,
    );
    assertExperimentIdentity(response.experiment_id, experimentId);
    assertRequestedPage(response, limit, offset);
    return response;
  }

  async getBenchmarks(
    runId: string,
    experimentId: string,
    limit: number,
    offset: number,
    signal?: AbortSignal,
  ): Promise<BenchmarksResponse> {
    const response = await this.requestForRun(
      `/api/v1/vertical-slices/${encodeURIComponent(runId)}/experiments/${encodeURIComponent(experimentId)}/benchmarks?limit=${limit}&offset=${offset}`,
      benchmarksSchema,
      runId,
      signal,
    );
    assertExperimentIdentity(response.experiment_id, experimentId);
    assertRequestedPage(response.cash_only, limit, offset);
    assertRequestedPage(response.buy_and_hold, limit, offset);
    assertRequestedPage(response.buy_and_hold_trades, limit, offset);
    return response;
  }
}

export const apiClient = new PaisaApiClient();

export function apiErrorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return "An unexpected dashboard error occurred.";
}
