import { describe, expect, it, vi } from "vitest";

import { PaisaApiClient } from "./client";
import type {
  EquityCurveRow,
  HistoricalPriceRow,
  HistoricalPricesPageResponse,
} from "./types";
import {
  EXPERIMENT_ONE,
  EXPERIMENT_TWO,
  makeCoreRunData,
  makeExperimentDetail,
  makeHistoricalPrices,
  makeRunList,
  makeRunListItem,
  OTHER_RUN_ID,
  RUN_ID,
  runContext,
} from "../test/fixtures";

function jsonResponse(payload: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(payload),
  } as unknown as Response;
}

const equityRow: EquityCurveRow = {
  date: "2026-01-05",
  series_id: "M-001",
  portfolio_type: "model",
  model_id: "M-001",
  portfolio_id: "portfolio-m-001",
  cash: "98979.90",
  shares: 10,
  market_price: "102.00",
  total_equity: "99999.90",
  drawdown: 0.001,
};

function historicalPricePage(
  rows: HistoricalPriceRow[],
  options: {
    offset?: number;
    total?: number;
    runId?: string;
    ticker?: string;
  } = {},
): HistoricalPricesPageResponse {
  const response = makeHistoricalPrices(options.runId);
  return {
    ...response,
    ticker: options.ticker ?? response.ticker,
    total: options.total ?? rows.length,
    limit: 5_000,
    offset: options.offset ?? 0,
    rows,
  };
}

describe("PaisaApiClient response validation", () => {
  it("fails closed on a malformed API response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        schema_version: "1.0",
        total: 1,
        limit: 1000,
        offset: 0,
        disclaimer: "Academic simulation; not financial advice.",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.listRuns()).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned data in an unexpected format.",
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not retry with or request a fixture endpoint after network failure", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("offline"));
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.listRuns()).rejects.toMatchObject({
      kind: "network",
      message: "Could not reach the PAISA API. Check that the backend is running.",
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      "http://127.0.0.1:8000/api/v1/vertical-slices?limit=1000&offset=0",
    );
  });

  it("classifies caller cancellation without logging or retrying", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const fetchMock = vi.fn(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            "abort",
            () => reject(new DOMException("Aborted", "AbortError")),
            { once: true },
          );
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);
    const controller = new AbortController();

    const request = client.listRuns(controller.signal);
    controller.abort();

    await expect(request).rejects.toMatchObject({
      kind: "cancelled",
      message: "The PAISA API request was cancelled.",
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(consoleError).not.toHaveBeenCalled();
    consoleError.mockRestore();
  });

  it("requires the exact academic disclaimer", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ ...makeRunList(), disclaimer: "" }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.listRuns()).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned data in an unexpected format.",
    });
  });

  it("validates unpaginated totals against their rows", async () => {
    const models = makeCoreRunData().models;
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ ...models, total: models.rows.length + 1 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getModels(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned data in an unexpected format.",
    });
  });

  it("rejects a response belonging to a different run", async () => {
    const models = makeCoreRunData(OTHER_RUN_ID).models;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(models)));
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getModels(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned inconsistent run identity.",
    });
  });

  it("rejects a detail response belonging to a different experiment", async () => {
    const detail = makeExperimentDetail(RUN_ID, EXPERIMENT_TWO, "M-002");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(detail)));
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getExperiment(RUN_ID, EXPERIMENT_ONE)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned inconsistent experiment identity.",
    });
  });

  it("collects every run-list page and preserves order", async () => {
    const first = makeRunListItem(RUN_ID);
    const second = makeRunListItem(OTHER_RUN_ID);
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ ...makeRunList([first]), total: 2, offset: 0 }),
      )
      .mockResolvedValueOnce(
        jsonResponse({ ...makeRunList([second]), total: 2, offset: 1 }),
      );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    const result = await client.listRuns();

    expect(result.rows.map((row) => row.run_id)).toEqual([RUN_ID, OTHER_RUN_ID]);
    expect(result.total).toBe(2);
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:8000/api/v1/vertical-slices?limit=1000&offset=1",
      expect.any(Object),
    );
  });

  it("rejects pagination when the reported total changes", async () => {
    const firstRow = { ...equityRow };
    const secondRow = { ...equityRow, date: "2026-01-06" };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          ...runContext(),
          total: 2,
          limit: 1000,
          offset: 0,
          rows: [firstRow],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          ...runContext(),
          total: 3,
          limit: 1000,
          offset: 1,
          rows: [secondRow],
        }),
      );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getAllEquityCurves(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned inconsistent pagination metadata.",
    });
  });

  it("rejects duplicate rows across paginated responses", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          ...runContext(),
          total: 2,
          limit: 1000,
          offset: 0,
          rows: [equityRow],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          ...runContext(),
          total: 2,
          limit: 1000,
          offset: 1,
          rows: [equityRow],
        }),
      );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getAllEquityCurves(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned duplicate paginated rows.",
    });
  });

  it("collects every run-scoped historical-price page without a legacy request", async () => {
    const history = makeHistoricalPrices();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(historicalPricePage([history.rows[0]!], { total: 2 })),
      )
      .mockResolvedValueOnce(
        jsonResponse(
          historicalPricePage([history.rows[1]!], { offset: 1, total: 2 }),
        ),
      );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    const response = await client.getHistoricalPrices(RUN_ID);

    expect(response.rows).toEqual(history.rows);
    expect(response.total).toBe(2);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      `http://127.0.0.1:8000/api/v1/vertical-slices/${RUN_ID}/stocks/OGDC/prices?limit=5000&offset=0`,
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      `http://127.0.0.1:8000/api/v1/vertical-slices/${RUN_ID}/stocks/OGDC/prices?limit=5000&offset=1`,
      expect.any(Object),
    );
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).startsWith("http://127.0.0.1:8000/stocks/"),
      ),
    ).toBe(false);
  });

  it("does not request a fixture or legacy route when historical prices fail", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("offline"));
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getHistoricalPrices(RUN_ID)).rejects.toMatchObject({
      kind: "network",
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      `http://127.0.0.1:8000/api/v1/vertical-slices/${RUN_ID}/stocks/OGDC/prices?limit=5000&offset=0`,
    );
  });

  it("rejects historical prices that are not chronological", async () => {
    const history = makeHistoricalPrices();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(historicalPricePage([...history.rows].reverse())),
      ),
    );
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getHistoricalPrices(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned historical prices outside chronological order.",
    });
  });

  it("rejects duplicate dates across historical-price pages", async () => {
    const history = makeHistoricalPrices();
    const duplicate = { ...history.rows[0]! };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(historicalPricePage([history.rows[0]!], { total: 2 })),
      )
      .mockResolvedValueOnce(
        jsonResponse(historicalPricePage([duplicate], { offset: 1, total: 2 })),
      );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getHistoricalPrices(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned duplicate paginated rows.",
    });
  });

  it("rejects historical-price context changes between pages", async () => {
    const history = makeHistoricalPrices();
    const changedContext = {
      ...historicalPricePage([history.rows[1]!], { offset: 1, total: 2 }),
      source_mode: "changed_source_mode",
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(historicalPricePage([history.rows[0]!], { total: 2 })),
      )
      .mockResolvedValueOnce(jsonResponse(changedContext));
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getHistoricalPrices(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned inconsistent historical-price context.",
    });
  });

  it("rejects a historical-price response for the wrong run or ticker", async () => {
    const history = makeHistoricalPrices();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(historicalPricePage(history.rows, { runId: OTHER_RUN_ID })),
      )
      .mockResolvedValueOnce(
        jsonResponse(historicalPricePage(history.rows, { ticker: "HBL" })),
      );
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getHistoricalPrices(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned inconsistent run identity.",
    });
    await expect(client.getHistoricalPrices(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned inconsistent ticker identity.",
    });
  });

  it.each([
    ["NaN close", { close: "NaN" }],
    ["infinite close", { close: "Infinity" }],
    ["unsafe volume", { volume: Number.MAX_SAFE_INTEGER + 1 }],
  ])("rejects %s in historical-price rows", async (_label, mutation) => {
    const history = makeHistoricalPrices();
    const row = { ...history.rows[0]!, ...mutation } as HistoricalPriceRow;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(historicalPricePage([row]))),
    );
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getHistoricalPrices(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API returned data in an unexpected format.",
    });
  });

  it("rejects an absolute path in historical-price source data", async () => {
    const history = makeHistoricalPrices();
    const row = {
      ...history.rows[0]!,
      source: "/srv/paisa/private/prices.csv",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(historicalPricePage([row]))),
    );
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.getHistoricalPrices(RUN_ID)).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API response contained prohibited local-path data.",
    });
  });

  it("rejects general POSIX backend paths but does not mistake web URLs for paths", async () => {
    const unsafeRun = makeRunListItem();
    unsafeRun.model_training_run_id = "/srv/paisa/models/model.joblib";
    const safeRun = makeRunListItem();
    safeRun.model_training_run_id =
      "https://models.example.test/usr/local/share/model.joblib";
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(makeRunList([unsafeRun])))
      .mockResolvedValueOnce(jsonResponse(makeRunList([safeRun])));
    vi.stubGlobal("fetch", fetchMock);
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.listRuns()).rejects.toMatchObject({
      kind: "malformed",
      message: "The PAISA API response contained prohibited local-path data.",
    });
    await expect(client.listRuns()).resolves.toMatchObject({
      rows: [{ model_training_run_id: safeRun.model_training_run_id }],
    });
  });

  it("never exposes arbitrary backend error details", async () => {
    const privateDetail =
      "Traceback: File /workspace/paisa/api.py: secret provider credentials";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: privateDetail }, 500)),
    );
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    await expect(client.listRuns()).rejects.toMatchObject({
      kind: "http",
      status: 500,
      message: "The PAISA API returned HTTP 500.",
    });
  });
});
