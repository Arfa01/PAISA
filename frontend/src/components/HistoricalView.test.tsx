import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PaisaApiClient } from "../api/client";
import type { HistoricalPricesResponse } from "../api/types";
import {
  DISCLAIMER,
  OTHER_RUN_ID,
  RUN_ID,
  makeCoreRunData,
  makeMockClient,
} from "../test/fixtures";
import { HistoricalView } from "./HistoricalView";

interface PlotTrace {
  customdata?: string[];
  name?: string;
  x?: string[];
  y?: number[];
}

interface PlotLayout {
  yaxis?: { tickformat?: string };
}

function response(
  runId = RUN_ID,
  rows: HistoricalPricesResponse["rows"] = [
    {
      date: "2021-09-18",
      open: "103.00",
      high: null,
      low: null,
      close: "104.00",
      volume: 98_765,
      source: "psx_dps",
    },
    {
      date: "2021-09-16",
      open: "100.00",
      high: null,
      low: null,
      close: "101.25",
      volume: 1_234_567,
      source: "psx_dps",
    },
    {
      date: "2021-09-17",
      open: "101.00",
      high: null,
      low: null,
      close: "102.00",
      volume: 2_000,
      source: "psx_dps",
    },
  ],
): HistoricalPricesResponse {
  return {
    schema_version: "1.0",
    run_id: runId,
    ticker: "OGDC",
    source_mode: "verified_legacy_psx_cache",
    data_date_range: { start: "2021-09-16", end: "2021-09-18" },
    total: rows.length,
    rows,
    warnings: [],
    disclaimer: DISCLAIMER,
  };
}

function readPlotData(element: HTMLElement): PlotTrace[] {
  return JSON.parse(element.dataset.plotData ?? "[]") as PlotTrace[];
}

function readPlotLayout(element: HTMLElement): PlotLayout {
  return JSON.parse(element.dataset.plotLayout ?? "{}") as PlotLayout;
}

describe("HistoricalView", () => {
  it("sorts dates chronologically, omits null high/low, and formats volume", async () => {
    const client = makeMockClient();
    vi.mocked(client.getHistoricalPrices).mockResolvedValue(response());
    const core = makeCoreRunData();

    render(
      <HistoricalView
        client={client}
        detail={core.detail}
        summary={core.summary}
      />,
    );

    const charts = await screen.findAllByTestId("plotly-chart");
    expect(charts).toHaveLength(2);
    const priceChart = charts[0]!;
    const volumeChart = charts[1]!;
    const priceTraces = readPlotData(priceChart);
    const volumeTraces = readPlotData(volumeChart);

    expect(client.getHistoricalPrices).toHaveBeenCalledWith(
      RUN_ID,
      expect.any(AbortSignal),
    );
    expect(priceTraces.map((trace) => trace.name)).toEqual(["OGDC close"]);
    expect(priceTraces[0]?.x).toEqual([
      "2021-09-16",
      "2021-09-17",
      "2021-09-18",
    ]);
    expect(priceTraces[0]?.y).toEqual([101.25, 102, 104]);
    expect(volumeTraces[0]?.x).toEqual([
      "2021-09-16",
      "2021-09-17",
      "2021-09-18",
    ]);
    expect(volumeTraces[0]?.customdata).toEqual([
      "1,234,567",
      "2,000",
      "98,765",
    ]);
    expect(readPlotLayout(volumeChart).yaxis?.tickformat).toBe(",d");
    expect(screen.getByText("Historical date range").parentElement).toHaveTextContent(
      "16 Sep 2021 – 18 Sep 2021",
    );
    expect(screen.getByText("Price source").parentElement).toHaveTextContent(
      "Psx Dps",
    );
  });

  it("adds high and low traces only for dates where those values are present", async () => {
    const client = makeMockClient();
    vi.mocked(client.getHistoricalPrices).mockResolvedValue(
      response(RUN_ID, [
        {
          date: "2021-09-16",
          open: "100.00",
          high: "105.50",
          low: null,
          close: "104.00",
          volume: 1_000,
          source: "psx_dps",
        },
        {
          date: "2021-09-17",
          open: "103.00",
          high: null,
          low: "98.25",
          close: "100.00",
          volume: 2_000,
          source: "psx_dps",
        },
      ]),
    );
    const core = makeCoreRunData();

    render(
      <HistoricalView
        client={client}
        detail={core.detail}
        summary={core.summary}
      />,
    );

    const charts = await screen.findAllByTestId("plotly-chart");
    expect(charts).toHaveLength(2);
    const traces = readPlotData(charts[0]!);
    expect(traces.map((trace) => trace.name)).toEqual([
      "OGDC close",
      "OGDC high",
      "OGDC low",
    ]);
    expect(traces[1]?.x).toEqual(["2021-09-16"]);
    expect(traces[2]?.x).toEqual(["2021-09-17"]);
  });

  it("reloads prices with the new run ID when the selected run changes", async () => {
    const client = makeMockClient();
    vi.mocked(client.getHistoricalPrices).mockImplementation(async (runId) =>
      response(runId),
    );
    const firstCore = makeCoreRunData(RUN_ID);
    const nextCore = makeCoreRunData(OTHER_RUN_ID);

    const { rerender } = render(
      <HistoricalView
        client={client}
        detail={firstCore.detail}
        summary={firstCore.summary}
      />,
    );

    await screen.findByRole("img", {
      name: "Line chart of OGDC daily closing prices",
    });

    rerender(
      <HistoricalView
        client={client as PaisaApiClient}
        detail={nextCore.detail}
        summary={nextCore.summary}
      />,
    );

    await waitFor(() =>
      expect(client.getHistoricalPrices).toHaveBeenLastCalledWith(
        OTHER_RUN_ID,
        expect.any(AbortSignal),
      ),
    );
    expect(client.getHistoricalPrices).toHaveBeenCalledTimes(2);
  });

  it.each([
    ["NaN", { close: "NaN" }],
    ["Infinity", { close: "Infinity" }],
    ["/srv/paisa/private/prices.csv", { source: "/srv/paisa/private/prices.csv" }],
  ])("does not render unsafe historical value %s", async (unsafeValue, mutation) => {
    const base = response(RUN_ID, [
      {
        date: "2021-09-16",
        open: "100.00",
        high: null,
        low: null,
        close: "101.25",
        volume: 1_234,
        source: "psx_dps",
      },
    ]);
    const payload = {
      ...base,
      limit: 5_000,
      offset: 0,
      rows: [{ ...base.rows[0]!, ...mutation }],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue(payload),
      } as unknown as Response),
    );
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);
    const core = makeCoreRunData();

    const { container } = render(
      <HistoricalView
        client={client}
        detail={core.detail}
        summary={core.summary}
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Historical price API unavailable",
    );
    expect(container).not.toHaveTextContent(unsafeValue);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
