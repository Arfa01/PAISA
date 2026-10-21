import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  apiErrorMessage,
  type PaisaApiClient,
} from "../api/client";
import type {
  HistoricalPriceRow,
  HistoricalPricesResponse,
  VerticalSliceDetailResponse,
  VerticalSliceSummaryResponse,
} from "../api/types";
import {
  finiteNumber,
  formatDate,
  formatInteger,
  humanize,
} from "../utils/format";
import { Plot } from "./Plot";

interface HistoricalViewProps {
  client: PaisaApiClient;
  detail: VerticalSliceDetailResponse;
  summary: VerticalSliceSummaryResponse;
}

const chartConfig = {
  displayModeBar: false,
  displaylogo: false,
  responsive: true,
} as const;

function isCancelled(error: unknown): boolean {
  return error instanceof ApiError && error.kind === "cancelled";
}

const datePattern = /^\d{4}-\d{2}-\d{2}$/;
const pricePattern = /^(-?)(\d+)\.(\d{2,18})$/;

interface PricePoint {
  date: string;
  label: string;
  value: number;
}

function formatPrice(value: string): string {
  const match = pricePattern.exec(value);
  if (!match) return "N/A";
  const sign = match[1] ?? "";
  const whole = match[2] ?? "";
  const fraction = match[3] ?? "";
  return `PKR ${sign}${whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}.${fraction}`;
}

function pricePoints(
  rows: HistoricalPriceRow[],
  getValue: (row: HistoricalPriceRow) => string | null,
): PricePoint[] {
  return rows.flatMap((row) => {
    const rawValue = getValue(row);
    const value = finiteNumber(rawValue);
    if (rawValue === null || value === null || !datePattern.test(row.date)) {
      return [];
    }
    return [{ date: row.date, label: formatPrice(rawValue), value }];
  });
}

export function HistoricalView({
  client,
  detail,
  summary,
}: HistoricalViewProps) {
  const [prices, setPrices] = useState<HistoricalPricesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let current = true;

    setLoading(true);
    setError(null);
    setPrices(null);

    void client
      .getHistoricalPrices(detail.run_id, controller.signal)
      .then((response) => {
        if (current) setPrices(response);
      })
      .catch((requestError: unknown) => {
        if (current && !isCancelled(requestError)) {
          setError(apiErrorMessage(requestError));
        }
      })
      .finally(() => {
        if (current) setLoading(false);
      });

    return () => {
      current = false;
      controller.abort();
    };
  }, [client, detail.run_id, requestVersion]);

  const historicalRows = useMemo(
    () => [...(prices?.rows ?? [])].sort((left, right) => left.date.localeCompare(right.date)),
    [prices],
  );
  const closeRows = useMemo(
    () => pricePoints(historicalRows, (row) => row.close),
    [historicalRows],
  );
  const highRows = useMemo(
    () => pricePoints(historicalRows, (row) => row.high),
    [historicalRows],
  );
  const lowRows = useMemo(
    () => pricePoints(historicalRows, (row) => row.low),
    [historicalRows],
  );
  const volumeRows = useMemo(
    () =>
      historicalRows.filter(
        (row) =>
          Number.isInteger(row.volume) &&
          row.volume >= 0 &&
          Number.isFinite(row.volume) &&
          datePattern.test(row.date),
      ),
    [historicalRows],
  );
  const providers = useMemo(
    () =>
      Array.from(
        new Set(
          historicalRows
            .map((row) => row.source)
            .filter((source): source is string => Boolean(source)),
        ),
      ).sort(),
    [historicalRows],
  );

  const priceTraces = useMemo(
    () => [
      {
        type: "scatter" as const,
        mode: "lines" as const,
        name: "OGDC close",
        x: closeRows.map((row) => row.date),
        y: closeRows.map((row) => row.value),
        customdata: closeRows.map((row) => row.label),
        line: { color: "#285b8f", width: 2 },
        hovertemplate: "%{x}<br>Close: %{customdata}<extra>OGDC</extra>",
      },
      ...(highRows.length > 0
        ? [
            {
              type: "scatter" as const,
              mode: "lines" as const,
              name: "OGDC high",
              x: highRows.map((row) => row.date),
              y: highRows.map((row) => row.value),
              customdata: highRows.map((row) => row.label),
              line: { color: "#2d7d65", dash: "dot" as const, width: 1 },
              hovertemplate: "%{x}<br>High: %{customdata}<extra>OGDC</extra>",
            },
          ]
        : []),
      ...(lowRows.length > 0
        ? [
            {
              type: "scatter" as const,
              mode: "lines" as const,
              name: "OGDC low",
              x: lowRows.map((row) => row.date),
              y: lowRows.map((row) => row.value),
              customdata: lowRows.map((row) => row.label),
              line: { color: "#a34a4a", dash: "dot" as const, width: 1 },
              hovertemplate: "%{x}<br>Low: %{customdata}<extra>OGDC</extra>",
            },
          ]
        : []),
    ],
    [closeRows, highRows, lowRows],
  );

  return (
    <section className="view historical-view" aria-labelledby="historical-heading">
      <header className="view-header">
        <div>
          <p className="eyebrow">Market history</p>
          <h2 id="historical-heading">Historical OGDC prices</h2>
          <p className="view-description">
            Daily closing prices and volume from the selected immutable vertical-slice
            run. This is an academic simulation and not financial advice.
          </p>
        </div>
      </header>

      <div className="card-grid count-grid" aria-label="Historical data context">
        <article className="metric-card">
          <span className="metric-card__label">Price source mode</span>
          <strong>{humanize(prices?.source_mode ?? detail.source_mode)}</strong>
        </article>
        <article className="metric-card">
          <span className="metric-card__label">Model feature coverage</span>
          <strong>
            {formatDate(summary.training_data.actual_feature_start)} –{" "}
            {formatDate(summary.training_data.actual_feature_end)}
          </strong>
        </article>
        <article className="metric-card">
          <span className="metric-card__label">Test feature period</span>
          <strong>
            {formatDate(detail.evaluation_dates.feature_start)} –{" "}
            {formatDate(detail.evaluation_dates.feature_end)}
          </strong>
        </article>
        <article className="metric-card">
          <span className="metric-card__label">Test target period</span>
          <strong>
            {formatDate(detail.evaluation_dates.target_start)} –{" "}
            {formatDate(detail.evaluation_dates.target_end)}
          </strong>
        </article>
      </div>

      <p className="supporting-copy">
        The current results API exposes overall model feature coverage and common
        test dates, but not distinct training or validation date ranges. Test
        feature dates and next-session target dates remain separate to preserve the
        evaluation contract.
      </p>

      {loading ? (
        <div className="state-panel state-panel--loading" role="status" aria-live="polite">
          <span className="loading-indicator" aria-hidden="true" />
          Loading historical prices from the PAISA API…
        </div>
      ) : null}

      {!loading && error ? (
        <div className="state-panel state-panel--error" role="alert">
          <div>
            <h3>Historical price API unavailable</h3>
            <p>{error}</p>
            <p>
              No fixture, sample, or synthetic price series has been substituted.
              Check that the selected run exposes its immutable historical-price
              snapshot, then retry.
            </p>
          </div>
          <button
            className="button button--secondary"
            type="button"
            onClick={() => setRequestVersion((version) => version + 1)}
          >
            Retry price request
          </button>
        </div>
      ) : null}

      {!loading && !error && prices && prices.rows.length === 0 ? (
        <div className="state-panel state-panel--empty" role="status">
          The run-scoped price API returned no OGDC observations. No chart can be
          drawn, and no fixture data has been substituted.
        </div>
      ) : null}

      {!loading && !error && prices && prices.rows.length > 0 ? (
        <>
          <div className="card-grid count-grid" aria-label="Price response details">
            <article className="metric-card">
              <span className="metric-card__label">Ticker</span>
              <strong>{prices.ticker}</strong>
            </article>
            <article className="metric-card">
              <span className="metric-card__label">Observations returned</span>
              <strong>{formatInteger(prices.total)}</strong>
            </article>
            <article className="metric-card">
              <span className="metric-card__label">Price source</span>
              <strong>
                {providers.length > 0
                  ? providers.map((provider) => humanize(provider)).join(", ")
                  : "N/A"}
              </strong>
            </article>
            <article className="metric-card">
              <span className="metric-card__label">Historical date range</span>
              <strong>
                {formatDate(prices.data_date_range.start)} –{" "}
                {formatDate(prices.data_date_range.end)}
              </strong>
            </article>
          </div>

          <div className="chart-stack historical-charts">
            <article className="panel" aria-labelledby="close-chart-heading">
              <div className="section-heading">
                <div>
                  <h3 id="close-chart-heading">OGDC daily closing price</h3>
                  <p>
                    Closing price in Pakistani rupees (PKR), with high and low
                    series only when the source provides them.
                  </p>
                </div>
              </div>
              {closeRows.length > 0 ? (
                <Plot
                  ariaLabel="Line chart of OGDC daily closing prices"
                  config={chartConfig}
                  data={priceTraces}
                  layout={{
                    autosize: true,
                    height: 390,
                    hovermode: "x unified",
                    xaxis: { title: { text: "Trading date" }, type: "date" },
                    yaxis: {
                      separatethousands: true,
                      tickprefix: "PKR ",
                      title: { text: "Price (PKR)" },
                    },
                    showlegend: priceTraces.length > 1,
                  }}
                />
              ) : (
                <p className="empty-state">No finite closing-price observations were returned.</p>
              )}
            </article>

            <article className="panel" aria-labelledby="volume-chart-heading">
              <div className="section-heading">
                <div>
                  <h3 id="volume-chart-heading">OGDC daily trading volume</h3>
                  <p>Reported share volume by trading date.</p>
                </div>
              </div>
              {volumeRows.length > 0 ? (
                <Plot
                  ariaLabel="Bar chart of OGDC daily trading volume"
                  config={chartConfig}
                  data={[
                    {
                      type: "bar",
                      name: "Volume",
                      x: volumeRows.map((row) => row.date),
                      y: volumeRows.map((row) => row.volume),
                      customdata: volumeRows.map((row) => formatInteger(row.volume)),
                      marker: { color: "#7c5cbf" },
                      hovertemplate: "%{x}<br>Volume: %{customdata}<extra>OGDC</extra>",
                    },
                  ]}
                  layout={{
                    autosize: true,
                    height: 340,
                    hovermode: "x unified",
                    xaxis: { title: { text: "Trading date" }, type: "date" },
                    yaxis: {
                      rangemode: "tozero",
                      separatethousands: true,
                      tickformat: ",d",
                      title: { text: "Volume (shares)" },
                    },
                    showlegend: false,
                  }}
                />
              ) : (
                <p className="empty-state">No finite volume observations were returned.</p>
              )}
            </article>
          </div>
        </>
      ) : null}
    </section>
  );
}
