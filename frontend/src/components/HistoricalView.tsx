import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  apiErrorMessage,
  type PaisaApiClient,
} from "../api/client";
import type {
  HistoricalPricesResponse,
  VerticalSliceDetailResponse,
  VerticalSliceSummaryResponse,
} from "../api/types";
import { formatDate, formatInteger, humanize } from "../utils/format";
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
      .getHistoricalPrices(controller.signal)
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

  const closeRows = useMemo(
    () =>
      (prices?.rows ?? []).filter(
        (row) => Number.isFinite(row.close) && /^\d{4}-\d{2}-\d{2}/.test(row.date),
      ),
    [prices],
  );
  const volumeRows = useMemo(
    () =>
      (prices?.rows ?? []).filter(
        (row) =>
          typeof row.volume === "number" &&
          Number.isFinite(row.volume) &&
          /^\d{4}-\d{2}-\d{2}/.test(row.date),
      ),
    [prices],
  );
  const providers = useMemo(
    () =>
      Array.from(
        new Set(
          (prices?.rows ?? [])
            .map((row) => row.source)
            .filter((source): source is string => Boolean(source)),
        ),
      ).sort(),
    [prices],
  );

  return (
    <section className="view historical-view" aria-labelledby="historical-heading">
      <header className="view-header">
        <div>
          <p className="eyebrow">Market history</p>
          <h2 id="historical-heading">Historical OGDC prices</h2>
          <p className="view-description">
            Daily closing prices and volume returned by the legacy PAISA stock-price
            API. These observations are shown as source data and are not trading
            recommendations.
          </p>
        </div>
      </header>

      <div className="card-grid count-grid" aria-label="Historical data context">
        <article className="metric-card">
          <span className="metric-card__label">Run source mode</span>
          <strong>{humanize(detail.source_mode)}</strong>
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
              No fixture or synthetic price series has been substituted. Start the
              backend with a populated legacy processed-price artifact, then retry.
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
          The stock-price API returned no OGDC observations. No chart can be drawn.
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
              <strong>{formatInteger(prices.count)}</strong>
            </article>
            <article className="metric-card">
              <span className="metric-card__label">Price provider</span>
              <strong>
                {providers.length > 0
                  ? providers.map((provider) => humanize(provider)).join(", ")
                  : humanize(summary.training_data.provider)}
              </strong>
            </article>
            <article className="metric-card">
              <span className="metric-card__label">Run data coverage</span>
              <strong>
                {formatDate(detail.data_date_range.start)} –{" "}
                {formatDate(detail.data_date_range.end)}
              </strong>
            </article>
          </div>

          <div className="chart-stack historical-charts">
            <article className="panel" aria-labelledby="close-chart-heading">
              <div className="section-heading">
                <div>
                  <h3 id="close-chart-heading">OGDC daily closing price</h3>
                  <p>Closing price in Pakistani rupees (PKR).</p>
                </div>
              </div>
              {closeRows.length > 0 ? (
                <Plot
                  ariaLabel="Line chart of OGDC daily closing prices"
                  config={chartConfig}
                  data={[
                    {
                      type: "scatter",
                      mode: "lines",
                      name: "OGDC close",
                      x: closeRows.map((row) => row.date),
                      y: closeRows.map((row) => row.close),
                      line: { color: "#285b8f", width: 2 },
                      hovertemplate: "%{x}<br>Close: PKR %{y:,.2f}<extra>OGDC</extra>",
                    },
                  ]}
                  layout={{
                    autosize: true,
                    height: 390,
                    hovermode: "x unified",
                    xaxis: { title: { text: "Trading date" }, type: "date" },
                    yaxis: {
                      separatethousands: true,
                      tickprefix: "PKR ",
                      title: { text: "Closing price (PKR)" },
                    },
                    showlegend: false,
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
                      y: volumeRows.map((row) => row.volume as number),
                      marker: { color: "#7c5cbf" },
                      hovertemplate: "%{x}<br>Volume: %{y:,.0f}<extra>OGDC</extra>",
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
