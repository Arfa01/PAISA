import type { Data } from "plotly.js";

import type {
  DrawdownRow,
  EquityCurveRow,
  PortfoliosResponse,
  VerticalSliceSummaryResponse,
} from "../api/types";
import {
  finiteNumber,
  formatInteger,
  formatNumber,
  formatPercent,
  formatPkr,
  formatPkrMinorUnits,
  humanize,
  parsePkrMinorUnits,
} from "../utils/format";
import { Plot } from "./Plot";

export interface PortfolioComparisonViewProps {
  portfolios: PortfoliosResponse;
  summary: VerticalSliceSummaryResponse;
  equityRows: EquityCurveRow[];
  drawdownRows: DrawdownRow[];
  loading: boolean;
  error: string | null;
}
interface SeriesIdentity {
  key: string;
  modelId: string | null;
  portfolioId: string;
  portfolioType: string;
  seriesId: string;
}

interface SeriesGroup<T> extends SeriesIdentity {
  rows: T[];
}

const SERIES_COLORS = [
  "#285b8f",
  "#2f8877",
  "#7c5cbf",
  "#c2762b",
  "#6a7685",
  "#9a4054",
];

function groupSeries<
  T extends {
    date: string;
    model_id: string | null;
    portfolio_id: string;
    portfolio_type: string;
    series_id: string;
  },
>(rows: T[]): SeriesGroup<T>[] {
  const groups = new Map<string, SeriesGroup<T>>();
  for (const row of rows) {
    const key = `${row.series_id}\u0000${row.portfolio_id}`;
    const existing = groups.get(key);
    if (existing) {
      existing.rows.push(row);
    } else {
      groups.set(key, {
        key,
        modelId: row.model_id,
        portfolioId: row.portfolio_id,
        portfolioType: row.portfolio_type,
        seriesId: row.series_id,
        rows: [row],
      });
    }
  }
  return Array.from(groups.values()).map((group) => ({
    ...group,
    rows: [...group.rows].sort((left, right) =>
      left.date.localeCompare(right.date),
    ),
  }));
}

function traceStyle(group: SeriesIdentity, index: number) {
  const fallbackColor = SERIES_COLORS[index % SERIES_COLORS.length] ?? "#285b8f";
  return {
    color: fallbackColor,
    dash:
      group.portfolioType === "cash_only"
        ? ("dot" as const)
        : group.portfolioType === "buy_and_hold"
          ? ("dash" as const)
          : ("solid" as const),
    width: group.portfolioType === "model" ? 2 : 3,
  };
}

function equityTraces(rows: EquityCurveRow[]): Data[] {
  return groupSeries(rows).map((group, index) => {
    const points = group.rows.flatMap((row) => {
      const minorUnits = parsePkrMinorUnits(row.total_equity);
      return minorUnits === null
        ? []
        : [
            {
              date: row.date,
              label: formatPkr(row.total_equity),
              minorUnits,
            },
          ];
    });
    return {
      type: "scatter",
      mode: "lines",
      name: group.seriesId,
      x: points.map((point) => point.date),
      // Plotly needs numeric coordinates. Integer paisa retain the API's exact
      // fixed-scale values without converting decimal PKR strings to floats.
      y: points.map((point) => point.minorUnits),
      customdata: points.map((point) => point.label),
      line: traceStyle(group, index),
      hovertemplate:
        "%{fullData.name}<br>Date: %{x|%d %b %Y}<br>Equity: %{customdata}<extra></extra>",
    };
  });
}

function equityAxisTicks(rows: EquityCurveRow[]) {
  const values = rows.flatMap((row) => {
    const value = parsePkrMinorUnits(row.total_equity);
    return value === null ? [] : [value];
  });
  if (values.length === 0) return {};
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (minimum === maximum) {
    return {
      tickmode: "array" as const,
      ticktext: [formatPkrMinorUnits(minimum)],
      tickvals: [minimum],
    };
  }

  const spread = maximum - minimum;
  if (!Number.isSafeInteger(spread)) {
    return {
      tickmode: "array" as const,
      ticktext: [formatPkrMinorUnits(minimum), formatPkrMinorUnits(maximum)],
      tickvals: [minimum, maximum],
    };
  }
  const roughStep = Math.max(1, spread / 4);
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  const step = Math.max(1, Math.round(multiplier * magnitude));
  const first = Math.ceil(minimum / step) * step;
  const tickvals: number[] = [];
  for (let value = first; value <= maximum && tickvals.length < 8; value += step) {
    tickvals.push(value);
  }
  if (tickvals.length === 0) tickvals.push(minimum, maximum);
  return {
    tickmode: "array" as const,
    ticktext: tickvals.map(formatPkrMinorUnits),
    tickvals,
  };
}

function seriesIntegrityError(
  portfolios: PortfoliosResponse,
  equityRows: EquityCurveRow[],
  drawdownRows: DrawdownRow[],
): string | null {
  const expected = new Map(
    portfolios.rows.map((row) => [
      row.model_id,
      {
        portfolioId: row.portfolio_id,
        portfolioType: row.portfolio_type,
      },
    ]),
  );
  if (expected.size !== portfolios.rows.length) {
    return "Portfolio metrics contain duplicate series identities.";
  }

  function validateRows(
    label: string,
    rows: Array<EquityCurveRow | DrawdownRow>,
  ): string | null {
    const groups = groupSeries(rows);
    const actualIds = new Set(groups.map((group) => group.seriesId));
    const missing = [...expected.keys()].filter((seriesId) => !actualIds.has(seriesId));
    const extra = [...actualIds].filter((seriesId) => !expected.has(seriesId));
    if (missing.length > 0 || extra.length > 0 || groups.length !== expected.size) {
      return `${label} do not contain exactly one series for every portfolio and benchmark.`;
    }
    for (const group of groups) {
      const identity = expected.get(group.seriesId);
      if (
        !identity ||
        identity.portfolioId !== group.portfolioId ||
        identity.portfolioType !== group.portfolioType
      ) {
        return `${label} contain a series identity that does not match the portfolio metrics.`;
      }
    }
    return null;
  }

  return (
    validateRows("Equity curves", equityRows) ??
    validateRows("Drawdown curves", drawdownRows)
  );
}

function drawdownTraces(rows: DrawdownRow[]): Data[] {
  return groupSeries(rows).map((group, index) => {
    const points = group.rows.flatMap((row) => {
      const value = finiteNumber(row.drawdown);
      return value === null ? [] : [{ date: row.date, value }];
    });
    return {
      type: "scatter",
      mode: "lines",
      name: group.seriesId,
      x: points.map((point) => point.date),
      y: points.map((point) => point.value),
      line: traceStyle(group, index),
      hovertemplate:
        "%{fullData.name}<br>Date: %{x|%d %b %Y}<br>Drawdown: %{y:.2%}<extra></extra>",
    };
  });
}

export function PortfolioComparisonView({
  portfolios,
  summary,
  equityRows,
  drawdownRows,
  loading,
  error,
}: PortfolioComparisonViewProps) {
  const bestPortfolio = summary.best_test_period_simulated_portfolio;
  const lowestDrawdown = summary.lowest_drawdown_portfolio;
  const integrityError =
    !loading && !error
      ? seriesIntegrityError(portfolios, equityRows, drawdownRows)
      : null;
  const modelPortfolios = portfolios.rows.filter(
    (row) => row.portfolio_type === "model",
  );
  const maximumTradeCount = modelPortfolios.reduce(
    (maximum, row) => Math.max(maximum, row.number_of_trades),
    0,
  );
  const mostActiveModels = modelPortfolios.filter(
    (row) => row.number_of_trades === maximumTradeCount,
  );
  const feeRows = modelPortfolios.flatMap((row) => {
    const minorUnits = parsePkrMinorUnits(row.fees_paid);
    return minorUnits === null ? [] : [{ minorUnits, row }];
  });
  const maximumFees = feeRows.reduce(
    (maximum, item) => Math.max(maximum, item.minorUnits),
    feeRows[0]?.minorUnits ?? 0,
  );
  const highestFeeModels = feeRows.filter(
    (item) => item.minorUnits === maximumFees,
  );

  return (
    <section aria-labelledby="portfolio-comparison-title" className="view-section">
      <header className="view-heading">
        <div>
          <p className="eyebrow">Academic simulation</p>
          <h2 id="portfolio-comparison-title">Portfolio comparison</h2>
        </div>
        <p>
          Model strategies are shown alongside independent cash-only and
          buy-and-hold benchmarks for the common test period.
        </p>
      </header>

      <div className="selection-summary">
        <article className="summary-card summary-card--accent">
          <span className="summary-card__label">
            Best simulated model portfolio
          </span>
          <strong>{bestPortfolio.model_id}</strong>
          <span>{formatPkr(bestPortfolio.ending_portfolio_value)}</span>
          <small>Total return {formatPercent(bestPortfolio.total_return)}</small>
        </article>
        <article className="summary-card">
          <span className="summary-card__label">
            Lowest-drawdown model portfolio
          </span>
          <strong>{lowestDrawdown.model_id}</strong>
          <span>
            Maximum drawdown {formatPercent(lowestDrawdown.maximum_drawdown)}
          </span>
          <small>Total return {formatPercent(lowestDrawdown.total_return)}</small>
        </article>
      </div>

      {portfolios.rows.length === 0 ? (
        <div className="state-panel state-panel--empty" role="status">
          No portfolio metrics are available for this run.
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <caption className="sr-only">
              Model portfolio and benchmark performance metrics
            </caption>
            <thead>
              <tr>
                <th scope="col">Portfolio</th>
                <th scope="col">Ending value</th>
                <th scope="col">Total return</th>
                <th scope="col">Maximum drawdown</th>
                <th scope="col">Volatility</th>
                <th scope="col">Sharpe ratio</th>
                <th scope="col">Trades</th>
                <th scope="col">Fees paid</th>
                <th scope="col">Return vs buy-and-hold</th>
              </tr>
            </thead>
            <tbody>
              {portfolios.rows.map((row) => {
                const isBest = row.portfolio_id === bestPortfolio.portfolio_id;
                const isLowestDrawdown =
                  row.portfolio_id === lowestDrawdown.portfolio_id;
                return (
                  <tr key={row.portfolio_id}>
                    <th scope="row">
                      <span className="table-primary">{row.model_id}</span>
                      <span className="table-secondary">
                        {humanize(row.portfolio_type)}
                      </span>
                      {isBest ? (
                        <span className="status-tag">Best simulated model</span>
                      ) : null}
                      {isLowestDrawdown ? (
                        <span className="status-tag status-tag--muted">
                          Lowest model drawdown
                        </span>
                      ) : null}
                    </th>
                    <td>{formatPkr(row.ending_portfolio_value)}</td>
                    <td>{formatPercent(row.total_return)}</td>
                    <td>{formatPercent(row.maximum_drawdown)}</td>
                    <td>{formatPercent(row.volatility)}</td>
                    <td>{formatNumber(row.sharpe_ratio)}</td>
                    <td>{formatInteger(row.number_of_trades)}</td>
                    <td>{formatPkr(row.fees_paid)}</td>
                    <td>{formatPercent(row.result_versus_buy_and_hold)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {mostActiveModels.length > 0 && highestFeeModels.length > 0 ? (
        <p className="criteria-note" aria-label="Trading activity and fee comparison">
          <strong>Trading activity and fees:</strong>{" "}
          {mostActiveModels.map(({ model_id }) => model_id).join(", ")}{" "}
          {mostActiveModels.length === 1 ? "has" : "have"} the
          highest model-portfolio trade count ({formatInteger(maximumTradeCount)}).
          {" "}
          {highestFeeModels.map(({ row }) => row.model_id).join(", ")}{" "}
          {highestFeeModels.length === 1 ? "has" : "have"} the
          highest reported model-portfolio fees ({formatPkr(
            highestFeeModels[0]?.row.fees_paid,
          )}). These comparisons are derived from the selected run’s API metrics.
        </p>
      ) : null}

      <div aria-live="polite" className="portfolio-charts">
        {error || integrityError ? (
          <div className="state-panel state-panel--error" role="alert">
            <strong>Unable to load portfolio charts.</strong>
            <span>{error ?? integrityError}</span>
          </div>
        ) : loading ? (
          <div className="state-panel state-panel--loading" role="status">
            Loading portfolio equity and drawdown curves…
          </div>
        ) : (
          <div className="chart-stack">
            {equityRows.length > 0 ? (
              <Plot
                ariaLabel="Line chart of total equity in PKR for every model portfolio, cash-only benchmark, and buy-and-hold benchmark."
                data={equityTraces(equityRows)}
                layout={{
                  hovermode: "x unified",
                  title: { text: "Portfolio and benchmark equity curves" },
                  xaxis: { title: { text: "Date" }, type: "date" },
                  yaxis: {
                    ...equityAxisTicks(equityRows),
                    title: { text: "Total equity (PKR)" },
                  },
                }}
              />
            ) : (
              <div className="state-panel state-panel--empty" role="status">
                No equity-curve points are available for this run.
              </div>
            )}

            {drawdownRows.length > 0 ? (
              <Plot
                ariaLabel="Line chart of drawdown percentages for every model portfolio, cash-only benchmark, and buy-and-hold benchmark."
                data={drawdownTraces(drawdownRows)}
                layout={{
                  hovermode: "x unified",
                  title: { text: "Portfolio and benchmark drawdowns" },
                  xaxis: { title: { text: "Date" }, type: "date" },
                  yaxis: {
                    rangemode: "tozero",
                    tickformat: ".0%",
                    title: { text: "Drawdown" },
                  },
                }}
              />
            ) : (
              <div className="state-panel state-panel--empty" role="status">
                No drawdown points are available for this run.
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
