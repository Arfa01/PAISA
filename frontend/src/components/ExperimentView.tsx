import { useEffect, useMemo, useState, type ReactNode } from "react";

import {
  ApiError,
  apiErrorMessage,
  type PaisaApiClient,
} from "../api/client";
import type {
  BenchmarksResponse,
  ExperimentDetailResponse,
  ExperimentsResponse,
  SnapshotsResponse,
  TradesResponse,
} from "../api/types";
import {
  formatDate,
  formatInteger,
  formatNumber,
  formatPercent,
  formatPkr,
  humanize,
} from "../utils/format";

interface ExperimentViewProps {
  client: PaisaApiClient;
  runId: string;
  experiments: ExperimentsResponse;
}

interface FieldProps {
  label: string;
  children: ReactNode;
}

const PAGE_SIZE = 10;

function isCancelled(error: unknown): boolean {
  return error instanceof ApiError && error.kind === "cancelled";
}

function ensureRequestedPage(
  response: { limit: number; offset: number },
  requestedOffset: number,
): void {
  if (response.limit !== PAGE_SIZE || response.offset !== requestedOffset) {
    throw new ApiError(
      "The PAISA API returned inconsistent pagination metadata.",
      "malformed",
    );
  }
}

function Field({ label, children }: FieldProps) {
  return (
    <div className="definition-item">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function PageControls({
  label,
  offset,
  total,
  loading,
  onPrevious,
  onNext,
}: {
  label: string;
  offset: number;
  total: number;
  loading: boolean;
  onPrevious: () => void;
  onNext: () => void;
}) {
  const first = total === 0 ? 0 : offset + 1;
  const last = Math.min(offset + PAGE_SIZE, total);
  return (
    <div className="pagination" aria-label={`${label} pagination`}>
      <p aria-live="polite">
        Rows {formatInteger(first)}–{formatInteger(last)} of {formatInteger(total)}
      </p>
      <div className="pagination__actions">
        <button
          aria-label={`${label}: previous page`}
          className="button button--secondary"
          type="button"
          disabled={loading || offset === 0}
          onClick={onPrevious}
        >
          Previous
        </button>
        <button
          aria-label={`${label}: next page`}
          className="button button--secondary"
          type="button"
          disabled={loading || offset + PAGE_SIZE >= total}
          onClick={onNext}
        >
          Next
        </button>
      </div>
    </div>
  );
}

export function ExperimentView({
  client,
  runId,
  experiments,
}: ExperimentViewProps) {
  const [selectedExperimentId, setSelectedExperimentId] = useState(
    () => experiments.rows[0]?.experiment_id ?? "",
  );
  const [detail, setDetail] = useState<ExperimentDetailResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarksResponse | null>(null);
  const [benchmarksLoading, setBenchmarksLoading] = useState(false);
  const [benchmarksError, setBenchmarksError] = useState<string | null>(null);
  const [trades, setTrades] = useState<TradesResponse | null>(null);
  const [tradesLoading, setTradesLoading] = useState(false);
  const [tradesError, setTradesError] = useState<string | null>(null);
  const [tradeOffset, setTradeOffset] = useState(0);
  const [snapshots, setSnapshots] = useState<SnapshotsResponse | null>(null);
  const [snapshotsLoading, setSnapshotsLoading] = useState(false);
  const [snapshotsError, setSnapshotsError] = useState<string | null>(null);
  const [snapshotOffset, setSnapshotOffset] = useState(0);

  useEffect(() => {
    if (
      selectedExperimentId &&
      experiments.rows.some((row) => row.experiment_id === selectedExperimentId)
    ) {
      return;
    }
    setSelectedExperimentId(experiments.rows[0]?.experiment_id ?? "");
    setTradeOffset(0);
    setSnapshotOffset(0);
  }, [experiments.rows, selectedExperimentId]);

  useEffect(() => {
    if (!selectedExperimentId) {
      setDetail(null);
      setDetailLoading(false);
      setDetailError(null);
      return;
    }

    const controller = new AbortController();
    let current = true;
    setDetailLoading(true);
    setDetailError(null);

    void client
      .getExperiment(runId, selectedExperimentId, controller.signal)
      .then((response) => {
        if (current) setDetail(response);
      })
      .catch((requestError: unknown) => {
        if (current && !isCancelled(requestError)) {
          setDetailError(apiErrorMessage(requestError));
          setDetail(null);
        }
      })
      .finally(() => {
        if (current) setDetailLoading(false);
      });

    return () => {
      current = false;
      controller.abort();
    };
  }, [client, runId, selectedExperimentId]);

  useEffect(() => {
    if (!selectedExperimentId) {
      setBenchmarks(null);
      setBenchmarksLoading(false);
      setBenchmarksError(null);
      return;
    }

    const controller = new AbortController();
    let current = true;
    setBenchmarksLoading(true);
    setBenchmarksError(null);

    void client
      .getBenchmarks(
        runId,
        selectedExperimentId,
        PAGE_SIZE,
        0,
        controller.signal,
      )
      .then((response) => {
        ensureRequestedPage(response.cash_only, 0);
        ensureRequestedPage(response.buy_and_hold, 0);
        ensureRequestedPage(response.buy_and_hold_trades, 0);
        if (current) setBenchmarks(response);
      })
      .catch((requestError: unknown) => {
        if (current && !isCancelled(requestError)) {
          setBenchmarksError(apiErrorMessage(requestError));
          setBenchmarks(null);
        }
      })
      .finally(() => {
        if (current) setBenchmarksLoading(false);
      });

    return () => {
      current = false;
      controller.abort();
    };
  }, [client, runId, selectedExperimentId]);

  useEffect(() => {
    if (!selectedExperimentId) {
      setTrades(null);
      setTradesLoading(false);
      setTradesError(null);
      return;
    }

    const requestedOffset = tradeOffset;
    const controller = new AbortController();
    let current = true;
    setTradesLoading(true);
    setTradesError(null);

    void client
      .getTrades(
        runId,
        selectedExperimentId,
        PAGE_SIZE,
        requestedOffset,
        controller.signal,
      )
      .then((response) => {
        ensureRequestedPage(response, requestedOffset);
        if (current) setTrades(response);
      })
      .catch((requestError: unknown) => {
        if (current && !isCancelled(requestError)) {
          setTradesError(apiErrorMessage(requestError));
          setTrades(null);
        }
      })
      .finally(() => {
        if (current) setTradesLoading(false);
      });

    return () => {
      current = false;
      controller.abort();
    };
  }, [client, runId, selectedExperimentId, tradeOffset]);

  useEffect(() => {
    if (!selectedExperimentId) {
      setSnapshots(null);
      setSnapshotsLoading(false);
      setSnapshotsError(null);
      return;
    }

    const requestedOffset = snapshotOffset;
    const controller = new AbortController();
    let current = true;
    setSnapshotsLoading(true);
    setSnapshotsError(null);

    void client
      .getSnapshots(
        runId,
        selectedExperimentId,
        PAGE_SIZE,
        requestedOffset,
        controller.signal,
      )
      .then((response) => {
        ensureRequestedPage(response, requestedOffset);
        if (current) setSnapshots(response);
      })
      .catch((requestError: unknown) => {
        if (current && !isCancelled(requestError)) {
          setSnapshotsError(apiErrorMessage(requestError));
          setSnapshots(null);
        }
      })
      .finally(() => {
        if (current) setSnapshotsLoading(false);
      });

    return () => {
      current = false;
      controller.abort();
    };
  }, [client, runId, selectedExperimentId, snapshotOffset]);

  const selectedListRow = useMemo(
    () =>
      experiments.rows.find(
        (row) => row.experiment_id === selectedExperimentId,
      ) ?? null,
    [experiments.rows, selectedExperimentId],
  );
  const visibleDetail =
    detail?.run_id === runId &&
    detail.configuration.experiment_id === selectedExperimentId
      ? detail
      : null;
  const visibleBenchmarks =
    benchmarks?.run_id === runId &&
    benchmarks.experiment_id === selectedExperimentId &&
    benchmarks.cash_only.offset === 0 &&
    benchmarks.buy_and_hold.offset === 0 &&
    benchmarks.buy_and_hold_trades.offset === 0
      ? benchmarks
      : null;
  const visibleTrades =
    trades?.run_id === runId &&
    trades.experiment_id === selectedExperimentId &&
    trades.offset === tradeOffset
      ? trades
      : null;
  const visibleSnapshots =
    snapshots?.run_id === runId &&
    snapshots.experiment_id === selectedExperimentId &&
    snapshots.offset === snapshotOffset
      ? snapshots
      : null;

  if (experiments.rows.length === 0) {
    return (
      <section className="view experiment-view" aria-labelledby="experiment-heading">
        <header className="view-header">
          <div>
            <p className="eyebrow">Experiment inspection</p>
            <h2 id="experiment-heading">Experiment details</h2>
          </div>
        </header>
        <div className="state-panel state-panel--empty" role="status">
          This completed run exposes no experiments to inspect.
        </div>
      </section>
    );
  }

  return (
    <section className="view experiment-view" aria-labelledby="experiment-heading">
      <header className="view-header view-header--with-control">
        <div>
          <p className="eyebrow">Experiment inspection</p>
          <h2 id="experiment-heading">Experiment details</h2>
          <p className="view-description">
            Inspect the persisted configuration, execution assumptions, trades,
            snapshots, and benchmark results for one model portfolio.
          </p>
        </div>
        <label className="field-control" htmlFor="experiment-selector">
          <span>Experiment</span>
          <select
            id="experiment-selector"
            value={selectedExperimentId}
            onChange={(event) => {
              setSelectedExperimentId(event.target.value);
              setTradeOffset(0);
              setSnapshotOffset(0);
            }}
          >
            {experiments.rows.map((experiment) => (
              <option
                key={experiment.experiment_id}
                value={experiment.experiment_id}
              >
                {experiment.variant_id} · {humanize(experiment.algorithm)}
              </option>
            ))}
          </select>
        </label>
      </header>

      {selectedListRow ? (
        <div className="card-grid count-grid" aria-label="Selected experiment summary">
          <article className="metric-card">
            <span className="metric-card__label">Model variant</span>
            <strong>{selectedListRow.variant_id}</strong>
          </article>
          <article className="metric-card">
            <span className="metric-card__label">Algorithm</span>
            <strong>{humanize(selectedListRow.algorithm)}</strong>
          </article>
          <article className="metric-card">
            <span className="metric-card__label">Ending equity</span>
            <strong>{formatPkr(selectedListRow.ending_equity)}</strong>
          </article>
          <article className="metric-card">
            <span className="metric-card__label">Total return</span>
            <strong>{formatPercent(selectedListRow.total_return)}</strong>
          </article>
        </div>
      ) : null}

      {detailLoading && !visibleDetail ? (
        <div className="state-panel state-panel--loading" role="status" aria-live="polite">
          <span className="loading-indicator" aria-hidden="true" />
          Loading experiment configuration…
        </div>
      ) : null}
      {detailError ? (
        <div className="state-panel state-panel--error" role="alert">
          <h3>Experiment details unavailable</h3>
          <p>{detailError}</p>
        </div>
      ) : null}

      {visibleDetail ? (
        <>
          <div className="card-grid experiment-detail-grid">
            <article className="panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Identity and controls</p>
                  <h3>Configuration</h3>
                </div>
              </div>
              <dl className="definition-list">
                <Field label="Experiment ID">
                  <span className="code-value">{visibleDetail.configuration.experiment_id}</span>
                </Field>
                <Field label="Model variant">{visibleDetail.variant_id}</Field>
                <Field label="Registered model ID">
                  <span className="code-value">{visibleDetail.configuration.model_id}</span>
                </Field>
                <Field label="Portfolio ID">
                  <span className="code-value">{visibleDetail.configuration.portfolio_id}</span>
                </Field>
                <Field label="Initial cash">
                  {formatPkr(visibleDetail.configuration.initial_cash)}
                </Field>
                <Field label="Buy when probability ≥">
                  {formatPercent(
                    visibleDetail.configuration.thresholds.buy_probability_up_gte,
                  )}
                </Field>
                <Field label="Sell when probability ≤">
                  {formatPercent(
                    visibleDetail.configuration.thresholds.sell_probability_up_lte,
                  )}
                </Field>
                <Field label="Maximum allocation">
                  {formatPercent(visibleDetail.configuration.maximum_allocation)}
                </Field>
                <Field label="Evaluation feature dates">
                  {formatDate(
                    visibleDetail.configuration.evaluation_dates.feature_start,
                  )} –{" "}
                  {formatDate(
                    visibleDetail.configuration.evaluation_dates.feature_end,
                  )}
                </Field>
                <Field label="Evaluation target dates">
                  {formatDate(
                    visibleDetail.configuration.evaluation_dates.target_start,
                  )} –{" "}
                  {formatDate(
                    visibleDetail.configuration.evaluation_dates.target_end,
                  )}
                </Field>
                <Field label="Transaction cost">
                  {formatNumber(visibleDetail.configuration.transaction_cost.value)} bps ·{" "}
                  {humanize(visibleDetail.configuration.transaction_cost.type)}
                </Field>
                <Field label="Created">
                  {formatDate(visibleDetail.configuration.created_at)}
                </Field>
              </dl>
            </article>

            <article className="panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Simulation contract</p>
                  <h3>Execution assumptions</h3>
                </div>
              </div>
              <dl className="definition-list">
                <Field label="Signal timing">
                  {humanize(visibleDetail.configuration.execution_policy.signal_timing)}
                </Field>
                <Field label="Execution session">
                  {humanize(
                    visibleDetail.configuration.execution_policy.execution_session,
                  )}
                </Field>
                <Field label="Reference price">
                  {humanize(
                    visibleDetail.configuration.execution_policy.reference_price_field,
                  )}
                </Field>
                <Field label="Missing execution price">
                  {humanize(
                    visibleDetail.configuration.execution_policy
                      .missing_execution_price,
                  )}
                </Field>
                <Field label="Slippage">
                  {formatNumber(
                    visibleDetail.configuration.execution_policy.slippage_bps,
                  )} bps
                </Field>
                <Field label="Allocation mode">
                  {humanize(
                    visibleDetail.configuration.execution_policy.allocation_mode,
                  )}
                </Field>
                <Field label="Whole-share rounding">
                  {humanize(
                    visibleDetail.configuration.execution_policy.whole_share_rounding,
                  )}
                </Field>
                <Field label="Valuation price">
                  {humanize(
                    visibleDetail.configuration.execution_policy.valuation_price_field,
                  )}
                </Field>
                <Field label="Missing valuation price">
                  {humanize(
                    visibleDetail.configuration.execution_policy
                      .missing_valuation_price,
                  )}
                </Field>
                <Field label="Ending position">
                  {humanize(
                    visibleDetail.configuration.execution_policy.ending_position,
                  )}
                </Field>
                <Field label="Annualization factor">
                  {formatInteger(
                    visibleDetail.configuration.metric_assumptions.annualization_factor,
                  )}
                </Field>
                <Field label="Risk-free rate">
                  {formatPercent(
                    visibleDetail.configuration.metric_assumptions.risk_free_rate_annual,
                  )}
                </Field>
              </dl>
            </article>
          </div>

          <article className="panel" aria-labelledby="benchmark-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Persisted comparison</p>
                <h3 id="benchmark-heading">Benchmark comparison</h3>
              </div>
            </div>
            <div className="comparison-grid">
              <div className="comparison-card">
                <span>Model strategy</span>
                <strong>{formatPkr(visibleDetail.result.ending_equity)}</strong>
                <small>{formatPercent(visibleDetail.result.total_return)} return</small>
              </div>
              <div className="comparison-card">
                <span>Cash only</span>
                <strong>
                  {formatPkr(
                    visibleDetail.result.benchmark_comparison.cash_only.ending_equity,
                  )}
                </strong>
                <small>
                  {formatPercent(
                    visibleDetail.result.benchmark_comparison.cash_only.total_return,
                  )} return
                </small>
              </div>
              <div className="comparison-card">
                <span>Buy and hold</span>
                <strong>
                  {formatPkr(
                    visibleDetail.result.benchmark_comparison.buy_and_hold.ending_equity,
                  )}
                </strong>
                <small>
                  {formatPercent(
                    visibleDetail.result.benchmark_comparison.buy_and_hold.total_return,
                  )} return
                </small>
              </div>
              <div className="comparison-card">
                <span>Excess vs buy and hold</span>
                <strong>
                  {formatPercent(
                    visibleDetail.result.benchmark_comparison
                      .excess_return_vs_buy_and_hold,
                  )}
                </strong>
                <small>Model strategy difference</small>
              </div>
            </div>
            <p className="supporting-copy">{visibleDetail.actual_class_usage}</p>
            {benchmarksLoading && !visibleBenchmarks ? (
              <p role="status">Loading benchmark observations…</p>
            ) : null}
            {benchmarksError ? (
              <div className="inline-error" role="alert">
                Benchmark observations unavailable: {benchmarksError}
              </div>
            ) : null}
            {visibleBenchmarks ? (
              <p className="supporting-copy">
                API evidence: {formatInteger(visibleBenchmarks.cash_only.total)} cash-only
                snapshots, {" "}
                {formatInteger(visibleBenchmarks.buy_and_hold.total)} buy-and-hold
                snapshots, and {" "}
                {formatInteger(visibleBenchmarks.buy_and_hold_trades.total)} buy-and-hold
                trades.
              </p>
            ) : null}
          </article>
        </>
      ) : null}

      <article className="panel" aria-labelledby="trades-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Executed state transitions</p>
            <h3 id="trades-heading">Trade history</h3>
          </div>
        </div>
        {tradesLoading && !visibleTrades ? (
          <p role="status">Loading trade page…</p>
        ) : null}
        {tradesError ? (
          <div className="inline-error" role="alert">
            Trade history unavailable: {tradesError}
          </div>
        ) : null}
        {visibleTrades && visibleTrades.rows.length === 0 ? (
          <p className="state-panel state-panel--empty">No trades appear on this page.</p>
        ) : null}
        {visibleTrades && visibleTrades.rows.length > 0 ? (
          <div className="table-wrap">
            <table>
              <caption className="sr-only">
                Paginated trade history for the selected experiment
              </caption>
              <thead>
                <tr>
                  <th scope="col">Signal date</th>
                  <th scope="col">Execution date</th>
                  <th scope="col">Action</th>
                  <th scope="col">Quantity</th>
                  <th scope="col">Reference price</th>
                  <th scope="col">Execution price</th>
                  <th scope="col">Fee</th>
                  <th scope="col">Cash after</th>
                  <th scope="col">Shares after</th>
                </tr>
              </thead>
              <tbody>
                {visibleTrades.rows.map((trade) => (
                  <tr key={trade.trade_id}>
                    <td>{formatDate(trade.signal_date)}</td>
                    <td>{formatDate(trade.execution_date)}</td>
                    <td><span className="status-chip">{trade.action}</span></td>
                    <td>{formatInteger(trade.quantity)}</td>
                    <td>{formatPkr(trade.reference_price)}</td>
                    <td>{formatPkr(trade.execution_price)}</td>
                    <td>{formatPkr(trade.fee)}</td>
                    <td>{formatPkr(trade.cash_after_trade)}</td>
                    <td>{formatInteger(trade.shares_after_trade)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        <PageControls
          label="Trade history"
          offset={tradeOffset}
          total={visibleTrades?.total ?? 0}
          loading={tradesLoading}
          onPrevious={() =>
            setTradeOffset((offset) => Math.max(0, offset - PAGE_SIZE))
          }
          onNext={() => setTradeOffset((offset) => offset + PAGE_SIZE)}
        />
      </article>

      <article className="panel" aria-labelledby="snapshots-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Daily portfolio state</p>
            <h3 id="snapshots-heading">Portfolio snapshots</h3>
          </div>
        </div>
        {snapshotsLoading && !visibleSnapshots ? (
          <p role="status">Loading portfolio snapshot page…</p>
        ) : null}
        {snapshotsError ? (
          <div className="inline-error" role="alert">
            Portfolio snapshots unavailable: {snapshotsError}
          </div>
        ) : null}
        {visibleSnapshots && visibleSnapshots.rows.length === 0 ? (
          <p className="state-panel state-panel--empty">No portfolio snapshots appear on this page.</p>
        ) : null}
        {visibleSnapshots && visibleSnapshots.rows.length > 0 ? (
          <div className="table-wrap">
            <table>
              <caption className="sr-only">
                Paginated daily portfolio snapshots for the selected experiment
              </caption>
              <thead>
                <tr>
                  <th scope="col">Date</th>
                  <th scope="col">Cash</th>
                  <th scope="col">Shares</th>
                  <th scope="col">Market price</th>
                  <th scope="col">Position value</th>
                  <th scope="col">Total equity</th>
                  <th scope="col">Realized P/L</th>
                  <th scope="col">Unrealized P/L</th>
                </tr>
              </thead>
              <tbody>
                {visibleSnapshots.rows.map((snapshot) => (
                  <tr key={`${snapshot.date}-${snapshot.portfolio_id}`}>
                    <td>{formatDate(snapshot.date)}</td>
                    <td>{formatPkr(snapshot.cash)}</td>
                    <td>{formatInteger(snapshot.shares)}</td>
                    <td>{formatPkr(snapshot.market_price)}</td>
                    <td>{formatPkr(snapshot.position_value)}</td>
                    <td>{formatPkr(snapshot.total_equity)}</td>
                    <td>{formatPkr(snapshot.realized_profit_loss)}</td>
                    <td>{formatPkr(snapshot.unrealized_profit_loss)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        <PageControls
          label="Portfolio snapshots"
          offset={snapshotOffset}
          total={visibleSnapshots?.total ?? 0}
          loading={snapshotsLoading}
          onPrevious={() =>
            setSnapshotOffset((offset) => Math.max(0, offset - PAGE_SIZE))
          }
          onNext={() => setSnapshotOffset((offset) => offset + PAGE_SIZE)}
        />
      </article>
    </section>
  );
}
