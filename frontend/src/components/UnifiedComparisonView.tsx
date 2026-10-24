import { useEffect, useState } from "react";

import {
  apiErrorMessage,
  type PaisaApiClient,
} from "../api/client";
import type {
  UnifiedComparisonData,
  UnifiedComparisonListResponse,
} from "../api/types";
import {
  formatInteger,
  formatNumber,
  formatPercent,
  formatPkr,
  humanize,
} from "../utils/format";

interface UnifiedComparisonViewProps {
  client: PaisaApiClient;
}

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

const initialInventoryState = (): AsyncState<UnifiedComparisonListResponse> => ({
  data: null,
  loading: true,
  error: null,
});

const initialComparisonState = (): AsyncState<UnifiedComparisonData> => ({
  data: null,
  loading: false,
  error: null,
});

export function UnifiedComparisonView({ client }: UnifiedComparisonViewProps) {
  const [inventory, setInventory] = useState(initialInventoryState);
  const [selectedComparisonId, setSelectedComparisonId] = useState("");
  const [comparison, setComparison] = useState(initialComparisonState);
  const [reloadInventory, setReloadInventory] = useState(0);
  const [reloadComparison, setReloadComparison] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setInventory(initialInventoryState());
    setSelectedComparisonId("");
    setComparison(initialComparisonState());
    client
      .listUnifiedComparisons(controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return;
        setInventory({ data: response, loading: false, error: null });
        setSelectedComparisonId(
          response.default_comparison_id ?? response.rows[0]?.comparison_id ?? "",
        );
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setInventory({ data: null, loading: false, error: apiErrorMessage(error) });
      });
    return () => controller.abort();
  }, [client, reloadInventory]);

  useEffect(() => {
    if (!selectedComparisonId) {
      setComparison(initialComparisonState());
      return;
    }
    const controller = new AbortController();
    setComparison({ data: null, loading: true, error: null });
    client
      .getUnifiedComparisonData(selectedComparisonId, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) {
          setComparison({ data: response, loading: false, error: null });
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setComparison({ data: null, loading: false, error: apiErrorMessage(error) });
        }
      });
    return () => controller.abort();
  }, [client, reloadComparison, selectedComparisonId]);

  const visibleComparison =
    comparison.data?.detail.comparison_id === selectedComparisonId
      ? comparison.data
      : null;
  const selectedIsUnavailable = Boolean(
    selectedComparisonId &&
      inventory.data &&
      !inventory.data.rows.some(
        (row) => row.comparison_id === selectedComparisonId,
      ),
  );

  if (inventory.loading) {
    return <ComparisonLoading label="Loading completed unified comparisons…" />;
  }
  if (inventory.error) {
    return (
      <ComparisonError
        title="Could not load unified comparisons"
        message={inventory.error}
        onRetry={() => setReloadInventory((value) => value + 1)}
      />
    );
  }
  if (!inventory.data || inventory.data.rows.length === 0) {
    return (
      <section className="state-panel empty-state" role="status">
        <span className="state-icon" aria-hidden="true">○</span>
        <p className="eyebrow">No completed comparisons</p>
        <h1>No immutable unified comparison runs are available.</h1>
        <p>
          The dashboard will not substitute fixture, fabricated, or recomputed
          comparison results.
        </p>
        <button
          className="button button-primary"
          type="button"
          onClick={() => setReloadInventory((value) => value + 1)}
        >
          Check again
        </button>
      </section>
    );
  }

  return (
    <section
      className="view unified-comparison-view"
      aria-labelledby="unified-comparison-title"
    >
      <header className="view-header view-header--with-control">
        <div>
          <p className="eyebrow">Authoritative price-only evidence</p>
          <h1 id="unified-comparison-title">Unified Six-Model Comparison</h1>
          <p className="view-description">
            M-001 through M-006 are reported on the same validation dates and
            shared project holdout. Validation selection, prediction quality,
            and simulated portfolio outcomes remain separate.
          </p>
        </div>
        <label className="field-control" htmlFor="comparison-run-select">
          <span>Comparison run</span>
          <select
            id="comparison-run-select"
            value={selectedComparisonId}
            onChange={(event) => {
              setSelectedComparisonId(event.target.value);
              setComparison(initialComparisonState());
            }}
          >
            {selectedIsUnavailable ? (
              <option value={selectedComparisonId}>
                {selectedComparisonId} (configured but unavailable)
              </option>
            ) : null}
            {inventory.data.rows.map((row) => (
              <option value={row.comparison_id} key={row.comparison_id}>
                {row.comparison_id}{row.is_default ? " · default" : ""}
              </option>
            ))}
          </select>
        </label>
      </header>

      {comparison.loading ? (
        <ComparisonLoading label={`Loading ${selectedComparisonId}…`} compact />
      ) : comparison.error ? (
        <ComparisonError
          title="This unified comparison could not be loaded"
          message={comparison.error}
          onRetry={() => setReloadComparison((value) => value + 1)}
          compact
        />
      ) : visibleComparison ? (
        <UnifiedComparisonResults data={visibleComparison} />
      ) : null}
    </section>
  );
}

function UnifiedComparisonResults({ data }: { data: UnifiedComparisonData }) {
  const { detail, summary, validationMetrics, predictionMetrics, portfolioMetrics } =
    data;
  const validationByModel = new Map<
    string,
    (typeof validationMetrics.models)[number]
  >(
    validationMetrics.models.map((row) => [row.competitor_id, row]),
  );
  const portfolioByModel = new Map<
    string,
    (typeof portfolioMetrics.models)[number]
  >(
    portfolioMetrics.models.map((row) => [row.variant_id, row]),
  );
  const outcomes = summary.reported_outcomes;
  const cash = portfolioMetrics.benchmarks.find(
    (row) => row.benchmark_id === "cash-only",
  );
  const buyAndHold = portfolioMetrics.benchmarks.find(
    (row) => row.benchmark_id === "buy-and-hold",
  );

  return (
    <>
      <aside className="disclaimer" aria-label="Academic simulation disclaimer">
        {summary.disclaimer}
      </aside>

      <section className="panel" aria-labelledby="unified-context-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Evaluation context</p>
            <h2 id="unified-context-title">Shared project holdout</h2>
          </div>
          <p>
            This holdout was previously exposed through earlier project work;
            it is not presented as a fresh independent test set.
          </p>
        </div>
        <dl className="metadata-grid">
          <div>
            <dt>Comparison ID</dt>
            <dd className="identifier">{detail.comparison_id}</dd>
          </div>
          <div>
            <dt>Status</dt>
            <dd>{detail.status}</dd>
          </div>
          <div>
            <dt>Symbol</dt>
            <dd>{detail.symbol}</dd>
          </div>
          <div>
            <dt>Models compared</dt>
            <dd>{formatInteger(detail.models.length)}</dd>
          </div>
          <div>
            <dt>Validation observations</dt>
            <dd>{formatInteger(data.commonDateContract.validation.common_count)}</dd>
          </div>
          <div>
            <dt>Shared holdout observations</dt>
            <dd>
              {formatInteger(
                data.commonDateContract.shared_project_holdout.common_count,
              )}
            </dd>
          </div>
          <div>
            <dt>Shared holdout target start</dt>
            <dd>{data.commonDateContract.shared_project_holdout.target_start}</dd>
          </div>
          <div>
            <dt>Shared holdout target end</dt>
            <dd>{data.commonDateContract.shared_project_holdout.target_end}</dd>
          </div>
        </dl>
      </section>

      <section className="panel" aria-labelledby="separate-results-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Separate criteria</p>
            <h2 id="separate-results-title">Authoritative reported outcomes</h2>
          </div>
          <p>
            These results are deliberately not combined into a cross-criterion
            score or a single model selection.
          </p>
        </div>
        <div className="card-grid unified-result-grid">
          <OutcomeCard
            label="Validation-selected model"
            primary={outcomes.validation_selected_model.variant_id}
            details={`Balanced accuracy ${formatPercent(outcomes.validation_selected_model.balanced_accuracy)}`}
          />
          <OutcomeCard
            label="Best shared project holdout balanced accuracy"
            primary={outcomes.best_shared_holdout_balanced_accuracy.variant_id}
            details={formatPercent(
              outcomes.best_shared_holdout_balanced_accuracy.balanced_accuracy,
            )}
          />
          <OutcomeCard
            label="Best shared project holdout F1"
            primary={outcomes.best_shared_holdout_f1.variant_id}
            details={formatPercent(outcomes.best_shared_holdout_f1.f1)}
          />
          <OutcomeCard
            label="Best model portfolio"
            primary={outcomes.best_simulated_model_portfolio.variant_id}
            details={`${formatPkr(outcomes.best_simulated_model_portfolio.ending_equity)} · ${formatPercent(outcomes.best_simulated_model_portfolio.total_return)}`}
          />
          <OutcomeCard
            label="Lowest drawdown"
            primary={outcomes.lowest_drawdown_model_portfolio.variant_id}
            details={formatPercent(
              outcomes.lowest_drawdown_model_portfolio.maximum_drawdown,
            )}
          />
          <OutcomeCard
            label="Lowest fees"
            primary={outcomes.lowest_fee_model_portfolio.variant_id}
            details={formatPkr(outcomes.lowest_fee_model_portfolio.fees_paid)}
          />
          <OutcomeCard
            label="Buy-and-hold"
            primary={formatPkr(outcomes.buy_and_hold_result.ending_equity)}
            details={`Return ${formatPercent(outcomes.buy_and_hold_result.total_return)}`}
          />
          <OutcomeCard
            label="Any model beat buy-and-hold?"
            primary={outcomes.any_model_beat_buy_and_hold.value ? "Yes" : "No"}
            details={
              outcomes.any_model_beat_buy_and_hold.value
                ? "At least one model portfolio had a higher simulated total return."
                : "No model portfolio had a higher simulated total return."
            }
          />
        </div>
        <p className="criteria-note">
          <strong>Different measurements:</strong> Better classification
          performance on the shared project holdout did not necessarily produce
          better simulated portfolio performance.
        </p>
      </section>

      <section className="panel" aria-labelledby="six-model-table-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">M-001 through M-006</p>
            <h2 id="six-model-table-title">Model and portfolio evidence</h2>
          </div>
          <p>Rows retain the backend&apos;s canonical order and persisted values.</p>
        </div>
        <div className="table-wrap">
          <table aria-label="Unified six-model prediction and portfolio metrics">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Algorithm</th>
                <th scope="col">Configuration family</th>
                <th scope="col">Validation balanced accuracy</th>
                <th scope="col">Validation F1</th>
                <th scope="col">Shared holdout accuracy</th>
                <th scope="col">Shared holdout balanced accuracy</th>
                <th scope="col">Shared holdout F1</th>
                <th scope="col">Ending equity</th>
                <th scope="col">Portfolio return</th>
                <th scope="col">Maximum drawdown</th>
                <th scope="col">Trades</th>
                <th scope="col">Fees</th>
              </tr>
            </thead>
            <tbody>
              {predictionMetrics.models.map((prediction) => {
                const validation = validationByModel.get(prediction.competitor_id);
                const portfolio = portfolioByModel.get(prediction.competitor_id);
                return (
                  <tr key={prediction.competitor_id}>
                    <th scope="row" className="table-primary">
                      {prediction.competitor_id}
                    </th>
                    <td>{humanize(prediction.algorithm)}</td>
                    <td>{prediction.configuration_family}</td>
                    <td>{formatPercent(validation?.balanced_accuracy)}</td>
                    <td>{formatPercent(validation?.f1)}</td>
                    <td>{formatPercent(prediction.accuracy)}</td>
                    <td>{formatPercent(prediction.balanced_accuracy)}</td>
                    <td>{formatPercent(prediction.f1)}</td>
                    <td>{formatPkr(portfolio?.ending_equity)}</td>
                    <td>{formatPercent(portfolio?.total_return)}</td>
                    <td>{formatPercent(portfolio?.maximum_drawdown)}</td>
                    <td>{formatInteger(portfolio?.number_of_trades)}</td>
                    <td>{formatPkr(portfolio?.fees_paid)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel" aria-labelledby="benchmark-results-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Controlled comparators</p>
            <h2 id="benchmark-results-title">Cash and buy-and-hold benchmarks</h2>
          </div>
        </div>
        <div className="comparison-grid">
          <BenchmarkCard label="Cash only" benchmark={cash} />
          <BenchmarkCard label="Buy-and-hold" benchmark={buyAndHold} />
        </div>
      </section>

      <section className="panel" aria-labelledby="validation-evidence-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Selection evidence</p>
            <h2 id="validation-evidence-title">Validation-only selection</h2>
          </div>
          <p>
            Shared project holdout metrics were not used to select or tune a
            prediction model.
          </p>
        </div>
        <dl className="metadata-grid">
          <div>
            <dt>Selected model</dt>
            <dd>{outcomes.validation_selected_model.variant_id}</dd>
          </div>
          <div>
            <dt>Primary criterion</dt>
            <dd>{humanize(outcomes.validation_selected_model.criterion)}</dd>
          </div>
          <div>
            <dt>LSTM validation preference</dt>
            <dd>{outcomes.lstm_validation_preferred_model.variant_id}</dd>
          </div>
          <div>
            <dt>Outcomes combined</dt>
            <dd>{summary.outcomes_are_not_combined ? "No" : "N/A"}</dd>
          </div>
        </dl>
      </section>

      <section className="panel" aria-labelledby="confusion-matrices-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Classification counts</p>
            <h2 id="confusion-matrices-title">Shared holdout confusion matrices</h2>
          </div>
        </div>
        <div className="unified-matrix-grid">
          {data.confusionMatrices.shared_project_holdout_models.map((row) => (
            <article className="comparison-card" key={row.competitor_id}>
              <span>{row.competitor_id}</span>
              <table aria-label={`${row.competitor_id} shared holdout confusion matrix`}>
                <thead>
                  <tr>
                    <th scope="col">Actual / predicted</th>
                    <th scope="col">Down</th>
                    <th scope="col">Up</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row">Down</th>
                    <td>{formatInteger(row.confusion_matrix[0]?.[0])}</td>
                    <td>{formatInteger(row.confusion_matrix[0]?.[1])}</td>
                  </tr>
                  <tr>
                    <th scope="row">Up</th>
                    <td>{formatInteger(row.confusion_matrix[1]?.[0])}</td>
                    <td>{formatInteger(row.confusion_matrix[1]?.[1])}</td>
                  </tr>
                </tbody>
              </table>
            </article>
          ))}
        </div>
      </section>

      <section className="panel" aria-labelledby="comparison-sources-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Immutable provenance</p>
            <h2 id="comparison-sources-title">Source references</h2>
          </div>
          <p>No fitted model binary is loaded or exposed by this view.</p>
        </div>
        <div className="table-wrap">
          <table aria-label="Unified comparison source runs">
            <thead>
              <tr>
                <th scope="col">Role</th>
                <th scope="col">Artifact type</th>
                <th scope="col">Run ID</th>
                <th scope="col">Payload files</th>
              </tr>
            </thead>
            <tbody>
              {data.sources.sources.map((source) => (
                <tr key={source.role}>
                  <th scope="row">{humanize(source.role)}</th>
                  <td>{humanize(source.artifact_type)}</td>
                  <td className="identifier">{source.run_id}</td>
                  <td>{formatInteger(source.payload_file_count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="notice-grid">
        <section className="panel notice-panel" aria-labelledby="comparison-limits-title">
          <h2 id="comparison-limits-title">Limitations</h2>
          <ul className="notice-list">
            {summary.limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
        </section>
        <section className="panel notice-panel" aria-labelledby="comparison-scope-title">
          <h2 id="comparison-scope-title">Scope</h2>
          <p>
            Price-only OGDC evidence, one historical period, and one controlled
            simulation policy. Results do not establish future performance.
          </p>
        </section>
      </div>
    </>
  );
}

function OutcomeCard({
  label,
  primary,
  details,
}: {
  label: string;
  primary: string;
  details: string;
}) {
  return (
    <article className="metric-card result-card">
      <p className="card-label">{label}</p>
      <h3>{primary}</h3>
      <p>{details}</p>
    </article>
  );
}

function BenchmarkCard({
  label,
  benchmark,
}: {
  label: string;
  benchmark:
    | UnifiedComparisonData["portfolioMetrics"]["benchmarks"][number]
    | undefined;
}) {
  return (
    <article className="comparison-card">
      <span>{label}</span>
      <strong>{formatPkr(benchmark?.ending_equity)}</strong>
      <small>
        Return {formatPercent(benchmark?.total_return)} · Drawdown {" "}
        {formatPercent(benchmark?.maximum_drawdown)} · Trades {" "}
        {formatInteger(benchmark?.number_of_trades)} · Fees {" "}
        {formatPkr(benchmark?.fees_paid)} · Sharpe {" "}
        {formatNumber(benchmark?.sharpe_ratio)}
      </small>
    </article>
  );
}

function ComparisonLoading({ label, compact = false }: { label: string; compact?: boolean }) {
  return (
    <section
      className={`state-panel loading-state${compact ? " state-panel--loading" : ""}`}
      role="status"
      aria-live="polite"
    >
      <div className="loading-orbit" aria-hidden="true"><span /></div>
      <p className="eyebrow">Unified comparison API</p>
      <h1>{label}</h1>
    </section>
  );
}

function ComparisonError({
  title,
  message,
  onRetry,
  compact = false,
}: {
  title: string;
  message: string;
  onRetry: () => void;
  compact?: boolean;
}) {
  return (
    <section
      className={`state-panel error-state${compact ? " state-panel--error" : ""}`}
      role="alert"
    >
      <span className="state-icon" aria-hidden="true">!</span>
      <p className="eyebrow">Unified comparison API error</p>
      <h1>{title}</h1>
      <p>{message}</p>
      <p className="state-help">No fixture or recomputed result has been substituted.</p>
      <button className="button button-primary" type="button" onClick={onRetry}>
        Retry request
      </button>
    </section>
  );
}
