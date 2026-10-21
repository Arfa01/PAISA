import type { CoreRunData } from "../api/types";
import {
  formatDate,
  formatInteger,
  formatNumber,
  formatPercent,
  formatPkr,
  humanize,
} from "../utils/format";

interface OverviewViewProps {
  data: CoreRunData;
}

export function OverviewView({ data }: OverviewViewProps) {
  const { detail, summary } = data;
  const splitCounts = data.models.rows[0];
  const assumptions = detail.shared_experiment_assumptions;
  const bestValidation = summary.best_validation_prediction_model;
  const bestPortfolio = summary.best_test_period_simulated_portfolio;
  const lowestDrawdown = summary.lowest_drawdown_portfolio;
  const buyAndHold = summary.buy_and_hold_result;

  return (
    <section className="view overview-view" aria-labelledby="overview-title">
      <header className="view-header">
        <div>
          <p className="eyebrow">Completed vertical slice</p>
          <h1 id="overview-title">PAISA — Predictive AI Stock Assessment</h1>
          <p className="view-description">
            A read-only view of the selected OGDC model evaluation and its
            independently simulated portfolios.
          </p>
        </div>
        <span className="status-badge" aria-label={`Run status: ${detail.status}`}>
          {detail.status}
        </span>
      </header>

      <aside className="disclaimer" aria-label="Academic simulation disclaimer">
        {summary.disclaimer}
      </aside>

      <section className="panel" aria-labelledby="run-context-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Run context</p>
            <h2 id="run-context-title">Selected result set</h2>
          </div>
        </div>
        <dl className="metadata-grid">
          <div>
            <dt>Selected run ID</dt>
            <dd className="identifier">{detail.run_id}</dd>
          </div>
          <div>
            <dt>Symbol</dt>
            <dd>{detail.symbol}</dd>
          </div>
          <div>
            <dt>Run status</dt>
            <dd>{detail.status}</dd>
          </div>
          <div>
            <dt>Data source mode</dt>
            <dd>{humanize(detail.source_mode)}</dd>
          </div>
          <div>
            <dt>OGDC data date range</dt>
            <dd>
              <time dateTime={detail.data_date_range.start}>
                {formatDate(detail.data_date_range.start)}
              </time>{" "}
              –{" "}
              <time dateTime={detail.data_date_range.end}>
                {formatDate(detail.data_date_range.end)}
              </time>
            </dd>
          </div>
          <div>
            <dt>Completed</dt>
            <dd>
              <time dateTime={summary.completed_at}>
                {formatDate(summary.completed_at)}
              </time>
            </dd>
          </div>
          <div>
            <dt>Test feature period</dt>
            <dd>
              <time dateTime={detail.evaluation_dates.feature_start}>
                {formatDate(detail.evaluation_dates.feature_start)}
              </time>{" "}
              –{" "}
              <time dateTime={detail.evaluation_dates.feature_end}>
                {formatDate(detail.evaluation_dates.feature_end)}
              </time>
            </dd>
          </div>
          <div>
            <dt>Test target period</dt>
            <dd>
              <time dateTime={detail.evaluation_dates.target_start}>
                {formatDate(detail.evaluation_dates.target_start)}
              </time>{" "}
              –{" "}
              <time dateTime={detail.evaluation_dates.target_end}>
                {formatDate(detail.evaluation_dates.target_end)}
              </time>
            </dd>
          </div>
        </dl>

        <div className="card-grid count-grid" aria-label="Chronological split counts">
          <article className="metric-card">
            <p>Training observations</p>
            <strong>{formatInteger(splitCounts?.train_count)}</strong>
          </article>
          <article className="metric-card">
            <p>Validation observations</p>
            <strong>{formatInteger(splitCounts?.validation_count)}</strong>
          </article>
          <article className="metric-card">
            <p>Test observations</p>
            <strong>{formatInteger(splitCounts?.test_count)}</strong>
          </article>
        </div>
        <p className="supporting-copy">
          These chronological split counts are shared by the model variants in
          this run. The selected test window remains separate from model
          selection.
        </p>
      </section>

      <section className="panel" aria-labelledby="results-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Separate evaluation criteria</p>
            <h2 id="results-title">Headline results</h2>
          </div>
          <p>
            Prediction selection, simulated portfolio performance, and risk
            are reported independently. Test accuracy does not select the
            validation champion.
          </p>
        </div>

        <div className="card-grid result-grid">
          <article className="metric-card result-card">
            <p className="card-label">Best validation prediction model</p>
            <h3>{bestValidation.model_id}</h3>
            <p>{humanize(bestValidation.algorithm)}</p>
            <dl className="compact-definition-list">
              <div>
                <dt>Validation balanced accuracy</dt>
                <dd>{formatPercent(bestValidation.validation_balanced_accuracy)}</dd>
              </div>
              <div>
                <dt>Validation F1</dt>
                <dd>{formatPercent(bestValidation.validation_f1)}</dd>
              </div>
              <div>
                <dt>Selection basis</dt>
                <dd>{bestValidation.selection_basis}</dd>
              </div>
            </dl>
          </article>

          <article className="metric-card result-card">
            <p className="card-label">Best simulated model portfolio</p>
            <h3>{bestPortfolio.model_id}</h3>
            <dl className="compact-definition-list">
              <div>
                <dt>Ending value</dt>
                <dd>{formatPkr(bestPortfolio.ending_portfolio_value)}</dd>
              </div>
              <div>
                <dt>Total return</dt>
                <dd>{formatPercent(bestPortfolio.total_return)}</dd>
              </div>
              <div>
                <dt>Sharpe ratio</dt>
                <dd>{formatNumber(bestPortfolio.sharpe_ratio)}</dd>
              </div>
              <div>
                <dt>Selection basis</dt>
                <dd>{bestPortfolio.selection_basis}</dd>
              </div>
            </dl>
          </article>

          <article className="metric-card result-card">
            <p className="card-label">Lowest-drawdown model portfolio</p>
            <h3>{lowestDrawdown.model_id}</h3>
            <dl className="compact-definition-list">
              <div>
                <dt>Maximum drawdown</dt>
                <dd>{formatPercent(lowestDrawdown.maximum_drawdown)}</dd>
              </div>
              <div>
                <dt>Total return</dt>
                <dd>{formatPercent(lowestDrawdown.total_return)}</dd>
              </div>
            </dl>
          </article>

          <article className="metric-card result-card">
            <p className="card-label">Buy-and-hold result</p>
            <h3>{formatPkr(buyAndHold.ending_portfolio_value)}</h3>
            <dl className="compact-definition-list">
              <div>
                <dt>Total return</dt>
                <dd>{formatPercent(buyAndHold.total_return)}</dd>
              </div>
              <div>
                <dt>Maximum drawdown</dt>
                <dd>{formatPercent(buyAndHold.maximum_drawdown)}</dd>
              </div>
              <div>
                <dt>Fees paid</dt>
                <dd>{formatPkr(buyAndHold.fees_paid)}</dd>
              </div>
            </dl>
          </article>

          <article className="metric-card result-card result-card--wide">
            <p className="card-label">Any model beat buy-and-hold?</p>
            <h3>
              {summary.any_model_portfolio_beat_buy_and_hold ? "Yes" : "No"}
            </h3>
            <p>
              {summary.any_model_portfolio_beat_buy_and_hold
                ? "At least one model portfolio exceeded the controlled buy-and-hold total return over the common test window."
                : "No model portfolio exceeded the controlled buy-and-hold total return over the common test window."}
            </p>
          </article>
        </div>
      </section>

      <section className="panel" aria-labelledby="interpretation-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Read the results carefully</p>
            <h2 id="interpretation-title">Interpretation</h2>
          </div>
        </div>
        <div className="card-grid interpretation-grid">
          <article className="info-card">
            <h3>Selection policy</h3>
            <p>{summary.selection_policy}</p>
          </article>
          <article className="info-card">
            <h3>Prediction target</h3>
            <p>{summary.target_interpretation}</p>
          </article>
          <article className="info-card">
            <h3>Portfolio return</h3>
            <p>{summary.portfolio_return_interpretation}</p>
          </article>
        </div>
      </section>

      <section className="panel" aria-labelledby="assumptions-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Controlled simulation</p>
            <h2 id="assumptions-title">Shared experiment assumptions</h2>
          </div>
        </div>
        <dl className="metadata-grid assumptions-grid">
          <div>
            <dt>Initial cash per portfolio</dt>
            <dd>{formatPkr(assumptions.initial_cash_pkr)}</dd>
          </div>
          <div>
            <dt>Buy threshold</dt>
            <dd>Probability up ≥ {formatPercent(assumptions.buy_probability_up_gte)}</dd>
          </div>
          <div>
            <dt>Sell threshold</dt>
            <dd>Probability up ≤ {formatPercent(assumptions.sell_probability_up_lte)}</dd>
          </div>
          <div>
            <dt>Maximum allocation</dt>
            <dd>{formatPercent(assumptions.maximum_allocation)}</dd>
          </div>
          <div>
            <dt>Position and quantity</dt>
            <dd>
              {assumptions.long_only ? "Long only" : "Not long only"};{" "}
              {assumptions.whole_shares ? "whole shares" : "fractional shares"}
            </dd>
          </div>
          <div>
            <dt>Transaction cost</dt>
            <dd>{formatNumber(assumptions.transaction_cost_bps, 0)} bps</dd>
          </div>
          <div>
            <dt>Slippage</dt>
            <dd>{formatNumber(assumptions.slippage_bps, 0)} bps</dd>
          </div>
          <div>
            <dt>Signal timing</dt>
            <dd>{humanize(assumptions.signal_timing)}</dd>
          </div>
          <div>
            <dt>Execution</dt>
            <dd>{humanize(assumptions.execution)}</dd>
          </div>
          <div>
            <dt>Valuation</dt>
            <dd>{humanize(assumptions.valuation)}</dd>
          </div>
          <div>
            <dt>Ending position</dt>
            <dd>{humanize(assumptions.ending_position)}</dd>
          </div>
        </dl>
      </section>

      <div className="notice-grid">
        <section className="panel notice-panel" aria-labelledby="warnings-title">
          <h2 id="warnings-title">Warnings</h2>
          {summary.warnings.length > 0 ? (
            <ul className="notice-list">
              {summary.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : (
            <p>No warnings were reported by the API.</p>
          )}
        </section>

        <section className="panel notice-panel" aria-labelledby="limitations-title">
          <h2 id="limitations-title">Limitations</h2>
          {summary.limitations.length > 0 ? (
            <ul className="notice-list">
              {summary.limitations.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
          ) : (
            <p>No additional limitations were reported by the API.</p>
          )}
        </section>
      </div>
    </section>
  );
}
