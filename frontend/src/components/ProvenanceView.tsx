import type { SourceResponse } from "../api/types";
import { formatDate, formatInteger, humanize } from "../utils/format";

interface ProvenanceViewProps {
  source: SourceResponse;
}

function uniqueMessages(messages: string[]): string[] {
  return Array.from(new Set(messages.filter((message) => message.trim().length > 0)));
}

export function ProvenanceView({ source }: ProvenanceViewProps) {
  const details = source.source;
  const migration = source.migration_evidence;
  const warnings = uniqueMessages(source.warnings);
  const limitations = uniqueMessages([
    ...source.limitations,
    ...(migration?.limitations ?? []),
  ]);
  const interpretationMessages = [...warnings, ...limitations];
  const corporateActionAndDividendNotice = interpretationMessages.find((message) =>
    /corporate.action|dividend|price adjustment|adjusted/i.test(message),
  );
  const targetExecutionNotice = interpretationMessages.find((message) =>
    /close.to.close|next.session open|prediction target|portfolio returns begin/i.test(
      message,
    ),
  );
  const isVerifiedLegacyCache =
    details.acquisition_mode === "verified_legacy_psx_cache";
  const provider = source.pipeline_evidence?.provider ?? details.original_provider;
  const migrationStrategy =
    migration?.migration_strategy ?? details.migration_strategy;
  const provenanceStrength =
    migration?.provenance_strength ?? details.provenance_strength;

  return (
    <section className="view provenance-view" aria-labelledby="provenance-title">
      <header className="view-header">
        <div>
          <p className="eyebrow">Audit context</p>
          <h1 id="provenance-title">Data provenance and limitations</h1>
          <p className="view-description">
            Public, non-sensitive evidence describing where this run’s OGDC
            observations came from and how they should be interpreted.
          </p>
        </div>
      </header>

      <section className="panel" aria-labelledby="source-summary-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Source summary</p>
            <h2 id="source-summary-title">Acquisition evidence</h2>
          </div>
        </div>
        <dl className="metadata-grid provenance-grid">
          <div>
            <dt>Selected run ID</dt>
            <dd className="identifier">{source.run_id}</dd>
          </div>
          <div>
            <dt>Acquisition mode</dt>
            <dd>{humanize(details.acquisition_mode)}</dd>
          </div>
          <div>
            <dt>Source label</dt>
            <dd>{humanize(details.source)}</dd>
          </div>
          <div>
            <dt>Original provider</dt>
            <dd>{humanize(details.original_provider)}</dd>
          </div>
          <div>
            <dt>Pipeline provider</dt>
            <dd>{humanize(provider)}</dd>
          </div>
          <div>
            <dt>Real market data</dt>
            <dd>{details.is_real_data ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Live fetch succeeded</dt>
            <dd>{details.live_fetch_succeeded ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Live acquisition error recorded</dt>
            <dd>{details.live_acquisition_error_present ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Verified legacy cache</dt>
            <dd>{isVerifiedLegacyCache ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Legacy metadata preserved</dt>
            <dd>{details.legacy_metadata_preserved ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Rows</dt>
            <dd>{formatInteger(details.row_count)}</dd>
          </div>
          <div>
            <dt>OGDC date coverage</dt>
            <dd>
              <time dateTime={source.data_date_range.start}>
                {formatDate(source.data_date_range.start)}
              </time>{" "}
              –{" "}
              <time dateTime={source.data_date_range.end}>
                {formatDate(source.data_date_range.end)}
              </time>
            </dd>
          </div>
        </dl>
      </section>

      <section className="panel" aria-labelledby="migration-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Evidence chain</p>
            <h2 id="migration-title">Cache and migration status</h2>
          </div>
        </div>
        <dl className="metadata-grid provenance-grid">
          <div>
            <dt>Migration status</dt>
            <dd>{humanize(migration?.status)}</dd>
          </div>
          <div>
            <dt>Migration strategy</dt>
            <dd>{humanize(migrationStrategy)}</dd>
          </div>
          <div>
            <dt>Provenance strength</dt>
            <dd>{humanize(provenanceStrength)}</dd>
          </div>
          <div>
            <dt>Requested symbols</dt>
            <dd>
              {source.pipeline_evidence?.symbols_requested.length
                ? source.pipeline_evidence.symbols_requested.join(", ")
                : "N/A"}
            </dd>
          </div>
          <div>
            <dt>Pipeline evidence generated</dt>
            <dd>
              {source.pipeline_evidence?.generated_at ? (
                <time dateTime={source.pipeline_evidence.generated_at}>
                  {formatDate(source.pipeline_evidence.generated_at)}
                </time>
              ) : (
                "N/A"
              )}
            </dd>
          </div>
          <div>
            <dt>Pipeline schema</dt>
            <dd>{source.pipeline_evidence?.schema_version ?? "N/A"}</dd>
          </div>
        </dl>
        <p className="supporting-copy">
          Integrity evidence can show whether recorded files changed; it is not
          an independent provider signature or a guarantee of future market
          outcomes.
        </p>
      </section>

      <section className="panel" aria-labelledby="interpretation-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Required context</p>
            <h2 id="interpretation-title">Interpretation limits</h2>
          </div>
        </div>
        <div className="card-grid interpretation-grid">
          <article className="info-card">
            <h3>Corporate actions and dividends</h3>
            <p>{corporateActionAndDividendNotice ?? "N/A — not reported by the API."}</p>
          </article>
          <article className="info-card">
            <h3>Prediction target and execution</h3>
            <p>{targetExecutionNotice ?? "N/A — not reported by the API."}</p>
          </article>
        </div>
      </section>

      <div className="notice-grid">
        <section className="panel notice-panel" aria-labelledby="source-warnings-title">
          <h2 id="source-warnings-title">Warnings</h2>
          {warnings.length > 0 ? (
            <ul className="notice-list">
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : (
            <p>No warnings were reported by the API.</p>
          )}
        </section>

        <section
          className="panel notice-panel"
          aria-labelledby="source-limitations-title"
        >
          <h2 id="source-limitations-title">Limitations</h2>
          {limitations.length > 0 ? (
            <ul className="notice-list">
              {limitations.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
          ) : (
            <p>No additional limitations were reported by the API.</p>
          )}
        </section>
      </div>

      <aside className="disclaimer" aria-label="Academic simulation disclaimer">
        {source.disclaimer}
      </aside>
    </section>
  );
}
