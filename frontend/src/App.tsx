import { useCallback, useEffect, useMemo, useState } from "react";

import {
  apiClient as defaultApiClient,
  apiErrorMessage,
  type PaisaApiClient,
} from "./api/client";
import type {
  CoreRunData,
  DrawdownRow,
  EquityCurveRow,
  VerticalSliceListResponse,
} from "./api/types";
import { ExperimentView } from "./components/ExperimentView";
import { HistoricalView } from "./components/HistoricalView";
import { ModelComparisonView } from "./components/ModelComparisonView";
import { OverviewView } from "./components/OverviewView";
import { PortfolioComparisonView } from "./components/PortfolioComparisonView";
import { ProvenanceView } from "./components/ProvenanceView";

const DEFAULT_DISCLAIMER = "Academic simulation; not financial advice.";

type ViewId =
  | "overview"
  | "history"
  | "models"
  | "portfolios"
  | "experiment"
  | "provenance";

const NAVIGATION: Array<{ id: ViewId; label: string; short: string }> = [
  { id: "overview", label: "Overview", short: "01" },
  { id: "history", label: "Historical OGDC", short: "02" },
  { id: "models", label: "Model comparison", short: "03" },
  { id: "portfolios", label: "Portfolio comparison", short: "04" },
  { id: "experiment", label: "Experiment detail", short: "05" },
  { id: "provenance", label: "Data & limitations", short: "06" },
];

interface AppProps {
  client?: PaisaApiClient;
}

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

const initialAsyncState = <T,>(): AsyncState<T> => ({
  data: null,
  loading: true,
  error: null,
});

export function App({ client = defaultApiClient }: AppProps) {
  const [activeView, setActiveView] = useState<ViewId>("overview");
  const [runs, setRuns] = useState<AsyncState<VerticalSliceListResponse>>(
    initialAsyncState,
  );
  const [selectedRunId, setSelectedRunId] = useState("");
  const [core, setCore] = useState<AsyncState<CoreRunData>>({
    data: null,
    loading: false,
    error: null,
  });
  const [reloadRuns, setReloadRuns] = useState(0);
  const [reloadCore, setReloadCore] = useState(0);
  const [portfolioSeries, setPortfolioSeries] = useState<{
    runId: string;
    equityRows: EquityCurveRow[];
    drawdownRows: DrawdownRow[];
    loading: boolean;
    error: string | null;
  }>({
    runId: "",
    equityRows: [],
    drawdownRows: [],
    loading: false,
    error: null,
  });
  const visibleCore =
    core.data?.detail.run_id === selectedRunId ? core.data : null;

  useEffect(() => {
    const controller = new AbortController();
    setRuns({ data: null, loading: true, error: null });
    client
      .listRuns(controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return;
        setRuns({ data: response, loading: false, error: null });
        setSelectedRunId((current) => {
          if (response.rows.some((run) => run.run_id === current)) return current;
          return response.rows[0]?.run_id ?? "";
        });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setRuns({ data: null, loading: false, error: apiErrorMessage(error) });
      });
    return () => controller.abort();
  }, [client, reloadRuns]);

  useEffect(() => {
    if (!selectedRunId) {
      setCore({ data: null, loading: false, error: null });
      return;
    }
    const controller = new AbortController();
    setCore({ data: null, loading: true, error: null });
    setPortfolioSeries({
      runId: "",
      equityRows: [],
      drawdownRows: [],
      loading: false,
      error: null,
    });
    client
      .getCoreRunData(selectedRunId, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) {
          setCore({ data: response, loading: false, error: null });
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setCore({ data: null, loading: false, error: apiErrorMessage(error) });
        }
      });
    return () => controller.abort();
  }, [client, reloadCore, selectedRunId]);

  useEffect(() => {
    if (activeView !== "portfolios" || !selectedRunId || !visibleCore) return;
    const controller = new AbortController();
    setPortfolioSeries({
      runId: selectedRunId,
      equityRows: [],
      drawdownRows: [],
      loading: true,
      error: null,
    });
    Promise.all([
      client.getAllEquityCurves(selectedRunId, controller.signal),
      client.getAllDrawdowns(selectedRunId, controller.signal),
    ])
      .then(([equityRows, drawdownRows]) => {
        if (!controller.signal.aborted) {
          setPortfolioSeries({
            runId: selectedRunId,
            equityRows,
            drawdownRows,
            loading: false,
            error: null,
          });
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setPortfolioSeries({
            runId: selectedRunId,
            equityRows: [],
            drawdownRows: [],
            loading: false,
            error: apiErrorMessage(error),
          });
        }
      });
    return () => controller.abort();
  }, [activeView, client, selectedRunId, visibleCore]);

  const retryRuns = useCallback(() => setReloadRuns((value) => value + 1), []);
  const retryCore = useCallback(() => setReloadCore((value) => value + 1), []);

  const disclaimer =
    visibleCore?.detail.disclaimer ?? runs.data?.disclaimer ?? DEFAULT_DISCLAIMER;
  const selectedRun = useMemo(
    () => runs.data?.rows.find((run) => run.run_id === selectedRunId),
    [runs.data, selectedRunId],
  );

  function renderContent() {
    if (runs.loading) {
      return <LoadingPanel label="Loading completed PAISA runs…" />;
    }
    if (runs.error) {
      return (
        <ErrorPanel
          title="Could not load PAISA runs"
          message={runs.error}
          onRetry={retryRuns}
        />
      );
    }
    if (!runs.data || runs.data.rows.length === 0) {
      return (
        <section className="state-panel empty-state" role="status">
          <span className="state-icon" aria-hidden="true">○</span>
          <p className="eyebrow">No completed runs</p>
          <h1>No immutable vertical-slice results are available.</h1>
          <p>
            Start the backend with a completed, checksum-valid PAISA run. The dashboard
            will not substitute fixture or fabricated results.
          </p>
          <button className="button button-primary" type="button" onClick={retryRuns}>
            Check again
          </button>
        </section>
      );
    }
    if (core.loading) return <LoadingPanel label={`Loading ${selectedRunId}…`} />;
    if (core.error) {
      return (
        <ErrorPanel
          title="This run could not be loaded"
          message={core.error}
          onRetry={retryCore}
        />
      );
    }
    if (!visibleCore) return null;

    switch (activeView) {
      case "history":
        return (
          <HistoricalView
            client={client}
            detail={visibleCore.detail}
            summary={visibleCore.summary}
          />
        );
      case "models":
        return (
          <ModelComparisonView models={visibleCore.models} summary={visibleCore.summary} />
        );
      case "portfolios":
        return (
          <PortfolioComparisonView
            portfolios={visibleCore.portfolios}
            summary={visibleCore.summary}
            equityRows={portfolioSeries.equityRows}
            drawdownRows={portfolioSeries.drawdownRows}
            loading={portfolioSeries.loading}
            error={portfolioSeries.error}
          />
        );
      case "experiment":
        return (
          <ExperimentView
            client={client}
            runId={selectedRunId}
            experiments={visibleCore.experiments}
          />
        );
      case "provenance":
        return <ProvenanceView source={visibleCore.source} />;
      case "overview":
      default:
        return <OverviewView data={visibleCore} />;
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <p className="brand-name">PAISA</p>
            <p className="brand-subtitle">Predictive AI Stock Assessment</p>
          </div>
        </div>
        <div className="run-control">
          <label htmlFor="run-select">Analysis run</label>
          <select
            id="run-select"
            value={selectedRunId}
            disabled={!runs.data?.rows.length}
            onChange={(event) => {
              const nextRunId = event.target.value;
              setSelectedRunId(nextRunId);
              setCore({ data: null, loading: Boolean(nextRunId), error: null });
              setPortfolioSeries({
                runId: "",
                equityRows: [],
                drawdownRows: [],
                loading: false,
                error: null,
              });
              setActiveView("overview");
            }}
          >
            {runs.data?.rows.map((run) => (
              <option value={run.run_id} key={run.run_id}>
                {run.run_id}
              </option>
            ))}
          </select>
          {selectedRun ? (
            <span className="run-status">
              <span className="status-dot" aria-hidden="true" />
              {selectedRun.status} · {selectedRun.symbol}
            </span>
          ) : null}
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar" aria-label="Dashboard sections">
          <nav>
            {NAVIGATION.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`nav-item${activeView === item.id ? " active" : ""}`}
                aria-current={activeView === item.id ? "page" : undefined}
                onClick={() => setActiveView(item.id)}
                disabled={!visibleCore}
              >
                <span>{item.short}</span>
                {item.label}
              </button>
            ))}
          </nav>
          <div className="sidebar-meta">
            <p>Read-only results</p>
            <span>Daily PSX · OGDC</span>
          </div>
        </aside>

        <main className="main-content" id="main-content">
          {renderContent()}
        </main>
      </div>

      <footer className="disclaimer-bar" role="note" aria-label="Simulation disclaimer">
        <span className="disclaimer-icon" aria-hidden="true">i</span>
        <strong>Academic simulation</strong>
        <span>{disclaimer}</span>
      </footer>
    </div>
  );
}

function LoadingPanel({ label }: { label: string }) {
  return (
    <section className="state-panel loading-state" role="status" aria-live="polite">
      <div className="loading-orbit" aria-hidden="true">
        <span />
      </div>
      <p className="eyebrow">PAISA results API</p>
      <h1>{label}</h1>
      <div className="skeleton-lines" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
    </section>
  );
}

function ErrorPanel({
  title,
  message,
  onRetry,
}: {
  title: string;
  message: string;
  onRetry: () => void;
}) {
  return (
    <section className="state-panel error-state" role="alert">
      <span className="state-icon" aria-hidden="true">!</span>
      <p className="eyebrow">API error</p>
      <h1>{title}</h1>
      <p>{message}</p>
      <p className="state-help">No fixture data has been substituted.</p>
      <button className="button button-primary" type="button" onClick={onRetry}>
        Retry request
      </button>
    </section>
  );
}
