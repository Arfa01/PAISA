import type { Data } from "plotly.js";

import type {
  ModelMetricRow,
  ModelsResponse,
  VerticalSliceSummaryResponse,
} from "../api/types";
import { formatPercent, humanize } from "../utils/format";
import { Plot } from "./Plot";

export interface ModelComparisonViewProps {
  models: ModelsResponse;
  summary: VerticalSliceSummaryResponse;
}
type MetricAccessor = (row: ModelMetricRow) => number | null;

function metricPoints(rows: ModelMetricRow[], getValue: MetricAccessor) {
  return rows.flatMap((row) => {
    const value = getValue(row);
    return typeof value === "number" && Number.isFinite(value)
      ? [{ modelId: row.model_id, value }]
      : [];
  });
}

function metricTrace(
  rows: ModelMetricRow[],
  name: string,
  color: string,
  getValue: MetricAccessor,
): Data {
  const points = metricPoints(rows, getValue);
  return {
    type: "bar",
    name,
    x: points.map((point) => point.modelId),
    y: points.map((point) => point.value),
    marker: { color },
    text: points.map((point) => formatPercent(point.value)),
    textposition: "auto",
    hovertemplate: `%{x}<br>${name}: %{y:.2%}<extra></extra>`,
  };
}

export function ModelComparisonView({
  models,
  summary,
}: ModelComparisonViewProps) {
  const validationModel = summary.best_validation_prediction_model;
  const portfolioModel = summary.best_test_period_simulated_portfolio;
  const validationData: Data[] = [
    metricTrace(
      models.rows,
      "Balanced accuracy",
      "#285b8f",
      (row) => row.validation_balanced_accuracy,
    ),
    metricTrace(
      models.rows,
      "F1 score",
      "#2f8877",
      (row) => row.validation_f1,
    ),
  ];
  const testData: Data[] = [
    metricTrace(
      models.rows,
      "Balanced accuracy",
      "#7c5cbf",
      (row) => row.test_balanced_accuracy,
    ),
    metricTrace(models.rows, "F1 score", "#c2762b", (row) => row.test_f1),
  ];

  return (
    <section aria-labelledby="model-comparison-title" className="view-section">
      <header className="view-heading">
        <div>
          <p className="eyebrow">Prediction evaluation</p>
          <h2 id="model-comparison-title">Model comparison</h2>
        </div>
        <p>
          Validation metrics govern model selection; test metrics report the
          frozen models&apos; held-out performance.
        </p>
      </header>

      <div className="selection-summary">
        <article className="summary-card summary-card--accent">
          <span className="summary-card__label">
            Validation-selected prediction model
          </span>
          <strong>{validationModel.model_id}</strong>
          <span>{humanize(validationModel.algorithm)}</span>
          <small>
            Validation balanced accuracy {" "}
            {formatPercent(validationModel.validation_balanced_accuracy)}
          </small>
        </article>
        <article className="summary-card">
          <span className="summary-card__label">
            Best simulated model portfolio
          </span>
          <strong>{portfolioModel.model_id}</strong>
          <span>Ranked by test-period portfolio outcome</span>
          <small>Total return {formatPercent(portfolioModel.total_return)}</small>
        </article>
      </div>

      <p className="criteria-note">
        <strong>Different criteria:</strong> {validationModel.model_id} was
        selected using validation prediction quality. {portfolioModel.model_id}
        {" "}produced the best simulated model-portfolio result during the test
        period. Test accuracy was not used to select the prediction model.
      </p>

      {models.rows.length === 0 ? (
        <div className="state-panel state-panel--empty" role="status">
          No model results are available for this run.
        </div>
      ) : (
        <>
          <div className="table-wrap">
            <table>
              <caption className="sr-only">
                Validation and test prediction metrics by model
              </caption>
              <thead>
                <tr>
                  <th scope="col">Model ID</th>
                  <th scope="col">Algorithm</th>
                  <th scope="col">Validation balanced accuracy</th>
                  <th scope="col">Validation F1</th>
                  <th scope="col">Test accuracy</th>
                  <th scope="col">Test balanced accuracy</th>
                  <th scope="col">Test F1</th>
                </tr>
              </thead>
              <tbody>
                {models.rows.map((row) => (
                  <tr key={row.registered_model_id}>
                    <th scope="row">
                      <span className="table-primary">{row.model_id}</span>
                      {row.model_id === validationModel.model_id ? (
                        <span className="status-tag">Validation selected</span>
                      ) : null}
                    </th>
                    <td>{humanize(row.algorithm)}</td>
                    <td>{formatPercent(row.validation_balanced_accuracy)}</td>
                    <td>{formatPercent(row.validation_f1)}</td>
                    <td>{formatPercent(row.test_accuracy)}</td>
                    <td>{formatPercent(row.test_balanced_accuracy)}</td>
                    <td>{formatPercent(row.test_f1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="chart-grid">
            <Plot
              ariaLabel="Grouped bar chart comparing validation balanced accuracy and validation F1 for every model."
              data={validationData}
              layout={{
                barmode: "group",
                title: { text: "Validation prediction metrics" },
                xaxis: { title: { text: "Model" }, type: "category" },
                yaxis: {
                  range: [0, 1],
                  tickformat: ".0%",
                  title: { text: "Score" },
                },
              }}
            />
            <Plot
              ariaLabel="Grouped bar chart comparing test balanced accuracy and test F1 for every model."
              data={testData}
              layout={{
                barmode: "group",
                title: { text: "Held-out test prediction metrics" },
                xaxis: { title: { text: "Model" }, type: "category" },
                yaxis: {
                  range: [0, 1],
                  tickformat: ".0%",
                  title: { text: "Score" },
                },
              }}
            />
          </div>
        </>
      )}
    </section>
  );
}
