import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  drawdownRows,
  equityRows,
  makeCoreRunData,
} from "../test/fixtures";
import { ModelComparisonView } from "./ModelComparisonView";
import { PortfolioComparisonView } from "./PortfolioComparisonView";

describe("model comparison leaderboard", () => {
  it("renders every model metric and distinguishes the two ranking criteria", () => {
    const data = makeCoreRunData();
    render(<ModelComparisonView models={data.models} summary={data.summary} />);

    const table = screen.getByRole("table", {
      name: "Validation and test prediction metrics by model",
    });
    for (const modelId of ["M-001", "M-002", "M-003", "M-004"]) {
      expect(within(table).getByRole("row", { name: new RegExp(modelId) })).toBeInTheDocument();
    }
    expect(screen.getByText("Validation-selected prediction model")).toBeInTheDocument();
    expect(screen.getByText("Best simulated model portfolio")).toBeInTheDocument();
    expect(screen.getByText("Different criteria:").closest("p")).toHaveTextContent(
      "M-001 was selected using validation prediction quality",
    );
    expect(screen.getAllByTestId("plotly-chart")).toHaveLength(2);
  });
});
describe("portfolio comparison leaderboard", () => {
  it("renders model and benchmark portfolios, with null Sharpe shown as N/A", () => {
    const data = makeCoreRunData();
    render(
      <PortfolioComparisonView
        portfolios={data.portfolios}
        summary={data.summary}
        equityRows={[]}
        drawdownRows={[]}
        loading={false}
        error={null}
      />,
    );

    const table = screen.getByRole("table", {
      name: "Model portfolio and benchmark performance metrics",
    });
    for (const portfolio of [
      "M-001",
      "M-002",
      "M-003",
      "M-004",
      "CASH-ONLY",
      "BUY-AND-HOLD",
    ]) {
      expect(
        within(table).getByRole("row", { name: new RegExp(portfolio) }),
      ).toBeInTheDocument();
    }
    const cashRow = within(table).getByRole("row", { name: /CASH-ONLY/ });
    expect(within(cashRow).getByText("N/A")).toBeInTheDocument();
    expect(within(cashRow).getByText("PKR 100,000.00")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Trading activity and fee comparison"),
    ).toHaveTextContent(
      "M-004 has the highest model-portfolio trade count (38)",
    );
    expect(
      screen.getByLabelText("Trading activity and fee comparison"),
    ).toHaveTextContent(
      "M-004 has the highest reported model-portfolio fees (PKR 1,800.00)",
    );
  });

  it("derives the activity conclusion from API rows rather than a fixed model", () => {
    const data = makeCoreRunData();
    const first = data.portfolios.rows.find((row) => row.model_id === "M-001")!;
    first.number_of_trades = 99;
    first.fees_paid = "9999.00";

    render(
      <PortfolioComparisonView
        portfolios={data.portfolios}
        summary={data.summary}
        equityRows={[]}
        drawdownRows={[]}
        loading={false}
        error={null}
      />,
    );

    const conclusion = screen.getByLabelText("Trading activity and fee comparison");
    expect(conclusion).toHaveTextContent("M-001 has the highest model-portfolio trade count (99)");
    expect(conclusion).toHaveTextContent("M-001 has the highest reported model-portfolio fees");
  });

  it("never renders NaN or Infinity from invalid numeric props", () => {
    const data = makeCoreRunData();
    data.portfolios.rows[0]!.total_return = Number.NaN;
    data.portfolios.rows[0]!.volatility = Number.POSITIVE_INFINITY;
    data.portfolios.rows[0]!.sharpe_ratio = Number.NEGATIVE_INFINITY;
    data.portfolios.rows[0]!.ending_portfolio_value = "Infinity";

    const { container } = render(
      <PortfolioComparisonView
        portfolios={data.portfolios}
        summary={data.summary}
        equityRows={[]}
        drawdownRows={[]}
        loading={false}
        error={null}
      />,
    );

    expect(container).not.toHaveTextContent("NaN");
    expect(container).not.toHaveTextContent("Infinity");
    expect(screen.getAllByText("N/A").length).toBeGreaterThanOrEqual(4);
  });

  it("renders all four models and both benchmarks in each complete chart", () => {
    const data = makeCoreRunData();
    render(
      <PortfolioComparisonView
        portfolios={data.portfolios}
        summary={data.summary}
        equityRows={equityRows}
        drawdownRows={drawdownRows}
        loading={false}
        error={null}
      />,
    );

    const charts = screen.getAllByTestId("plotly-chart");
    expect(charts).toHaveLength(2);
    expect(charts[0]).toHaveAttribute("data-trace-count", "6");
    expect(charts[1]).toHaveAttribute("data-trace-count", "6");
  });

  it("rejects incomplete curve sets instead of presenting a partial comparison", () => {
    const data = makeCoreRunData();
    render(
      <PortfolioComparisonView
        portfolios={data.portfolios}
        summary={data.summary}
        equityRows={equityRows.slice(1)}
        drawdownRows={drawdownRows}
        loading={false}
        error={null}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent(
      "do not contain exactly one series for every portfolio and benchmark",
    );
    expect(screen.queryByTestId("plotly-chart")).not.toBeInTheDocument();
  });
});
