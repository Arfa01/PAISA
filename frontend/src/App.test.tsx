import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { ApiError, PaisaApiClient } from "./api/client";
import {
  DISCLAIMER,
  OTHER_RUN_ID,
  RUN_ID,
  makeCoreRunData,
  makeMockClient,
  makeRunList,
  makeRunListItem,
} from "./test/fixtures";

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, reject, resolve };
}

describe("PAISA dashboard shell", () => {
  it("renders a completed run overview and keeps the disclaimer visible", async () => {
    const client = makeMockClient();
    render(<App client={client} />);

    expect(
      await screen.findByRole("heading", {
        name: "PAISA — Predictive AI Stock Assessment",
      }),
    ).toBeInTheDocument();
    const main = screen.getByRole("main");
    expect(within(main).getByText(RUN_ID)).toBeInTheDocument();
    expect(within(main).getByText("Best validation prediction model")).toBeInTheDocument();
    expect(within(main).getByText("Best simulated model portfolio")).toBeInTheDocument();
    expect(within(main).getByText("Lowest-drawdown model portfolio")).toBeInTheDocument();
    expect(within(main).getByText("Buy-and-hold result")).toBeInTheDocument();

    const disclaimer = screen.getByRole("note", {
      name: "Simulation disclaimer",
    });
    expect(disclaimer).toHaveTextContent(DISCLAIMER);
    expect(disclaimer).toBeVisible();
  });

  it("shows a loading state while the run inventory is pending", () => {
    const client = makeMockClient();
    vi.mocked(client.listRuns).mockReturnValue(new Promise(() => undefined));

    render(<App client={client} />);

    expect(screen.getByRole("status")).toHaveTextContent(
      "Loading completed PAISA runs",
    );
  });

  it("shows an API failure without silently substituting fixture results", async () => {
    const client = makeMockClient();
    vi.mocked(client.listRuns).mockRejectedValue(
      new ApiError("Could not reach the test API.", "network"),
    );

    render(<App client={client} />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Could not load PAISA runs");
    expect(alert).toHaveTextContent("Could not reach the test API.");
    expect(alert).toHaveTextContent("No fixture data has been substituted.");
    expect(screen.queryByText("M-001")).not.toBeInTheDocument();
    expect(screen.queryByText("PKR 101,250.75")).not.toBeInTheDocument();
  });

  it("shows an explicit empty-run state and does not fabricate a result", async () => {
    const client = makeMockClient({ runList: makeRunList([]) });

    render(<App client={client} />);

    await screen.findByRole("heading", {
      name: "No immutable vertical-slice results are available.",
    });
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent(
      "No immutable vertical-slice results are available.",
    );
    expect(status).toHaveTextContent(
      "will not substitute fixture or fabricated results",
    );
    expect(client.getCoreRunData).not.toHaveBeenCalled();
  });

  it("loads the newly selected run and returns to its overview", async () => {
    const user = userEvent.setup();
    const runList = makeRunList([
      makeRunListItem(RUN_ID),
      makeRunListItem(OTHER_RUN_ID),
    ]);
    const client = makeMockClient({ runList });
    vi.mocked(client.getCoreRunData).mockImplementation(async (runId) =>
      makeCoreRunData(runId),
    );
    render(<App client={client} />);

    await screen.findByRole("heading", {
      name: "PAISA — Predictive AI Stock Assessment",
    });
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Analysis run" }),
      OTHER_RUN_ID,
    );

    await waitFor(() =>
      expect(client.getCoreRunData).toHaveBeenLastCalledWith(
        OTHER_RUN_ID,
        expect.any(AbortSignal),
      ),
    );
    const main = screen.getByRole("main");
    expect(await within(main).findByText(OTHER_RUN_ID)).toBeInTheDocument();
    expect(
      within(main).getByRole("heading", {
        name: "PAISA — Predictive AI Stock Assessment",
      }),
    ).toBeInTheDocument();
  });

  it("clears the previous run while a newly selected run is still loading", async () => {
    const user = userEvent.setup();
    const nextRun = deferred<ReturnType<typeof makeCoreRunData>>();
    const runList = makeRunList([
      makeRunListItem(RUN_ID),
      makeRunListItem(OTHER_RUN_ID),
    ]);
    const client = makeMockClient({ runList });
    vi.mocked(client.getCoreRunData).mockImplementation((runId) =>
      runId === OTHER_RUN_ID
        ? nextRun.promise
        : Promise.resolve(makeCoreRunData(runId)),
    );
    render(<App client={client} />);

    const main = screen.getByRole("main");
    expect(await within(main).findByText(RUN_ID)).toBeInTheDocument();

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Analysis run" }),
      OTHER_RUN_ID,
    );

    expect(
      await within(main).findByText(`Loading ${OTHER_RUN_ID}…`),
    ).toBeInTheDocument();
    expect(within(main).queryByText(RUN_ID)).not.toBeInTheDocument();
    expect(
      within(main).queryByText("Best validation prediction model"),
    ).not.toBeInTheDocument();

    nextRun.resolve(makeCoreRunData(OTHER_RUN_ID));
    expect(await within(main).findByText(OTHER_RUN_ID)).toBeInTheDocument();
  });

  it("loads portfolio curves after navigation and renders both chart series sets", async () => {
    const user = userEvent.setup();
    const client = makeMockClient();
    render(<App client={client} />);

    await screen.findByRole("heading", {
      name: "PAISA — Predictive AI Stock Assessment",
    });
    await user.click(
      screen.getByRole("button", { name: /Portfolio comparison/ }),
    );

    expect(
      await screen.findByRole("img", {
        name: /Line chart of total equity in PKR/,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", {
        name: /Line chart of drawdown percentages/,
      }),
    ).toBeInTheDocument();
    expect(client.getAllEquityCurves).toHaveBeenCalledWith(
      RUN_ID,
      expect.any(AbortSignal),
    );
    expect(client.getAllDrawdowns).toHaveBeenCalledWith(
      RUN_ID,
      expect.any(AbortSignal),
    );
    expect(screen.queryByText(/Loading portfolio equity/)).not.toBeInTheDocument();

    const charts = screen.getAllByTestId("plotly-chart");
    expect(charts).toHaveLength(2);
    expect(charts.every((chart) => Number(chart.dataset.traceCount) > 0)).toBe(true);
  });

  it("renders historical price and volume charts from the API response", async () => {
    const user = userEvent.setup();
    const client = makeMockClient();
    render(<App client={client} />);

    await screen.findByRole("heading", {
      name: "PAISA — Predictive AI Stock Assessment",
    });
    await user.click(screen.getByRole("button", { name: /Historical OGDC/ }));

    expect(
      await screen.findByRole("img", {
        name: "Line chart of OGDC daily closing prices",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", {
        name: "Bar chart of OGDC daily trading volume",
      }),
    ).toBeInTheDocument();
    expect(client.getHistoricalPrices).toHaveBeenCalledWith(
      expect.any(AbortSignal),
    );
    expect(screen.getByText("Observations returned").parentElement).toHaveTextContent(
      "1",
    );
  });

  it("shows the explicit no-fixture state when historical prices fail", async () => {
    const user = userEvent.setup();
    const client = makeMockClient();
    vi.mocked(client.getHistoricalPrices).mockRejectedValue(
      new ApiError("The requested PAISA result was not found.", "http", 404),
    );
    render(<App client={client} />);

    await screen.findByRole("heading", {
      name: "PAISA — Predictive AI Stock Assessment",
    });
    await user.click(screen.getByRole("button", { name: /Historical OGDC/ }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Historical price API unavailable");
    expect(alert).toHaveTextContent("The requested PAISA result was not found.");
    expect(alert).toHaveTextContent(
      "No fixture or synthetic price series has been substituted.",
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("rejects an API payload containing an absolute local path before rendering it", async () => {
    const unsafePath = "/Users/private-owner/secret-run.json";
    const payload = makeRunList();
    payload.disclaimer = `Internal artifact: ${unsafePath}`;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue(payload),
      } as unknown as Response),
    );
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    const { container } = render(<App client={client} />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "The PAISA API response contained prohibited local-path data.",
    );
    expect(container).not.toHaveTextContent(unsafePath);
    expect(screen.queryByText(RUN_ID)).not.toBeInTheDocument();
  });
});
