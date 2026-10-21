import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  EXPERIMENT_ONE,
  EXPERIMENT_TWO,
  RUN_ID,
  makeCoreRunData,
  makeMockClient,
  makeSnapshotsResponse,
} from "../test/fixtures";
import { ExperimentView } from "./ExperimentView";

function snapshotsPage(
  offset: number,
  experimentId = EXPERIMENT_ONE,
) {
  const response = makeSnapshotsResponse(experimentId, RUN_ID);
  const row = response.rows[0]!;
  return {
    ...response,
    total: 12,
    offset,
    rows: [
      {
        ...row,
        date: offset === 0 ? "2026-01-05" : "2026-03-02",
        portfolio_id: `portfolio-${experimentId}`,
      },
    ],
  };
}

describe("experiment detail view", () => {
  it("loads a newly selected experiment and replaces stale detail", async () => {
    const user = userEvent.setup();
    const data = makeCoreRunData();
    const client = makeMockClient({ core: data });
    render(
      <ExperimentView
        client={client}
        runId={RUN_ID}
        experiments={data.experiments}
      />,
    );

    expect(await screen.findByText("registered-m-001")).toBeInTheDocument();
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Experiment" }),
      EXPERIMENT_TWO,
    );

    expect(await screen.findByText("registered-m-002")).toBeInTheDocument();
    expect(screen.queryByText("registered-m-001")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(client.getExperiment).toHaveBeenLastCalledWith(
        RUN_ID,
        EXPERIMENT_TWO,
        expect.any(AbortSignal),
      ),
    );
  });

  it("requests and displays the next trade page safely", async () => {
    const user = userEvent.setup();
    const data = makeCoreRunData();
    const client = makeMockClient({ core: data });
    render(
      <ExperimentView
        client={client}
        runId={RUN_ID}
        experiments={data.experiments}
      />,
    );

    expect(await screen.findByText("02 Jan 2026")).toBeInTheDocument();
    const next = screen.getByRole("button", {
      name: "Trade history: next page",
    });
    await waitFor(() => expect(next).toBeEnabled());
    await user.click(next);

    expect(await screen.findByText("02 Feb 2026")).toBeInTheDocument();
    expect(screen.getByText("Rows 11–12 of 12")).toBeInTheDocument();
    expect(client.getTrades).toHaveBeenLastCalledWith(
      RUN_ID,
      EXPERIMENT_ONE,
      10,
      10,
      expect.any(AbortSignal),
    );
  });

  it("paginates snapshots independently from trade history", async () => {
    const user = userEvent.setup();
    const data = makeCoreRunData();
    const client = makeMockClient({ core: data });
    vi.mocked(client.getSnapshots).mockImplementation(
      async (_runId, experimentId, _limit, offset) =>
        snapshotsPage(offset, experimentId),
    );
    render(
      <ExperimentView
        client={client}
        runId={RUN_ID}
        experiments={data.experiments}
      />,
    );

    const snapshotPanel = screen
      .getByRole("heading", { name: "Portfolio snapshots" })
      .closest("article")!;
    expect(await within(snapshotPanel).findByText("05 Jan 2026")).toBeInTheDocument();
    const next = screen.getByRole("button", {
      name: "Portfolio snapshots: next page",
    });
    await waitFor(() => expect(next).toBeEnabled());
    await user.click(next);

    expect(await screen.findByText("02 Mar 2026")).toBeInTheDocument();
    expect(screen.getByText("Rows 11–12 of 12")).toBeInTheDocument();
    expect(client.getSnapshots).toHaveBeenLastCalledWith(
      RUN_ID,
      EXPERIMENT_ONE,
      10,
      10,
      expect.any(AbortSignal),
    );
    expect(client.getTrades).not.toHaveBeenCalledWith(
      RUN_ID,
      EXPERIMENT_ONE,
      10,
      10,
      expect.any(AbortSignal),
    );
    expect(screen.getByText("02 Jan 2026")).toBeInTheDocument();
  });

  it("resets both independent page offsets when the experiment changes", async () => {
    const user = userEvent.setup();
    const data = makeCoreRunData();
    const client = makeMockClient({ core: data });
    vi.mocked(client.getSnapshots).mockImplementation(
      async (_runId, experimentId, _limit, offset) =>
        snapshotsPage(offset, experimentId),
    );
    render(
      <ExperimentView
        client={client}
        runId={RUN_ID}
        experiments={data.experiments}
      />,
    );

    const tradeNext = screen.getByRole("button", {
      name: "Trade history: next page",
    });
    const snapshotNext = screen.getByRole("button", {
      name: "Portfolio snapshots: next page",
    });
    await waitFor(() => {
      expect(tradeNext).toBeEnabled();
      expect(snapshotNext).toBeEnabled();
    });
    await user.click(tradeNext);
    await user.click(snapshotNext);
    await waitFor(() => {
      expect(client.getTrades).toHaveBeenLastCalledWith(
        RUN_ID,
        EXPERIMENT_ONE,
        10,
        10,
        expect.any(AbortSignal),
      );
      expect(client.getSnapshots).toHaveBeenLastCalledWith(
        RUN_ID,
        EXPERIMENT_ONE,
        10,
        10,
        expect.any(AbortSignal),
      );
    });

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Experiment" }),
      EXPERIMENT_TWO,
    );

    await waitFor(() => {
      expect(client.getTrades).toHaveBeenLastCalledWith(
        RUN_ID,
        EXPERIMENT_TWO,
        10,
        0,
        expect.any(AbortSignal),
      );
      expect(client.getSnapshots).toHaveBeenLastCalledWith(
        RUN_ID,
        EXPERIMENT_TWO,
        10,
        0,
        expect.any(AbortSignal),
      );
    });
    expect(screen.getAllByText("Rows 1–10 of 12")).toHaveLength(2);
    expect(screen.queryByText("02 Mar 2026")).not.toBeInTheDocument();
  });

  it("rejects malformed snapshot pagination metadata visibly", async () => {
    const data = makeCoreRunData();
    const client = makeMockClient({ core: data });
    vi.mocked(client.getSnapshots).mockResolvedValue({
      ...snapshotsPage(0),
      limit: 25,
    });
    render(
      <ExperimentView
        client={client}
        runId={RUN_ID}
        experiments={data.experiments}
      />,
    );

    const snapshotPanel = screen
      .getByRole("heading", { name: "Portfolio snapshots" })
      .closest("article")!;
    const alert = await within(snapshotPanel).findByRole("alert");
    expect(alert).toHaveTextContent(
      "Portfolio snapshots unavailable: The PAISA API returned inconsistent pagination metadata.",
    );
    expect(within(snapshotPanel).queryByText("05 Jan 2026")).not.toBeInTheDocument();
  });
});
