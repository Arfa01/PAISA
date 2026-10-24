import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ApiError, PaisaApiClient } from "../api/client";
import {
  COMPARISON_ID,
  DISCLAIMER,
  OTHER_COMPARISON_ID,
  UNIFIED_MODELS,
  makeMockClient,
  makeUnifiedComparisonData,
  makeUnifiedComparisonList,
} from "../test/fixtures";
import { UnifiedComparisonView } from "./UnifiedComparisonView";

describe("UnifiedComparisonView", () => {
  it("renders each of the six models exactly once in the authoritative table", async () => {
    const client = makeMockClient();
    render(<UnifiedComparisonView client={client} />);

    expect(
      await screen.findByRole("heading", { name: "Unified Six-Model Comparison" }),
    ).toBeInTheDocument();
    const table = await screen.findByRole("table", {
      name: "Unified six-model prediction and portfolio metrics",
    });
    expect(within(table).getAllByRole("row")).toHaveLength(7);
    for (const model of UNIFIED_MODELS) {
      expect(within(table).getByRole("row", { name: new RegExp(model) })).toBeInTheDocument();
    }
    expect(within(table).getByText("20-session regularized recurrent")).toBeInTheDocument();
  });

  it("keeps every outcome separate and explains classification versus portfolio performance", async () => {
    const client = makeMockClient();
    const { container } = render(<UnifiedComparisonView client={client} />);

    await screen.findByRole("heading", { name: "Authoritative reported outcomes" });
    for (const label of [
      "Validation-selected model",
      "Best shared project holdout balanced accuracy",
      "Best shared project holdout F1",
      "Best model portfolio",
      "Lowest-drawdown model portfolio",
      "Lowest-fee model portfolio",
      "Buy-and-hold",
      "Any model beat buy-and-hold?",
    ]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    expect(container).toHaveTextContent("shared project holdout");
    expect(container).toHaveTextContent(
      "Better classification performance on the shared project holdout did not necessarily produce better simulated portfolio performance.",
    );
    expect(container).not.toHaveTextContent(/universal winner/i);
    expect(container).not.toHaveTextContent(/single winner/i);
    expect(screen.getByLabelText("Academic simulation disclaimer")).toHaveTextContent(
      DISCLAIMER,
    );
  });

  it("uses the exact shared project holdout term in visible and accessible labels", async () => {
    const client = makeMockClient();
    const { container } = render(<UnifiedComparisonView client={client} />);

    expect(
      await screen.findByRole("heading", {
        name: "Shared project holdout confusion matrices",
      }),
    ).toBeInTheDocument();
    for (const label of [
      "Shared project holdout observations",
      "Shared project holdout target start",
      "Shared project holdout target end",
    ]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    const evidenceTable = screen.getByRole("table", {
      name: "Unified six-model prediction and portfolio metrics",
    });
    for (const column of [
      "Shared project holdout accuracy",
      "Shared project holdout balanced accuracy",
      "Shared project holdout F1",
    ]) {
      expect(
        within(evidenceTable).getByRole("columnheader", { name: column }),
      ).toBeInTheDocument();
    }
    for (const model of UNIFIED_MODELS) {
      expect(
        screen.getByRole("table", {
          name: `${model} shared project holdout confusion matrix`,
        }),
      ).toBeInTheDocument();
    }
    expect(container.textContent).not.toMatch(/shared holdout/i);
  });

  it("uses persisted summary outcomes instead of deriving cards from metric rows", async () => {
    const data = makeUnifiedComparisonData();
    data.summary.reported_outcomes.best_shared_holdout_balanced_accuracy = {
      ...data.summary.reported_outcomes.best_shared_holdout_balanced_accuracy,
      variant_id: "M-001",
      registered_model_id: "registered-m-001",
      balanced_accuracy: 0.11,
    };
    const client = makeMockClient();
    vi.mocked(client.getUnifiedComparisonData).mockResolvedValue(data);
    render(<UnifiedComparisonView client={client} />);

    const label = await screen.findByText(
      "Best shared project holdout balanced accuracy",
    );
    const card = label.closest("article");
    expect(card).toHaveTextContent("M-001");
    expect(card).toHaveTextContent("11.00%");
  });

  it("formats PKR, percentages, and null values without unsafe numeric text", async () => {
    const data = makeUnifiedComparisonData();
    data.predictionMetrics.models[0]!.accuracy = Number.NaN;
    data.portfolioMetrics.models[0]!.total_return = Number.POSITIVE_INFINITY;
    data.portfolioMetrics.models[0]!.ending_equity = "Infinity";
    Object.assign(data.detail as object, {
      hidden_model_binary: "models/private.joblib",
      hidden_local_path: "/Users/private-owner/model.joblib",
    });
    const client = makeMockClient();
    vi.mocked(client.getUnifiedComparisonData).mockResolvedValue(data);
    const { container } = render(<UnifiedComparisonView client={client} />);

    await screen.findByRole("heading", { name: "Model and portfolio evidence" });
    expect(container).toHaveTextContent("PKR 105,987.04");
    expect(container).toHaveTextContent("5.99%");
    expect(container).toHaveTextContent("N/A");
    expect(container).not.toHaveTextContent("NaN");
    expect(container).not.toHaveTextContent("Infinity");
    expect(container).not.toHaveTextContent("joblib");
    expect(container).not.toHaveTextContent("/Users/");
  });

  it.each([
    "models/private.joblib",
    "models/private.joblib/metadata",
    "models\\private.joblib\\metadata",
    "models/private.joblib.",
    "models/private.pb",
    "//server/share/model.keras",
    "See https://example.test/report?source=/Users/private/report.csv",
    "See https://example.test/report?source=%2FUsers%2Fprivate%2Freport.csv",
    "See https://example.test/models/private%2Ejoblib",
  ])("rejects renderable prohibited limitation text: %s", async (unsafeText) => {
    const data = makeUnifiedComparisonData();
    data.summary.limitations = [unsafeText];
    const responses = new Map<string, unknown>([
      [`/api/v1/unified-comparisons/${COMPARISON_ID}`, data.detail],
      [`/api/v1/unified-comparisons/${COMPARISON_ID}/sources`, data.sources],
      [`/api/v1/unified-comparisons/${COMPARISON_ID}/summary`, data.summary],
      [`/api/v1/unified-comparisons/${COMPARISON_ID}/prediction-metrics`, data.predictionMetrics],
      [`/api/v1/unified-comparisons/${COMPARISON_ID}/validation-metrics`, data.validationMetrics],
      [`/api/v1/unified-comparisons/${COMPARISON_ID}/portfolio-metrics`, data.portfolioMetrics],
      [`/api/v1/unified-comparisons/${COMPARISON_ID}/prediction-rankings`, data.predictionRankings],
      [`/api/v1/unified-comparisons/${COMPARISON_ID}/portfolio-rankings`, data.portfolioRankings],
      [`/api/v1/unified-comparisons/${COMPARISON_ID}/common-date-contract`, data.commonDateContract],
      [`/api/v1/unified-comparisons/${COMPARISON_ID}/confusion-matrices`, data.confusionMatrices],
    ]);
    vi.stubGlobal("fetch", vi.fn((input: string | URL | Request) => {
      const url = new URL(String(input));
      if (url.pathname === "/api/v1/unified-comparisons") {
        return Promise.resolve(new Response(JSON.stringify(makeUnifiedComparisonList()), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      }
      return Promise.resolve(new Response(JSON.stringify(responses.get(url.pathname)), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }));
    }));
    const client = new PaisaApiClient("http://127.0.0.1:8000", 1_000);

    const { container } = render(<UnifiedComparisonView client={client} />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "The PAISA API response contained prohibited path or model-binary data.",
    );
    expect(container).not.toHaveTextContent(unsafeText);
  });

  it("shows an inventory loading state, error state, and empty state without fallback data", async () => {
    const pendingClient = makeMockClient();
    vi.mocked(pendingClient.listUnifiedComparisons).mockReturnValue(
      new Promise(() => undefined),
    );
    const pending = render(<UnifiedComparisonView client={pendingClient} />);
    expect(screen.getByRole("status")).toHaveTextContent(
      "Loading completed unified comparisons",
    );
    pending.unmount();

    const errorClient = makeMockClient();
    vi.mocked(errorClient.listUnifiedComparisons).mockRejectedValue(
      new ApiError("Could not reach the comparison API.", "network"),
    );
    const failed = render(<UnifiedComparisonView client={errorClient} />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Could not load unified comparisons");
    expect(alert).toHaveTextContent("No fixture or recomputed result has been substituted.");
    failed.unmount();

    const emptyClient = makeMockClient();
    vi.mocked(emptyClient.listUnifiedComparisons).mockResolvedValue(
      makeUnifiedComparisonList([], null),
    );
    render(<UnifiedComparisonView client={emptyClient} />);
    await screen.findByRole("heading", {
      name: "No immutable unified comparison runs are available.",
    });
    const empty = screen.getByRole("status");
    expect(empty).toHaveTextContent("No immutable unified comparison runs are available.");
    expect(emptyClient.getUnifiedComparisonData).not.toHaveBeenCalled();
  });

  it("surfaces a configured default that is absent instead of silently selecting another run", async () => {
    const client = makeMockClient();
    vi.mocked(client.listUnifiedComparisons).mockResolvedValue(
      makeUnifiedComparisonList(
        [
          {
            ...makeUnifiedComparisonList().rows[0]!,
            is_default: false,
          },
        ],
        OTHER_COMPARISON_ID,
      ),
    );
    vi.mocked(client.getUnifiedComparisonData).mockRejectedValue(
      new ApiError("The configured comparison failed integrity checks.", "http", 409),
    );
    render(<UnifiedComparisonView client={client} />);

    const alert = await screen.findByRole("alert");
    expect(client.getUnifiedComparisonData).toHaveBeenCalledWith(
      OTHER_COMPARISON_ID,
      expect.any(AbortSignal),
    );
    expect(alert).toHaveTextContent("failed integrity checks");
    expect(screen.getByRole("combobox", { name: "Comparison run" })).toHaveValue(
      OTHER_COMPARISON_ID,
    );
  });

  it("clears stale comparison content while switching runs", async () => {
    const user = userEvent.setup();
    const rows = [
      makeUnifiedComparisonList().rows[0]!,
      {
        ...makeUnifiedComparisonList().rows[0]!,
        comparison_id: OTHER_COMPARISON_ID,
        is_default: false,
      },
    ];
    const client = makeMockClient();
    vi.mocked(client.listUnifiedComparisons).mockResolvedValue(
      makeUnifiedComparisonList(rows, COMPARISON_ID),
    );
    let resolveNext!: (data: ReturnType<typeof makeUnifiedComparisonData>) => void;
    const next = new Promise<ReturnType<typeof makeUnifiedComparisonData>>((resolve) => {
      resolveNext = resolve;
    });
    vi.mocked(client.getUnifiedComparisonData).mockImplementation((comparisonId) =>
      comparisonId === OTHER_COMPARISON_ID
        ? next
        : Promise.resolve(makeUnifiedComparisonData(comparisonId)),
    );
    render(<UnifiedComparisonView client={client} />);

    await screen.findByText(COMPARISON_ID);
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Comparison run" }),
      OTHER_COMPARISON_ID,
    );
    expect(await screen.findByText(`Loading ${OTHER_COMPARISON_ID}…`)).toBeInTheDocument();
    expect(screen.queryByText(COMPARISON_ID)).not.toBeInTheDocument();

    resolveNext(makeUnifiedComparisonData(OTHER_COMPARISON_ID));
    await waitFor(() =>
      expect(screen.getByText(OTHER_COMPARISON_ID)).toBeInTheDocument(),
    );
  });
});
