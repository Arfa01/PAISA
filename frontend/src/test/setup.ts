import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

vi.mock("react-plotly.js/factory.js", async () => {
  const React = await import("react");
  return {
    default: () =>
      ({ data }: { data: unknown[] }) =>
        React.createElement("div", {
          "data-testid": "plotly-chart",
          "data-trace-count": data.length,
        }),
  };
});
vi.mock("plotly.js-basic-dist-min", () => ({ default: {} }));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
