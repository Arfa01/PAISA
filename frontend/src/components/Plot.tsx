import type { Config, Data, Layout } from "plotly.js";
import type { ComponentType } from "react";
import type { PlotParams } from "react-plotly.js";
import createPlotlyComponentImport from "react-plotly.js/factory.js";
import PlotlyImport from "plotly.js-basic-dist-min";

type ModuleWithDefault<T> = T | { default: ModuleWithDefault<T> };

function unwrapDefault<T>(value: ModuleWithDefault<T>): T {
  let current: ModuleWithDefault<T> = value;
  for (let depth = 0; depth < 3; depth += 1) {
    if (
      typeof current === "object" &&
      current !== null &&
      "default" in current
    ) {
      current = current.default;
      continue;
    }
    break;
  }
  return current as T;
}
const createPlotlyComponent = unwrapDefault<
  (plotly: object) => ComponentType<PlotParams>
>(createPlotlyComponentImport);
const Plotly = unwrapDefault<object>(PlotlyImport);
const ReactPlot = createPlotlyComponent(Plotly);

export interface PlotProps {
  ariaLabel: string;
  className?: string;
  config?: Partial<Config>;
  data: Data[];
  layout: Partial<Layout>;
}

const baseLayout: Partial<Layout> = {
  autosize: true,
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: {
    color: "#263445",
    family: "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif",
    size: 12,
  },
  hovermode: "closest",
  margin: { b: 64, l: 68, r: 24, t: 56 },
  legend: {
    orientation: "h",
    x: 0,
    xanchor: "left",
    y: -0.2,
    yanchor: "top",
  },
};

const baseConfig: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  scrollZoom: false,
};

/**
 * Responsive Plotly host with a concise text alternative for non-visual users.
 * The views pair every chart with a data table, so the chart is supplementary.
 */
export function Plot({
  ariaLabel,
  className,
  config,
  data,
  layout,
}: PlotProps) {
  return (
    <figure className={["plot", className].filter(Boolean).join(" ")}>
      <div
        aria-label={ariaLabel}
        className="plot__canvas"
        role="img"
        tabIndex={0}
      >
        <ReactPlot
          config={{ ...baseConfig, ...config }}
          data={data}
          layout={{
            ...baseLayout,
            ...layout,
            font: { ...baseLayout.font, ...layout.font },
            legend: { ...baseLayout.legend, ...layout.legend },
            margin: { ...baseLayout.margin, ...layout.margin },
          }}
          style={{ height: "100%", minHeight: "22rem", width: "100%" }}
          useResizeHandler
        />
      </div>
      <figcaption className="sr-only">{ariaLabel}</figcaption>
    </figure>
  );
}
