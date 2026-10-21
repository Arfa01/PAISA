import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeCoreRunData } from "../test/fixtures";
import { ProvenanceView } from "./ProvenanceView";

describe("data provenance and limitations", () => {
  it("renders source evidence and the required interpretation limits", () => {
    const { source } = makeCoreRunData();
    const { container } = render(<ProvenanceView source={source} />);

    expect(
      screen.getByRole("heading", { name: "Data provenance and limitations" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Verified Legacy Psx Cache")).toBeInTheDocument();
    expect(screen.getAllByText("Psx Dps")).toHaveLength(2);
    expect(screen.getByText("Copy And Seal")).toBeInTheDocument();
    expect(screen.getByText("Checksum Bound")).toBeInTheDocument();
    expect(screen.getByText("1,234")).toBeInTheDocument();
    expect(screen.getAllByText(/not adjusted for corporate actions/i)).toHaveLength(2);
    expect(
      screen.getAllByText(/close-to-close; portfolio returns begin/i),
    ).toHaveLength(2);
    expect(container).not.toHaveTextContent("/Users/");
    expect(container).not.toHaveTextContent("/private/");
  });
});
