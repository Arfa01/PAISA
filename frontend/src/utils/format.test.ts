import { describe, expect, it } from "vitest";

import {
  formatNumber,
  formatPercent,
  formatPkr,
  formatPkrMinorUnits,
  parsePkrMinorUnits,
} from "./format";

describe("dashboard number formatting", () => {
  it("formats exact fixed-scale PKR strings without binary-float conversion", () => {
    expect(formatPkr("1234567.89")).toBe("PKR 1,234,567.89");
    expect(formatPkr("-1250.50")).toBe("PKR -1,250.50");
    expect(formatPkr("1250.5")).toBe("N/A");
    expect(formatPkr(null)).toBe("N/A");
  });

  it("formats decimal rates as percentages consistently", () => {
    expect(formatPercent(0.12345)).toBe("12.35%");
    expect(formatPercent(-0.0070176, 3)).toBe("-0.702%");
    expect(formatPercent(null)).toBe("N/A");
  });

  it("never emits NaN or Infinity", () => {
    expect(formatPercent(Number.NaN)).toBe("N/A");
    expect(formatPercent(Number.POSITIVE_INFINITY)).toBe("N/A");
    expect(formatNumber(Number.NEGATIVE_INFINITY)).toBe("N/A");
    expect(formatPercent(Number.MAX_VALUE)).toBe("N/A");
  });

  it("converts chart money through exact integer minor units", () => {
    expect(parsePkrMinorUnits("1234567.89")).toBe(123456789);
    expect(parsePkrMinorUnits("-1250.50")).toBe(-125050);
    expect(formatPkrMinorUnits(123456789)).toBe("PKR 1,234,567.89");
    expect(parsePkrMinorUnits("90071992547410.00")).toBeNull();
    expect(formatPkrMinorUnits(Number.POSITIVE_INFINITY)).toBe("N/A");
  });
});
