const MONEY_PATTERN = /^(-?)(\d+)(?:\.(\d{2}))$/;

export function formatPkr(value: string | null | undefined): string {
  if (typeof value !== "string") return "N/A";
  const match = MONEY_PATTERN.exec(value.trim());
  if (!match) return "N/A";
  const sign = match[1] ?? "";
  const whole = match[2] ?? "";
  const fraction = match[3] ?? "";
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return `PKR ${sign}${grouped}.${fraction}`;
}

export function formatPercent(
  value: number | null | undefined,
  digits = 2,
): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  const percentage = value * 100;
  if (!Number.isFinite(percentage)) return "N/A";
  return `${percentage.toFixed(digits)}%`;
}

/**
 * Convert a fixed-scale PKR API string into an exactly represented integer
 * count of paisa. Values outside JavaScript's safe integer range are rejected.
 */
export function parsePkrMinorUnits(
  value: string | null | undefined,
): number | null {
  if (typeof value !== "string") return null;
  const match = MONEY_PATTERN.exec(value.trim());
  if (!match) return null;
  const whole = BigInt(match[2] ?? "0");
  const fraction = BigInt(match[3] ?? "0");
  const minorUnits = (whole * 100n + fraction) * (match[1] === "-" ? -1n : 1n);
  if (
    minorUnits > BigInt(Number.MAX_SAFE_INTEGER) ||
    minorUnits < BigInt(Number.MIN_SAFE_INTEGER)
  ) {
    return null;
  }
  return Number(minorUnits);
}

export function formatPkrMinorUnits(value: number): string {
  if (!Number.isSafeInteger(value)) return "N/A";
  const sign = value < 0 ? "-" : "";
  const absolute = BigInt(Math.abs(value));
  const whole = (absolute / 100n).toString();
  const fraction = (absolute % 100n).toString().padStart(2, "0");
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return `PKR ${sign}${grouped}.${fraction}`;
}

export function formatNumber(
  value: number | null | undefined,
  digits = 2,
): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return value.toLocaleString("en-PK", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatInteger(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return Math.trunc(value).toLocaleString("en-PK");
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "N/A";
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value);
  if (!match) return "N/A";
  return `${match[3]} ${new Intl.DateTimeFormat("en", { month: "short" }).format(
    new Date(Date.UTC(2000, Number(match[2]) - 1, 1)),
  )} ${match[1]}`;
}

export function humanize(value: string | null | undefined): string {
  if (!value) return "N/A";
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function finiteNumber(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined || value === "") return null;
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}
