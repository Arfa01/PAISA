import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

describe("PAISA favicon", () => {
  it("is explicitly linked and remains a small self-contained SVG", async () => {
    const [indexHtml, favicon] = await Promise.all([
      readFile(resolve(process.cwd(), "index.html"), "utf-8"),
      readFile(resolve(process.cwd(), "public", "favicon.svg"), "utf-8"),
    ]);

    expect(indexHtml).toContain(
      '<link rel="icon" type="image/svg+xml" href="/favicon.svg" />',
    );
    expect(favicon).toMatch(/^<svg[\s>]/);
    expect(favicon).not.toMatch(/<(?:image|script)\b/i);
    expect(Buffer.byteLength(favicon, "utf-8")).toBeLessThan(2_048);
  });
});
