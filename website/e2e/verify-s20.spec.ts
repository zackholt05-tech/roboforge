// `make web-verify-s20` (Session 20, DECISIONS D-136, D-137): the viewer verification of
// e2e/verify.spec.ts, run again because the visual pass changed files under src/three/. Same page
// (src/verify/VerifyPage.tsx), same checks and same D-091 thresholds: the viewer's silhouettes
// through the exact MuJoCo cameras against MuJoCo's segmentation renders of the same states.
// Writes reports/session20/observatory/ only; the Session 10 result and montage are never written.
import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const OUT = join(import.meta.dirname, "..", "..", "reports", "session20", "observatory");

test("viewer silhouettes match MuJoCo's reference frames", async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto("/?verify=all");
  await page.waitForFunction(() => (window as unknown as { __verification?: unknown }).__verification, null, { timeout: 540_000 });
  const doc = await page.evaluate(() => (window as unknown as { __verification: Record<string, unknown> }).__verification);
  expect(doc.error, String(doc.error)).toBeUndefined();
  mkdirSync(join(OUT, "verification"), { recursive: true });
  writeFileSync(join(OUT, "viewer_verification.json"), JSON.stringify(doc, null, 1) + "\n");
  await page.screenshot({ path: join(OUT, "verification", "viewer_montage.jpg"), fullPage: true, type: "jpeg", quality: 82 });
  const frames = doc.frames as { pass: boolean }[];
  expect(frames.length).toBeGreaterThanOrEqual(3);
  expect(doc.pass).toBe(true);
});
