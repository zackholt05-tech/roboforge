// `make web-verify-ext`: the viewer verification of e2e/verify.spec.ts, for the v1.1 extension
// clips (D-134): the viewer's silhouettes through the exact MuJoCo cameras against MuJoCo's
// segmentation renders of the same recorded states, with the D-091 thresholds. Writes the
// committed result reports/session18/observatory/viewer_verification.json and a review montage;
// the Session 10 result is never touched.
import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const OUT = join(import.meta.dirname, "..", "..", "reports", "session18", "observatory");

test("extension clips: viewer silhouettes match MuJoCo's reference frames", async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto("/?verify=all&ext");
  await page.waitForFunction(() => (window as unknown as { __verification?: unknown }).__verification, null, { timeout: 540_000 });
  const doc = await page.evaluate(() => (window as unknown as { __verification: Record<string, unknown> }).__verification);
  expect(doc.error, String(doc.error)).toBeUndefined();
  expect(doc.scope).toBe("v1.1 extension clips (D-134)");
  writeFileSync(join(OUT, "viewer_verification.json"), JSON.stringify(doc, null, 1) + "\n");
  mkdirSync(join(OUT, "verification"), { recursive: true });
  await page.screenshot({ path: join(OUT, "verification", "viewer_montage.jpg"), fullPage: true, type: "jpeg", quality: 82 });
  const frames = doc.frames as { pass: boolean }[];
  expect(frames.length).toBeGreaterThanOrEqual(3);
  expect(doc.pass).toBe(true);
});
