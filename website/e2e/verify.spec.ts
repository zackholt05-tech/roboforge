// `make web-verify`: render the viewer's silhouettes through the exact MuJoCo cameras and
// compare them with MuJoCo's segmentation renders (src/verify/VerifyPage.tsx). Writes the
// committed result reports/session10/observatory/viewer_verification.json and a review montage.
import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const OUT = join(import.meta.dirname, "..", "..", "reports", "session10", "observatory");

test("viewer silhouettes match MuJoCo's reference frames", async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto("/?verify=all");
  await page.waitForFunction(() => (window as unknown as { __verification?: unknown }).__verification, null, { timeout: 540_000 });
  const doc = await page.evaluate(() => (window as unknown as { __verification: Record<string, unknown> }).__verification);
  expect(doc.error, String(doc.error)).toBeUndefined();
  writeFileSync(join(OUT, "viewer_verification.json"), JSON.stringify(doc, null, 1) + "\n");
  mkdirSync(join(OUT, "verification"), { recursive: true });
  await page.screenshot({ path: join(OUT, "verification", "viewer_montage.jpg"), fullPage: true, type: "jpeg", quality: 82 });
  const frames = doc.frames as { pass: boolean }[];
  expect(frames.length).toBeGreaterThanOrEqual(3);
  expect(doc.pass).toBe(true);
});
