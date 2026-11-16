// E2E/UI sanity checks of the built Observatory (MASTER_SPEC §8.6): every section renders from
// the generated data without console errors, keyboard control works, reduced motion is
// respected, phones get a usable layout without horizontal scroll, and a missing artifact or a
// missing WebGL shows an explicit message instead of a blank viewer.
import { expect, test, type Page } from "@playwright/test";

const SECTIONS = ["why", "lab", "compare", "inside", "findings", "followups", "pixels", "methods"];

function trackErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(e.message));
  return errors;
}

async function settle(page: Page) {
  await page.waitForSelector("#methods h2", { timeout: 60_000 });
  for (const id of SECTIONS) {
    await page.locator(`#${id}`).scrollIntoViewIfNeeded();
    await page.waitForTimeout(250);
  }
}

test("every section renders from the exported data, without console errors", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto("/");
  await expect(page.locator(".hero .clip-caption")).toContainText("MuJoCo simulation");
  await expect(page.locator(".hero .clip-caption")).toContainText("task: Lift");
  await expect(page.locator(".hero .clip-caption")).toContainText("trained checkpoint");
  await settle(page);
  for (const id of SECTIONS) await expect(page.locator(`#${id} h2`).first()).toBeVisible();
  await expect(page.locator("#pixels .h5 .big-number")).toContainText("−41.3");
  await expect(page.locator("#findings")).toContainText("hierarchical paired");
  await expect(page.locator("#compare .h2h-side .channels")).toHaveCount(2);
  await expect(page.locator("#findings .robustness")).toContainText("validation regime, not held-out");
  await expect(page.locator("#findings .robustness")).toContainText("BC-MLP-chunk, 79.3 %");
  await expect(page.locator("#findings .robustness .manifests summary")).toContainText("(33)");
  await expect(page.locator("#pixels .erratum")).toContainText("42.2");
  await expect(page.locator(".status-error")).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("the v1.1 follow-up studies render with their own labels (D-134)", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto("/#followups");
  const s = page.locator("#followups");
  await s.scrollIntoViewIfNeeded();
  await expect(s.locator("h2")).toHaveText("After v1.0.0: three follow-up studies");
  for (const label of ["exploratory, after the freeze", "validation regime, not held-out", "separate replication",
    "X2 (Can MH)", "within-operator only"]) await expect(s).toContainText(label);
  // X1: three arms on one start; X2: diffusion against one BC family.
  await expect(s.locator(".h2h-pair.n3 .h2h-side")).toHaveCount(3, { timeout: 30_000 });
  await expect(s.locator(".h2h-pair.n2 .h2h-side")).toHaveCount(2, { timeout: 30_000 });
  await expect(s.locator(".h2h-pair.n3")).toContainText("BC-MLP-chunk, clamped");
  await expect(s.locator(".h2h-pair.n2 .provenance")).toHaveCount(2);
  for (const p of await s.locator(".h2h-pair.n2 .provenance").all()) {
    await expect(p).toContainText("KI-065");
    await expect(p).toContainText("X2 (Can MH): trained on the robomimic Can MH demonstrations");
  }
  await s.getByRole("button", { name: "BC-RNN" }).click();
  await s.getByRole("button", { name: "Only BC-RNN succeeds" }).click();
  await expect(s.locator(".h2h-pair.n2 .h2h-name").nth(1)).toHaveText("BC-RNN");
  await expect(s).toContainText("H2-c on Can.");
  await expect(s.locator("table.ext-contrasts")).toHaveCount(2);
  await expect(s.locator(".status-error")).toHaveCount(0);
  await expect(page.locator('nav a[href="#followups"]')).toHaveText("After v1.0");
  expect(errors).toEqual([]);
});

test("keyboard: skip link, transport keys and the denoising controls", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(page.locator(".skip")).toBeFocused();
  await page.locator("#lab").scrollIntoViewIfNeeded();
  const transport = page.locator("#lab .transport");
  await transport.locator("button").first().focus();
  const readout = transport.locator(".readout");
  const before = await readout.textContent();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowRight");
  await expect(readout).not.toHaveText(before ?? "");
  await expect(readout).toContainText("frame 2");
  await page.locator("#inside").scrollIntoViewIfNeeded();
  const stage = page.locator("#stage-range");
  await stage.focus();
  await page.keyboard.press("Home");
  await expect(page.locator("#inside .stage-label")).toContainText("Denoising step 1 of 10");
  await page.keyboard.press("End");
  await expect(page.locator("#inside .stage-label")).toContainText("Sampler output");
});

test("reduced motion: nothing plays until asked", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.locator(".hero-controls button").first()).toHaveText("Play");
  await expect(page.locator(".hero-controls button").nth(1)).toHaveText("Camera drift off");
  await expect(page.locator("#inside .sampler-head .btn")).toHaveText("Next denoising step");
});

test("phone layout: no horizontal scroll; video leads, 3D on request", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await settle(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await expect(page.locator(".hero .stage-fallback video")).toBeVisible();
  await expect(page.locator(".hero .stage-fallback .btn")).toHaveText("Load the 3D reconstruction");
});

test("without WebGL the simulator video is shown, with the reason", async ({ page }) => {
  await page.goto("/?nowebgl");
  await expect(page.locator(".hero .stage-note")).toContainText("WebGL is not available");
  await expect(page.locator(".hero video")).toBeVisible();
});

test("a missing artifact produces an explicit error, not a blank viewer", async ({ page }) => {
  await page.route("**/data/clips/*.json.gz", (r) => r.fulfill({ status: 404, body: "" }));
  await page.goto("/");
  await expect(page.locator(".hero .status-error")).toContainText("make web-data");
});

test("no unbounded render loop: canvases off screen are released", async ({ page }) => {
  await page.goto("/");
  await settle(page);
  await page.locator("#methods").scrollIntoViewIfNeeded();
  await page.waitForTimeout(800);
  const live = await page.evaluate(() => document.querySelectorAll("canvas").length);
  expect(live).toBeLessThanOrEqual(3);
});
