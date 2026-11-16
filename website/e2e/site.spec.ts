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

/** The text of `selector` that a visitor reads without opening anything: closed <details> are
 * left out (their summaries stay in), and so is anything that is not rendered. */
async function shownText(page: Page, selector: string): Promise<string> {
  return page.locator(selector).first().evaluate((root) => {
    const closed = (n: Node) => {
      for (let e = n.parentElement; e && e !== root.parentElement; e = e.parentElement) {
        if (e.tagName === "DETAILS" && !(e as HTMLDetailsElement).open) {
          const sum = e.querySelector(":scope > summary");
          if (!sum || !sum.contains(n)) return true;
        }
      }
      return false;
    };
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const out: string[] = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (!closed(n) && n.parentElement!.getClientRects().length > 0) out.push((n.textContent ?? "").trim());
    }
    return out.filter(Boolean).join(" ").replace(/\s+/g, " ");
  });
}

// Session 20 (D-136): the visual pass moved long provenance behind a "Provenance" disclosure and
// cut text. These checks pin what must survive that: every caveat stays readable without opening
// anything in the section that shows its numbers, and every clip's source stays one click away.
test("after the visual pass: caveats visible by default, each clip's source one click away (D-136)", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto("/");
  await settle(page);

  // Hero: what it is, the three headline results from findings.json and their caveats.
  const hero = await shownText(page, "#top");
  for (const t of ["recorded rollout", "MuJoCo simulation", "outcome: success", "source: ", "Simulation only", "privileged",
    "KI-052", "KI-034", "DP3-inspired", "not evidence that the two are equal", "separated, negative"]) expect(hero).toContain(t);
  await expect(page.locator(".glance-cards li")).toHaveCount(3);
  await expect(page.locator(".glance-number").nth(0)).toContainText("+18.7");
  await expect(page.locator(".glance-number").nth(1)).toContainText("−0.7");
  await expect(page.locator(".glance-number").nth(2)).toContainText("−41.3");
  await expect(page.locator(".glance-number").nth(2)).toHaveText(
    (await page.locator("#pixels .h5 .big-number").innerText()).replace(/\s+/g, " ").trim());

  // The hero clip's provenance: closed by default, and complete when opened.
  expect(hero).not.toContain("physics_sha256");
  const run = ((await page.locator(".hero .clip-caption .run-id").getAttribute("title")) ?? "").split(": ").pop()!;
  expect(run).toMatch(/^s\d+_.+__/);
  const prov = page.locator(".hero details.prov");
  await expect(prov).not.toHaveAttribute("open", "");
  await prov.locator("summary").click();
  await expect(prov.locator(".provenance")).toBeVisible();
  for (const t of [run, "Checkpoint", "Start state", "lift_final_test_0000", "physics_sha256", "Recompute check"]) {
    await expect(prov.locator(".provenance")).toContainText(t);
  }

  // Every recorded clip on the page carries its selection label and a provenance disclosure with its run id.
  for (const id of ["lab", "compare", "inside", "followups", "pixels"]) {
    const captions = await page.locator(`#${id} .clip-caption`).count();
    expect(captions, id).toBeGreaterThan(0);
    await expect(page.locator(`#${id} details.prov`), id).toHaveCount(captions);
    for (const d of await page.locator(`#${id} details.prov`).all()) await expect(d).toContainText("source: run_id");
    for (const c of await page.locator(`#${id} .clip-caption`).all()) await expect(c).toContainText("recorded rollout");
  }

  // Findings: privilege, the reading key, KI-052 beside the diffusion comparisons, KI-034 with the Lift numbers.
  const findings = await shownText(page, "#findings");
  for (const t of ["privileged simulator object state", "not evidence of equivalence", "KI-052",
    "validation regime, not held-out"]) expect(findings).toContain(t);
  await page.locator("#findings").getByRole("button", { name: "Lift", exact: true }).first().click();
  expect(await shownText(page, "#findings")).toContain("Lift is saturated");

  // Follow-ups: every study label and X2 condition, without opening a disclosure.
  const follow = await shownText(page, "#followups");
  for (const t of ["exploratory, after the freeze", "validation regime, not held-out", "separate replication", "X2 (Can MH)",
    "never pooled", "within-operator only", "500-step horizon", "Level 1 episodes end at 400 steps", "KI-052",
    "not evidence of equivalence", "never read as equality", "a local build, not peer reviewed",
    "X2 (Can MH): trained on the robomimic Can MH demonstrations", "KI-065"]) expect(follow).toContain(t);
  expect(follow).not.toContain("Recorded evaluator label");
  const x2 = page.locator("#followups .h2h-pair.n2 details.prov").first();
  await x2.locator("summary").click();
  await expect(x2.locator(".provenance")).toBeVisible();
  await expect(x2.locator(".provenance")).toContainText("Recorded evaluator label");
  await expect(x2.locator(".provenance")).toContainText("KI-065");
  await expect(x2.locator(".provenance")).toContainText("source: run_id");
  // The X2 suite is named in the caption; the partition the evaluator recorded stays in Provenance (D-123).
  for (const c of await page.locator("#followups .h2h-pair.n2 .clip-caption").all()) await expect(c).toContainText("X2 held-out start");
  await expect(x2.locator(".provenance")).toContainText("recorded partition eval_adhoc");

  // Cameras and point clouds, and the credits.
  const pixels = await shownText(page, "#pixels");
  for (const t of ["DP3-inspired, not a DP3 reproduction", "KI-034", "Not separated is not evidence of equivalence",
    "separated, negative"]) expect(pixels).toContain(t);
  expect(pixels).not.toMatch(/\d\.\d{6,}/); // no unrounded float (the baseline showed "7.000000000000001 cm")
  expect(await shownText(page, "#methods")).toContain("a local build, not peer reviewed");

  // The navigation marks the section being read.
  await page.locator('nav a[href="#findings"]').click();
  await expect(page.locator('nav a[href="#findings"]')).toHaveAttribute("aria-current", "true");
  await expect(page.locator('nav a[aria-current="true"]')).toHaveCount(1);
  expect(errors).toEqual([]);
});

test("phone layout: row charts fit the screen, labels above their rows", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await settle(page);
  const charts = page.locator("#findings figure.chart.fit .chart-scroll");
  expect(await charts.count()).toBeGreaterThan(2);
  for (const c of await charts.all()) {
    expect(await c.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
  }
  await expect(page.locator("#findings figure.chart.fit svg").first()).toHaveAttribute("aria-label", /95 % CI/);
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
