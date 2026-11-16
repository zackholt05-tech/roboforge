// `make web-audit AUDIT_TAG=before|after` (Session 20, DECISIONS D-136, D-137): the visual audit
// record of the built Observatory. It only looks at the site: screenshots of every section as a
// visitor sees it (viewport tiles, scrolled through each section) at three sizes, with reduced
// motion and without WebGL; each 3D viewer during playback; the denoising explorer at its first
// and last step; and, from the DOM, the words and text blocks visible by default per section.
// Output: logs/session20/visual_audit/<tag>/ (JPEG tiles, word_counts.json, index.json).
import { expect, test, type Locator, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const TAG = process.env.AUDIT_TAG ?? "";
const OUT = join(import.meta.dirname, "..", "..", "logs", "session20", "visual_audit", TAG);
/** Section ids in page order; the hero's id is "top". */
const SECTIONS = ["top", "why", "lab", "compare", "inside", "findings", "followups", "pixels", "methods"];
const NAME: Record<string, string> = { top: "hero" };
const SIZES = [[1440, 900], [1280, 800], [390, 844]] as const;
const TOPBAR = 56;
const QUALITY = 58;
const index: Record<string, unknown> = { tag: TAG, tiles: {} as Record<string, string[]> };

test.describe.configure({ mode: "serial" });
test.beforeAll(() => {
  if (TAG !== "before" && TAG !== "after") throw new Error("set AUDIT_TAG=before or AUDIT_TAG=after");
  mkdirSync(OUT, { recursive: true });
});
test.afterAll(() => writeFileSync(join(OUT, "index.json"), JSON.stringify(index, null, 1) + "\n"));

async function ready(page: Page) {
  await page.waitForSelector("#methods h2", { timeout: 60_000 });
  for (const id of SECTIONS) {
    await page.locator(`#${id}`).scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);
  }
  await expect(page.locator('main [role="status"]')).toHaveCount(0, { timeout: 60_000 });
  await page.evaluate(() => document.fonts.ready);
}

/** Viewport tiles of one section: what a visitor sees while scrolling through it. */
async function tiles(page: Page, id: string, dir: string, limit = 99): Promise<string[]> {
  mkdirSync(join(OUT, dir), { recursive: true });
  const vh = page.viewportSize()!.height;
  const box = await page.locator(`#${id}`).evaluate((el) => {
    const r = el.getBoundingClientRect();
    return { top: r.top + window.scrollY, height: r.height };
  });
  const step = vh - TOPBAR - 24;
  const start = id === "top" ? 0 : box.top - TOPBAR;
  const out: string[] = [];
  for (let i = 0; i < limit; i++) {
    const y = start + i * step;
    if (i > 0 && y >= box.top + box.height - TOPBAR - 40) break;
    await page.evaluate((yy) => window.scrollTo({ top: yy, behavior: "instant" as ScrollBehavior }), y);
    await page.waitForTimeout(450);
    await expect(page.locator(`#${id} [role="status"]`)).toHaveCount(0, { timeout: 30_000 });
    const file = `${dir}/${NAME[id] ?? id}_${String(i + 1).padStart(2, "0")}.jpg`;
    await page.screenshot({ path: join(OUT, file), type: "jpeg", quality: QUALITY, scale: "css" });
    out.push(file);
  }
  (index.tiles as Record<string, string[]>)[`${dir}/${NAME[id] ?? id}`] = out;
  return out;
}

for (const [w, h] of SIZES) {
  test(`sections at ${w}x${h}`, async ({ page }) => {
    test.setTimeout(900_000);
    await page.setViewportSize({ width: w, height: h });
    await page.goto("/");
    await ready(page);
    for (const id of SECTIONS) await tiles(page, id, `${w}x${h}`);
  });
}

test("reduced motion: the top of every section at 1440x900", async ({ page }) => {
  test.setTimeout(600_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await ready(page);
  for (const id of SECTIONS) await tiles(page, id, "reduced_motion", 2);
});

test("no WebGL: the top of every section at 1440x900", async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto("/?nowebgl");
  await ready(page);
  for (const id of SECTIONS) await tiles(page, id, "nowebgl", 2);
});

/** Move a transport's scrubber to a fraction of the episode, play briefly, and capture. */
async function midPlayback(page: Page, transport: Locator, target: Locator, file: string, fraction = 0.4) {
  await transport.scrollIntoViewIfNeeded();
  await transport.locator('input[type="range"]').evaluate((el, f) => {
    const input = el as HTMLInputElement;
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
    set.call(input, String(Math.round(Number(input.max) * f)));
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }, fraction);
  await target.scrollIntoViewIfNeeded();
  await page.waitForTimeout(500);
  await transport.getByRole("button", { name: "Play" }).click();
  await page.waitForTimeout(700);
  await target.screenshot({ path: join(OUT, file), type: "jpeg", quality: QUALITY, scale: "css" });
  await transport.getByRole("button", { name: "Pause" }).click().catch(() => {});
}

test("each 3D viewer during playback, and the denoising explorer at its first and last step", async ({ page }) => {
  test.setTimeout(600_000);
  mkdirSync(join(OUT, "viewers"), { recursive: true });
  await page.goto("/");
  await ready(page);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior }));
  await page.waitForTimeout(2600);
  await page.screenshot({ path: join(OUT, "viewers/hero_mid.jpg"), type: "jpeg", quality: QUALITY, scale: "css" });
  const shots: [string, string, string, number][] = [
    ["lab", "#lab .transport", "#lab .stage-wrap", 0],
    ["compare", "#compare .transport", "#compare .h2h-pair", 0],
    ["inside", "#inside .transport", "#inside .inside-grid", 0],
    ["followups_x1", "#followups .transport", "#followups .h2h-pair.n3", 0],
    ["followups_x2", "#followups .transport", "#followups .h2h-pair.n2", 1],
    ["pixels", "#pixels .transport", "#pixels .h2h-pair", 0],
  ];
  for (const [name, tr, target, nth] of shots) {
    await midPlayback(page, page.locator(tr).nth(nth), page.locator(target).first(), `viewers/${name}_mid.jpg`);
  }
  const stage = page.locator("#stage-range");
  await stage.scrollIntoViewIfNeeded();
  await stage.focus();
  for (const [key, name] of [["Home", "first"], ["End", "last"]] as const) {
    await page.keyboard.press(key);
    await page.waitForTimeout(600);
    await page.locator("#inside .inside-grid").screenshot({ path: join(OUT, `viewers/denoise_${name}_step.jpg`), type: "jpeg", quality: QUALITY, scale: "css" });
  }
});

/** Words and text blocks a visitor can read without opening anything, per section. */
async function wordCounts(page: Page) {
  return page.evaluate((ids) => {
    const words = (s: string) => s.split(/\s+/).filter((t) => /[\p{L}\p{N}]/u.test(t)).length;
    const closedDetails = (n: Node, root: Element): boolean => {
      for (let e = n.parentElement; e && e !== root.parentElement; e = e.parentElement) {
        if (e.tagName === "DETAILS" && !(e as HTMLDetailsElement).open) {
          const sum = e.querySelector(":scope > summary");
          if (!sum || !sum.contains(n)) return true;
        }
      }
      return false;
    };
    const out: Record<string, Record<string, number>> = {};
    for (const id of ids) {
      const root = document.getElementById(id)!;
      const c = { prose: 0, small_print: 0, controls: 0, table: 0, svg: 0, in_closed_disclosures: 0 };
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        const el = n.parentElement!;
        const k = words(n.textContent ?? "");
        if (!k || el.closest("title, option, script, style, .visually-hidden")) continue;
        if (closedDetails(n, root)) { c.in_closed_disclosures += k; continue; }
        if (el.getClientRects().length === 0) continue;
        if (el.closest("svg")) c.svg += k;
        else if (el.closest("table")) c.table += k;
        else if (el.closest("button, legend, label, select, summary, nav")) c.controls += k;
        else if (el.closest(".fine, figcaption, .clip-caption, .stage-note, .erratum, .legend")) c.small_print += k;
        else c.prose += k;
      }
      const visible = (e: Element) => e.getClientRects().length > 0 && !closedDetails(e.firstChild ?? e, root) &&
        words(e.textContent ?? "") > 0;
      const blocks = (sel: string) => [...root.querySelectorAll(sel)].filter(visible).length;
      out[id] = {
        ...c,
        visible_words: c.prose + c.small_print + c.controls + c.table + c.svg,
        reading_words: c.prose + c.small_print,
        text_blocks: blocks("h1, h2, h3, h4, p, li, figcaption, caption, dd"),
        paragraphs: blocks("p"),
        small_print_blocks: blocks("p.fine, figcaption, p.erratum, p.clip-caption"),
        charts: [...root.querySelectorAll("figure.chart")].filter((e) => e.getClientRects().length > 0).length,
        tables: [...root.querySelectorAll("table")].filter((e) => e.getClientRects().length > 0 && !closedDetails(e, root)).length,
        closed_disclosures: [...root.querySelectorAll("details:not([open])")].filter((e) => e.getClientRects().length > 0).length,
        height_px: Math.round(root.getBoundingClientRect().height),
      };
    }
    return out;
  }, SECTIONS);
}

test("words and text blocks visible by default, per section", async ({ page }) => {
  test.setTimeout(300_000);
  const doc: Record<string, unknown> = {
    tag: TAG,
    definition: "Text nodes inside each section that are rendered and not inside a closed <details> (its <summary> counts). " +
      "prose: body text; small_print: .fine notes, figure captions, clip captions, legends and errata; controls: buttons, " +
      "legends of control groups, labels and summaries; table: table cells and captions; svg: chart labels. " +
      "reading_words = prose + small_print. A word is a whitespace-separated token with a letter or digit.",
  };
  for (const [w, h] of [[1440, 900], [390, 844]] as const) {
    await page.setViewportSize({ width: w, height: h });
    await page.goto("/");
    await ready(page);
    const per = await wordCounts(page);
    const total: Record<string, number> = {};
    for (const s of Object.values(per)) for (const [k, v] of Object.entries(s)) total[k] = (total[k] ?? 0) + v;
    doc[`${w}x${h}`] = { sections: Object.fromEntries(Object.entries(per).map(([k, v]) => [NAME[k] ?? k, v])), page: total };
  }
  writeFileSync(join(OUT, "word_counts.json"), JSON.stringify(doc, null, 1) + "\n");
});
