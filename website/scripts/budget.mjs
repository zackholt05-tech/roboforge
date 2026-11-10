// Asset budgets of the built site (MASTER_SPEC §8.5 "avoid shipping 100s of MB"; D-088).
// Fails the build when a file class exceeds its budget. Sizes are bytes on disk (gzipped JSON
// is already compressed); JavaScript and CSS are also reported gzipped, as served.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { gzipSync } from "node:zlib";

const DIST = new URL("../dist/", import.meta.url).pathname;
const MB = 1024 * 1024, KB = 1024;
const BUDGETS = [
  { name: "JavaScript, all chunks (gzip)", match: (f) => f.endsWith(".js"), gzip: true, total: 520 * KB },
  { name: "CSS (gzip)", match: (f) => f.endsWith(".css"), gzip: true, total: 40 * KB },
  { name: "fonts", match: (f) => /\.(woff2?|ttf)$/.test(f), total: 1.6 * MB },
  { name: "robot meshes", match: (f) => f.startsWith("assets/meshes"), total: 4 * MB },
  { name: "one clip", match: (f) => f.startsWith("data/clips/"), each: 1.5 * MB },
  { name: "one denoising trace", match: (f) => f.startsWith("data/denoise/"), each: 4 * MB },
  { name: "one point-cloud file", match: (f) => f.startsWith("data/clouds/"), each: 1.5 * MB },
  { name: "one video", match: (f) => f.endsWith(".mp4"), each: 3 * MB },
  { name: "all data", match: (f) => f.startsWith("data/"), total: 80 * MB },
  { name: "first view (page, JS, CSS, registry, findings, hero clip, meshes)", first: true, total: 6.5 * MB },
];

const files = [];
(function walk(d) {
  for (const e of readdirSync(d)) {
    const p = join(d, e);
    if (statSync(p).isDirectory()) walk(p);
    else files.push(relative(DIST, p));
  }
})(DIST);
const size = (f, gz) => (gz ? gzipSync(readFileSync(join(DIST, f))).length : statSync(join(DIST, f)).size);
const registry = JSON.parse(readFileSync(join(DIST, "data/registry.json"), "utf8"));
const hero = registry.clips.find((c) => c.roles.includes("hero"));
const first = files.filter((f) => f === "index.html" || /\.(js|css)$/.test(f) || f === "data/registry.json" ||
  f === "data/findings.json" || f.startsWith("assets/meshes") || (hero && f === `data/clips/${hero.id}.json.gz`));
let bad = 0;
const fmt = (b) => (b >= MB ? `${(b / MB).toFixed(2)} MB` : `${(b / KB).toFixed(1)} KB`);
for (const b of BUDGETS) {
  const set = b.first ? first : files.filter(b.match);
  const sizes = set.map((f) => [f, size(f, b.gzip || (b.first && /\.(js|css|html)$/.test(f)))]);
  if (b.each !== undefined) {
    const worst = sizes.reduce((a, s) => (s[1] > a[1] ? s : a), ["none", 0]);
    const ok = worst[1] <= b.each;
    if (!ok) bad++;
    console.log(`${ok ? "ok  " : "OVER"} ${b.name}: largest ${fmt(worst[1])} (${worst[0]}), budget ${fmt(b.each)}, ${sizes.length} files`);
  } else {
    const t = sizes.reduce((a, s) => a + s[1], 0);
    const ok = t <= b.total;
    if (!ok) bad++;
    console.log(`${ok ? "ok  " : "OVER"} ${b.name}: ${fmt(t)} of ${fmt(b.total)}, ${sizes.length} files`);
  }
}
if (bad) { console.error(`${bad} asset budget(s) exceeded`); process.exit(1); }
