// Test helpers: read the generated site data (website/public/data, `make web-data`) from disk.
// Tests that need it fail with the fix when it is missing, rather than passing vacuously.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";

export const DATA = join(import.meta.dirname, "..", "..", "public", "data");

export function requireData(): void {
  if (!existsSync(join(DATA, "registry.json"))) {
    throw new Error(`${DATA}/registry.json is missing: run \`make web-data\` first`);
  }
}

export function readData(rel: string): unknown {
  const buf = readFileSync(join(DATA, rel));
  return JSON.parse((rel.endsWith(".gz") && buf[0] === 0x1f ? gunzipSync(buf) : buf).toString("utf8"));
}

export function listData(sub: string): string[] {
  const dir = join(DATA, sub);
  return existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".json.gz")).map((f) => `${sub}/${f}`) : [];
}
