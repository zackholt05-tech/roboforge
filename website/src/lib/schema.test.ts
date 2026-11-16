// Every exported data file parses against its runtime schema (MASTER_SPEC §8.5), and the clips
// satisfy the structural invariants: frame counts, monotonic timestamps, joint naming, no
// missing transforms, and registry entries that match the files.
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { DATA, listData, readData, requireData } from "../test/fixtures";
import { Clip, Clouds, Denoise, ExtClip, ExtRegistry, ExtSelection, Findings, FindingsExt, Registry } from "./schema";

describe("exported Observatory data", () => {
  requireData();
  const reg = Registry.parse(readData("registry.json"));

  it("the registry lists every clip file and nothing else", () => {
    const files = listData("clips").map((f) => f.replace("clips/", "").replace(".json.gz", "")).sort();
    expect(reg.clips.map((c) => c.id).sort()).toEqual(files);
  });

  it("findings.json parses and copies the frozen sources", () => {
    const f = Findings.parse(readData("findings.json"));
    expect(f.sources.length).toBeGreaterThanOrEqual(6);
    expect(Object.keys(f.level1.tasks).sort()).toEqual(["can", "lift"]);
  });

  for (const rc of reg.clips) {
    it(`clip ${rc.id} parses and is internally consistent`, () => {
      const c = Clip.parse(readData(`clips/${rc.id}.json.gz`));
      const n = c.frames.n;
      expect(c.frames.time_s).toHaveLength(n);
      for (let k = 1; k < n; k++) expect(c.frames.time_s[k]).toBeGreaterThan(c.frames.time_s[k - 1]);
      expect(c.frames.pos).toHaveLength(n);
      expect(c.frames.quat_wxyz).toHaveLength(n);
      expect(c.frames.joint_qpos).toHaveLength(n);
      for (let k = 0; k < n; k++) {
        expect(c.frames.pos[k]).toHaveLength(c.frames.dynamic_index.length);
        expect(c.frames.joint_qpos[k]).toHaveLength(c.joints.length);
        for (const q of c.frames.quat_wxyz[k]) expect(Math.abs(Math.hypot(...q) - 1)).toBeLessThan(1e-5);
      }
      expect(c.frames.static_pos).toHaveLength(c.bodies.length);
      for (const g of c.geoms) expect(c.bodies.some((b) => b.name === g.body)).toBe(true);
      expect(c.joints.filter((j) => /^robot0_joint[1-7]$/.test(j))).toHaveLength(7);
      expect(c.actions.executed).toHaveLength(n - 1);
      expect(c.outcome.steps).toBe(n - 1);
      expect(c.provenance.physics_sha256).toBe(rc.physics_sha256);
      expect(c.provenance.eval_run_id).toBe(rc.eval_run_id);
      if (c.checkpoint.sha256 === null) expect(c.policy.learned).toBe(false);
      expect(c.verification.recomputed_vs_recorded.body_pos_max_abs_m).toBeLessThan(1e-9);
    });
  }

  for (const file of listData("denoise")) {
    it(`denoising trace ${file} parses; validation starts only`, () => {
      const d = Denoise.parse(readData(file));
      expect(d.partition).toBe("eval_validation");
      for (const x of d.decisions) expect(x.checks.executed_branch_equals_record).toBe(true);
    });
  }

  for (const file of listData("clouds")) {
    it(`point clouds ${file} parse`, () => {
      const c = Clouds.parse(readData(file));
      expect(c.points.every((p) => p.length === c.spec.n_points)).toBe(true);
    });
  }
});

describe("v1.1 extension data (D-134)", () => {
  requireData();
  const reg = ExtRegistry.parse(readData("ext/ext_registry.json"));
  const sel = ExtSelection.parse(readData("ext/selection.json"));
  const sha = (rel: string) => createHash("sha256").update(readFileSync(join(DATA, "ext", rel))).digest("hex");

  it("the registry lists every extension clip file, and every copied file has its hash", () => {
    const files = listData("ext/clips").map((f) => f.replace("ext/clips/", "").replace(".json.gz", "")).sort();
    expect(reg.clips.map((c) => c.id).sort()).toEqual(files);
    for (const [rel, h] of Object.entries(reg.files)) expect(sha(rel), rel).toBe(h);
  });

  it("findings_ext.json parses and cites the three pinned results with their labels", () => {
    const f = FindingsExt.parse(readData("ext/findings_ext.json"));
    expect(f.sources.map((s) => s.key)).toEqual(["x1", "x3", "x2"]);
    expect(f.sources.find((s) => s.key === "x2")?.label).toBe(f.x2.study);
    expect(f.x3.label).toBe(f.sources.find((s) => s.key === "x3")?.label);
    expect(f.x1.label).toBe(f.sources.find((s) => s.key === "x1")?.label);
  });

  it("every rule-selected case resolves to registry clips of its own study", () => {
    const ids = new Set(reg.clips.map((c) => c.id));
    for (const c of sel.cases) {
      if (c.state_id === null) { expect(c.reason).toBeTruthy(); continue; }
      for (const id of c.clips ?? []) {
        expect(ids.has(id), id).toBe(true);
        expect(reg.clips.find((r) => r.id === id)?.study).toBe(c.role.split(":")[0]);
      }
    }
    for (const id of sel.verification_clips) expect(reg.clips.find((r) => r.id === id)?.has).toContain("reference_frames");
  });

  for (const rc of reg.clips) {
    it(`extension clip ${rc.id} parses, matches its registry entry and is labelled for its study`, () => {
      const c = ExtClip.parse(readData(`ext/clips/${rc.id}.json.gz`));
      expect([c.study, c.arm, c.state_id, c.seed]).toEqual([rc.study, rc.arm, rc.state_id, rc.seed]);
      expect(c.kind).toBe(`${rc.study}_record`);
      expect(c.frames.time_s).toHaveLength(c.frames.n);
      expect(c.actions.executed).toHaveLength(c.frames.n - 1);
      expect(c.outcome.steps).toBe(c.frames.n - 1);
      expect(c.outcome.success).toBe(rc.outcome.success);
      expect(c.provenance.physics_sha256).toBe(rc.physics_sha256);
      expect(c.provenance.eval_run_id).toBe(rc.eval_run_id);
      expect(c.verification.recomputed_vs_recorded.body_pos_max_abs_m).toBeLessThan(1e-9);
      expect(c.media.video.file.startsWith("ext/media/")).toBe(true);
      expect(existsSync(join(DATA, c.media.video.file)), c.media.video.file).toBe(true);
      expect(c.policy.label).toBe(rc.display_label);
      if (c.study === "x2") {
        // KI-065: the display label names Can MH; the recorded PH label survives as provenance only.
        for (const s of [c.label, c.policy.label, c.media.video.label]) expect(s).not.toMatch(/\bPH\b/);
        expect(c.policy.label).toMatch(/Can MH/);
        expect(c.provenance.recorded_policy_label).toBeTruthy();
      } else {
        expect(c.provenance.recorded_policy_label).toBeUndefined();
        expect(c.arm?.endsWith("+clamp")).toBe(/X1 action clamp/.test(c.policy.label));
      }
    });
  }
});
