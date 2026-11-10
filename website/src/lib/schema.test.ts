// Every exported data file parses against its runtime schema (MASTER_SPEC §8.5), and the clips
// satisfy the structural invariants: frame counts, monotonic timestamps, joint naming, no
// missing transforms, and registry entries that match the files.
import { describe, expect, it } from "vitest";
import { listData, readData, requireData } from "../test/fixtures";
import { Clip, Clouds, Denoise, Findings, Registry } from "./schema";

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
