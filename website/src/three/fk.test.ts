// Articulated playback check (MASTER_SPEC §8.3): rebuild every body's world pose from the
// recorded joint angles through the exported kinematic tree with three.js, and compare it with
// the recorded world transform the viewer draws. A wrong joint axis, anchor, parent or
// quaternion order shows up here as a pose error.
import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { Clip, Registry } from "../lib/schema";
import { readData, requireData } from "../test/fixtures";
import { forwardKinematics, q3 } from "./robotScene";

describe("forward kinematics reproduces the recorded poses", () => {
  requireData();
  const reg = Registry.parse(readData("registry.json"));
  const ids = [...new Set(["lift", "can"].flatMap((t) => reg.clips.filter((c) => c.task === t).slice(0, 2).map((c) => c.id)))];
  for (const id of ids) {
    it(id, () => {
      const c = Clip.parse(readData(`clips/${id}.json.gz`));
      let worstP = 0, worstQ = 0;
      const step = Math.max(1, Math.floor(c.frames.n / 12));
      for (let k = 0; k < c.frames.n; k += step) {
        const fk = forwardKinematics(c, k);
        c.frames.dynamic_index.forEach((bi, j) => {
          const name = c.bodies[bi].name;
          const got = fk.get(name);
          if (!got) throw new Error(`body ${name} missing from the kinematic tree`);
          const p = c.frames.pos[k][j], q = c.frames.quat_wxyz[k][j];
          worstP = Math.max(worstP, got.p.distanceTo(new THREE.Vector3(p[0], p[1], p[2])));
          // Exported quaternions are rounded to 6 decimals, so normalize before comparing angles.
          worstQ = Math.max(worstQ, got.q.clone().normalize().angleTo(q3(q[0], q[1], q[2], q[3]).normalize()));
        });
      }
      expect(worstP).toBeLessThan(1e-4);
      expect(worstQ).toBeLessThan(1e-3);
    });
  }
});
