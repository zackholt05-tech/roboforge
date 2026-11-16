// Camera presets. MuJoCo cameras look along their local −z with +y up, the three.js camera
// convention, so a MuJoCo camera pose converts like any body (D-030): position (x, y, z) →
// (x, z, −y) and q_three = q_C ⊗ q_mujoco with q_C a −90° rotation about x.
import * as THREE from "three";
import type { Clip } from "../lib/schema";
import { q3 } from "./robotScene";

export type PresetId = "hero" | "orbit" | "workspace" | "frontview" | "agentview" | "sideview" | "birdview" |
  "iso" | "top" | "side";

export const PRESETS: { id: PresetId; label: string; mujoco?: boolean; ortho?: boolean }[] = [
  { id: "workspace", label: "Workspace" },
  { id: "orbit", label: "Whole arm" },
  { id: "frontview", label: "MuJoCo frontview", mujoco: true },
  { id: "agentview", label: "MuJoCo agentview", mujoco: true },
  { id: "sideview", label: "MuJoCo sideview", mujoco: true },
  { id: "iso", label: "Isometric", ortho: true },
  { id: "top", label: "Top", ortho: true },
  { id: "side", label: "Side", ortho: true },
];

const Q_C = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);

export function mjToThreePos(p: [number, number, number] | number[]): THREE.Vector3 {
  return new THREE.Vector3(p[0], p[2], -p[1]);
}

export function mujocoCameraPose(pose: Clip["cameras"][string]): { position: THREE.Vector3;
  quaternion: THREE.Quaternion; fov: number } {
  const q = Q_C.clone().multiply(q3(...pose.quat_wxyz));
  return { position: mjToThreePos(pose.pos), quaternion: q, fov: pose.fovy_deg };
}

/** Workspace centre in three.js coordinates (above the table top). */
export function workspaceTarget(clip: Clip): THREE.Vector3 {
  const z = clip.success_rule.table_top_z_m ?? (clip.task === "can" ? 0.82 : 0.8);
  return clip.task === "can" ? new THREE.Vector3(0.05, z + 0.05, 0.0) : new THREE.Vector3(0, z + 0.06, 0);
}

export interface PresetPose { position: THREE.Vector3; target: THREE.Vector3; up: THREE.Vector3;
  fov: number; ortho: boolean; zoom: number; quaternion?: THREE.Quaternion }

/** `aspect` (canvas width / height) only moves the viewer's own presets back on narrow canvases;
 * the MuJoCo cameras always use their exported pose and field of view. */
export function presetPose(id: PresetId, clip: Clip, aspect = 1.6): PresetPose {
  const tgt = workspaceTarget(clip);
  const up = new THREE.Vector3(0, 1, 0);
  const cam = clip.cameras[id as string];
  if (cam && (id === "frontview" || id === "agentview" || id === "sideview" || id === "birdview")) {
    const c = mujocoCameraPose(cam);
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(c.quaternion);
    const target = c.position.clone().add(fwd.multiplyScalar(c.position.distanceTo(tgt)));
    return { position: c.position, target, up: new THREE.Vector3(0, 1, 0).applyQuaternion(c.quaternion),
      fov: c.fov, ortho: false, zoom: 1, quaternion: c.quaternion };
  }
  // Narrow canvases see less of the scene sideways, so the camera backs away along its view line.
  const back = Math.max(1, Math.min(1.7, 1.45 / aspect));
  const from = (target: THREE.Vector3, dir: [number, number, number], dist: number) =>
    target.clone().add(new THREE.Vector3(...dir).normalize().multiplyScalar(dist * back));
  const can = clip.task === "can";
  switch (id) {
    case "hero": {
      const narrow = typeof window !== "undefined" && window.innerWidth < 720;
      return { position: tgt.clone().add(narrow ? new THREE.Vector3(1.55, 0.75, -0.55) : new THREE.Vector3(1.7, 0.82, -1.2)),
        target: tgt.clone().add(narrow ? new THREE.Vector3(-0.3, 0.12, 0) : new THREE.Vector3(-0.3, 0.3, 0.72)), up, fov: 30, ortho: false, zoom: 1 };
    }
    case "workspace": {
      // Three-quarter view of the gripper, the object and where it has to go.
      const target = tgt.clone().add(new THREE.Vector3(can ? -0.02 : -0.08, can ? 0.1 : 0.12, 0));
      return { position: from(target, [1, 0.62, 0.55], can ? 1.75 : 1.25), target, up, fov: 32, ortho: false, zoom: 1 };
    }
    case "iso":
      return { position: tgt.clone().add(new THREE.Vector3(1.6, 1.25, 1.6)), target: tgt, up, fov: 30, ortho: true, zoom: 520 };
    case "top":
      return { position: tgt.clone().add(new THREE.Vector3(0, 2.4, 0.0001)), target: tgt, up: new THREE.Vector3(-1, 0, 0), fov: 30, ortho: true, zoom: 620 };
    case "side":
      return { position: tgt.clone().add(new THREE.Vector3(0, 0.15, 2.4)), target: tgt.clone().add(new THREE.Vector3(0, 0.15, 0)), up, fov: 30, ortho: true, zoom: 560 };
    default: {
      // The whole arm, from the shoulder to the table.
      const target = tgt.clone().add(new THREE.Vector3(-0.3, 0.4, 0));
      return { position: from(target, [1, 0.42, 0.8], 2.55), target, up, fov: 32, ortho: false, zoom: 1 };
    }
  }
}
