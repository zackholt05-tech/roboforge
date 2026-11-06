// Check an exported RoboForge clip with the real three.js library (DECISIONS D-030).
//
//   node check_clip.mjs <clip.json.gz> <result.json>
//
// 1. Root-group conversion: MuJoCo body poses placed verbatim under a Group rotated -90° about
//    x must have world poses equal to the clip's three.js-frame poses (Python conversion).
// 2. Camera projection: THREE.PerspectiveCamera + Vector3.project of the clip's verification
//    points must give the Python MuJoCo-frame pixel coordinates (which scripts/export_clip.py
//    checked against MuJoCo segmentation renders).
// 3. Articulated playback: an Object3D hierarchy driven only by the exported local rest
//    transforms and joint values must reproduce every body pose at every frame.
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";

const TOL = { conversion: 1e-12, projection_px: 1e-6, fk: 1e-9 };
const [clipPath, outPath] = process.argv.slice(2);
const clip = JSON.parse(gunzipSync(readFileSync(clipPath)).toString("utf8"));
const F = clip.frames;
const names = clip.bodies.names;

const quatErr = (a, b) => {
  const d1 = Math.max(...[0, 1, 2, 3].map((i) => Math.abs(a[i] - b[i])));
  const d2 = Math.max(...[0, 1, 2, 3].map((i) => Math.abs(a[i] + b[i])));
  return Math.min(d1, d2);
};
const vecErr = (a, b) => Math.max(...[0, 1, 2].map((i) => Math.abs(a[i] - b[i])));
const xyzw = (q) => [q.x, q.y, q.z, q.w];

// 1. root-group conversion
const root = new THREE.Group();
root.rotation.x = -Math.PI / 2;
const objs = names.map(() => { const o = new THREE.Object3D(); root.add(o); return o; });
let convPos = 0, convQuat = 0;
for (let k = 0; k < F.body_pos_mujoco.length; k++) {
  names.forEach((_, i) => {
    const [w, x, y, z] = F.body_quat_wxyz_mujoco[k][i];
    objs[i].position.fromArray(F.body_pos_mujoco[k][i]);
    objs[i].quaternion.set(x, y, z, w);
  });
  root.updateMatrixWorld(true);
  names.forEach((_, i) => {
    const p = objs[i].getWorldPosition(new THREE.Vector3()).toArray();
    const q = xyzw(objs[i].getWorldQuaternion(new THREE.Quaternion()));
    convPos = Math.max(convPos, vecErr(p, F.body_pos_three[k][i]));
    convQuat = Math.max(convQuat, quatErr(q, F.body_quat_xyzw_three[k][i]));
  });
}

// 2. camera projection
let projErr = 0;
for (const vp of clip.verification_points) {
  const c = vp.camera_three;
  const cam = new THREE.PerspectiveCamera(c.fov_deg, c.width / c.height, 0.01, 100);
  cam.position.fromArray(c.position);
  const [x, y, z, w] = c.quaternion_xyzw;
  cam.quaternion.set(x, y, z, w);
  cam.updateMatrixWorld(true);
  cam.updateProjectionMatrix();
  vp.three.forEach((p, i) => {
    const v = new THREE.Vector3().fromArray(p).project(cam);
    const px = [(v.x + 1) / 2 * c.width, (1 - v.y) / 2 * c.height];
    const ref = vp.pixels_mujoco_projection[i];
    projErr = Math.max(projErr, Math.abs(px[0] - ref[0]), Math.abs(px[1] - ref[1]));
  });
}

// 3. articulated playback from local rest transforms + joint values
const H = clip.bodies.hierarchy;
const fkRoot = new THREE.Group();
fkRoot.rotation.x = -Math.PI / 2;
const nodes = {};
for (const b of H) {
  nodes[b.name] = new THREE.Object3D();
  (b.parent ? nodes[b.parent] : fkRoot).add(nodes[b.name]);
}
const quat = ([w, x, y, z]) => new THREE.Quaternion(x, y, z, w);
let fkPos = 0, fkQuat = 0;
for (let k = 0; k < F.joint_qpos.length; k++) {
  for (const b of H) {
    const pos = new THREE.Vector3().fromArray(b.local_pos);
    const q = quat(b.local_quat_wxyz);
    for (const j of b.joints) {
      const v = F.joint_qpos[k][j.name];
      if (j.type === "free") {
        pos.fromArray(v.slice(0, 3));
        q.copy(quat(v.slice(3, 7))).normalize();
        continue;
      }
      const axis = new THREE.Vector3().fromArray(j.axis);
      const anchor = new THREE.Vector3().fromArray(j.pos);
      const delta = v[0] - j.qpos0[0];
      if (j.type === "hinge") {
        const before = anchor.clone().applyQuaternion(q);
        q.multiply(new THREE.Quaternion().setFromAxisAngle(axis, delta));
        pos.add(before).sub(anchor.clone().applyQuaternion(q));
      } else if (j.type === "slide") {
        pos.add(axis.clone().applyQuaternion(q).multiplyScalar(delta));
      } else {
        throw new Error(`unsupported joint type ${j.type}`);
      }
    }
    nodes[b.name].position.copy(pos);
    nodes[b.name].quaternion.copy(q);
  }
  fkRoot.updateMatrixWorld(true);
  names.forEach((n, i) => {
    const p = nodes[n].getWorldPosition(new THREE.Vector3()).toArray();
    const q = xyzw(nodes[n].getWorldQuaternion(new THREE.Quaternion()));
    fkPos = Math.max(fkPos, vecErr(p, F.body_pos_three[k][i]));
    fkQuat = Math.max(fkQuat, quatErr(q, F.body_quat_xyzw_three[k][i]));
  });
}

const result = {
  three_revision: THREE.REVISION, clip: clipPath, tolerances: TOL,
  conversion_max_pos_err: convPos, conversion_max_quat_err: convQuat,
  projection_max_px_err: projErr, fk_max_pos_err: fkPos, fk_max_quat_err: fkQuat,
  n_frames: F.body_pos_mujoco.length, n_bodies: names.length,
  n_projected_points: clip.verification_points.reduce((s, v) => s + v.three.length, 0),
};
result.checks = {
  root_group_conversion: convPos <= TOL.conversion && convQuat <= TOL.conversion,
  camera_projection_matches_mujoco: projErr <= TOL.projection_px,
  articulated_playback_matches_mujoco: fkPos <= TOL.fk && fkQuat <= TOL.fk,
};
writeFileSync(outPath, JSON.stringify(result, null, 1) + "\n");
for (const [k, v] of Object.entries(result.checks)) console.log(`[${v ? "PASS" : "FAIL"}] ${k}`);
console.log(JSON.stringify({ convPos, convQuat, projErr, fkPos, fkQuat, three: THREE.REVISION }));
process.exit(Object.values(result.checks).every(Boolean) ? 0 : 1);
