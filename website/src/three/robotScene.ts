// Builds the three.js scene of one recorded clip and poses it per frame (MASTER_SPEC §8.3).
//
// Coordinates (DECISIONS D-030): everything lives under one root Group rotated −90° about x, so
// MuJoCo world positions and (w, x, y, z) quaternions are used verbatim inside it (reordered to
// three.js's (x, y, z, w)). Each body is posed with its *recorded* world transform; each visual
// geom keeps its MuJoCo pose relative to its body (geom_pos, geom_quat). The same builder serves
// the styled viewer and the flat-colour silhouette renderer that `make web-verify` compares with
// MuJoCo's segmentation renders.
//
// Visual approximations (listed on the site): between two recorded frames poses are linearly
// interpolated (slerp for orientations) when playback runs between frames; colours and
// materials are restyled; textures are omitted; the floor and walls are replaced by a measured
// grid; a mesh missing from assets/ is drawn as its bounding box and flagged.
import * as THREE from "three";
import type { Clip } from "../lib/schema";
import { meshEdges, type MeshLibrary } from "./meshes";

export type RenderMode = "styled" | "silhouette";

export const PALETTE = {
  graphite: 0x1e2125,
  ivoryShell: 0xe6dfcf,
  darkShell: 0x3a3e45,
  pedestal: 0x2b2e33,
  object: 0xd8ae5a,
  table: 0x353940,
  wood: 0x4a443c,
  signal: 0x3cc8f0,
  error: 0xff5c6e,
  sand: 0xc9a86b,
  line: 0x111316,
};

export const CLASS_COLOUR = { robot: 0xffffff, object: 0xff0000, other: 0x0000ff };

export interface BuiltScene {
  root: THREE.Group;
  nodes: THREE.Object3D[];
  bodyIndex: Map<string, number>;
  apply(frame: number): void;
  /** MuJoCo-frame world position of a body at (fractional) frame f. */
  bodyPos(name: string, f: number): THREE.Vector3;
  approximate: string[];
  dispose(): void;
}

const tmpQ0 = new THREE.Quaternion();
const tmpQ1 = new THREE.Quaternion();

export const q3 = (w: number, x: number, y: number, z: number) => new THREE.Quaternion(x, y, z, w);

function primitive(g: Clip["geoms"][number]): THREE.BufferGeometry {
  const [a, b, c] = g.size;
  switch (g.kind) {
    case "box":
      return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    case "sphere":
      return new THREE.SphereGeometry(a, 24, 16);
    case "ellipsoid":
      return new THREE.SphereGeometry(1, 24, 16).scale(a, b, c);
    case "cylinder": // MuJoCo: radius, half-length along local z; three.js: along y
      return new THREE.CylinderGeometry(a, a, 2 * b, 32).rotateX(Math.PI / 2);
    case "capsule":
      return new THREE.CapsuleGeometry(a, 2 * b, 8, 24).rotateX(Math.PI / 2);
    default:
      throw new Error(`no primitive for ${g.kind}`);
  }
}

function styledMaterial(role: Clip["geoms"][number]["role"]): THREE.Material {
  const std = (color: number, roughness: number, metalness = 0, extra = {}) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness, ...extra });
  switch (role) {
    case "robot_light": return std(PALETTE.ivoryShell, 0.62, 0.02);
    case "robot_dark": return std(PALETTE.darkShell, 0.48, 0.25);
    case "robot_accent": return std(PALETTE.signal, 0.4, 0, { emissive: PALETTE.signal, emissiveIntensity: 0.35 });
    case "pedestal": return std(PALETTE.pedestal, 0.7, 0.1);
    case "object": return std(PALETTE.object, 0.42, 0.05);
    case "table": return std(PALETTE.table, 0.9, 0);
    case "fixture": return std(PALETTE.wood, 0.85, 0);
    case "target": return std(0xeee7d7, 0.6, 0, { transparent: true, opacity: 0.07, depthWrite: false });
  }
}

function classOf(category: string): keyof typeof CLASS_COLOUR {
  return category === "robot" ? "robot" : category === "free_object" ? "object" : "other";
}

export function buildScene(clip: Clip, lib: MeshLibrary | null, mode: RenderMode): BuiltScene {
  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2;
  root.name = `clip:${clip.id}`;
  const bodyIndex = new Map(clip.bodies.map((b, i) => [b.name, i] as const));
  const nodes = clip.bodies.map((b, i) => {
    const o = new THREE.Object3D();
    o.name = b.name;
    const p = clip.frames.static_pos[i];
    const q = clip.frames.static_quat_wxyz[i];
    o.position.set(p[0], p[1], p[2]);
    o.quaternion.copy(q3(q[0], q[1], q[2], q[3]));
    root.add(o);
    return o;
  });
  const approximate: string[] = [];
  const disposables: { dispose(): void }[] = [];
  const classMats = Object.fromEntries(Object.entries(CLASS_COLOUR).map(([k, c]) =>
    [k, new THREE.MeshBasicMaterial({ color: c, toneMapped: false })])) as
    Record<keyof typeof CLASS_COLOUR, THREE.MeshBasicMaterial>;
  const darkLine = new THREE.LineBasicMaterial({ color: PALETTE.line, transparent: true, opacity: 0.55 });
  const lightLine = new THREE.LineBasicMaterial({ color: 0xeee7d7, transparent: true, opacity: 0.16 });
  disposables.push(...Object.values(classMats), darkLine, lightLine);
  for (const g of clip.geoms) {
    if (g.rgba[3] <= 0) continue;
    const bi = bodyIndex.get(g.body);
    if (bi === undefined) continue;
    let geom: THREE.BufferGeometry | undefined;
    let edges: THREE.BufferGeometry | null = null;
    if (g.kind === "mesh") {
      geom = lib?.geometry.get(g.mesh!);
      if (geom && lib) edges = meshEdges(lib, g.mesh!);
      if (!geom) {
        const s = clip.meshes[g.mesh!] as unknown as { vert_min?: number[]; vert_max?: number[] };
        const lo = s.vert_min ?? [-0.01, -0.01, -0.01];
        const hi = s.vert_max ?? [0.01, 0.01, 0.01];
        geom = new THREE.BoxGeometry(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
          .translate((hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, (hi[2] + lo[2]) / 2);
        disposables.push(geom);
        approximate.push(g.mesh!);
      }
    } else {
      geom = primitive(g);
      disposables.push(geom);
      if (mode === "styled") {
        edges = new THREE.EdgesGeometry(geom, 30);
        disposables.push(edges);
      }
    }
    const cat = clip.bodies[bi].category;
    const mat = mode === "silhouette" ? classMats[classOf(cat)] : styledMaterial(g.role);
    if (mode === "styled") disposables.push(mat);
    const mesh = new THREE.Mesh(geom, mat);
    mesh.name = g.name ?? `${g.body}:${g.kind}`;
    mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
    mesh.quaternion.copy(q3(g.quat_wxyz[0], g.quat_wxyz[1], g.quat_wxyz[2], g.quat_wxyz[3]));
    mesh.castShadow = mode === "styled" && g.role !== "table" && g.role !== "target";
    mesh.receiveShadow = mode === "styled";
    mesh.userData = { role: g.role, cls: classOf(cat) };
    if (mode === "styled" && edges && g.role !== "target") {
      const light = g.role === "robot_dark" || g.role === "pedestal" || g.role === "table" || g.role === "fixture";
      const ls = new THREE.LineSegments(edges, light ? lightLine : darkLine);
      ls.raycast = () => {};
      mesh.add(ls);
    }
    nodes[bi].add(mesh);
  }
  const dyn = clip.frames.dynamic_index;
  const n = clip.frames.n;
  const apply = (f: number) => {
    const ff = Math.max(0, Math.min(n - 1, f));
    const k0 = Math.floor(ff);
    const k1 = Math.min(n - 1, k0 + 1);
    const a = ff - k0;
    const P0 = clip.frames.pos[k0], P1 = clip.frames.pos[k1];
    const Q0 = clip.frames.quat_wxyz[k0], Q1 = clip.frames.quat_wxyz[k1];
    for (let j = 0; j < dyn.length; j++) {
      const o = nodes[dyn[j]];
      const p0 = P0[j], p1 = P1[j];
      o.position.set(p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]), p0[2] + a * (p1[2] - p0[2]));
      const q0 = Q0[j], q1 = Q1[j];
      tmpQ0.set(q0[1], q0[2], q0[3], q0[0]);
      if (a === 0) o.quaternion.copy(tmpQ0);
      else {
        tmpQ1.set(q1[1], q1[2], q1[3], q1[0]);
        o.quaternion.slerpQuaternions(tmpQ0, tmpQ1, a);
      }
    }
  };
  const dynSlot = new Map(dyn.map((bi, j) => [bi, j] as const));
  const bodyPos = (name: string, f: number) => {
    const bi = bodyIndex.get(name)!;
    const j = dynSlot.get(bi);
    if (j === undefined) return new THREE.Vector3(...clip.frames.static_pos[bi]);
    const ff = Math.max(0, Math.min(n - 1, f));
    const k0 = Math.floor(ff), k1 = Math.min(n - 1, k0 + 1), a = ff - k0;
    const p0 = clip.frames.pos[k0][j], p1 = clip.frames.pos[k1][j];
    return new THREE.Vector3(p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]), p0[2] + a * (p1[2] - p0[2]));
  };
  apply(0);
  return {
    root, nodes, bodyIndex, apply, bodyPos, approximate,
    dispose: () => disposables.forEach((d) => d.dispose()),
  };
}

/** Body world poses from the exported hierarchy and joint values (forward kinematics, the
 * articulated-playback check of tests/web_contract and src/three/fk.test.ts). */
export function forwardKinematics(clip: Clip, frame: number): Map<string, { p: THREE.Vector3; q: THREE.Quaternion }> {
  const out = new Map<string, { p: THREE.Vector3; q: THREE.Quaternion }>();
  const jq = new Map(clip.joints.map((n, i) => [n, clip.frames.joint_qpos[frame][i]] as const));
  for (const b of clip.hierarchy) {
    const parent = b.parent ? out.get(b.parent) : undefined;
    const pp = parent ? parent.p : new THREE.Vector3();
    const pq = parent ? parent.q : new THREE.Quaternion();
    const p = new THREE.Vector3(...b.local_pos).applyQuaternion(pq).add(pp);
    const q = pq.clone().multiply(q3(...b.local_quat_wxyz));
    for (const j of b.joints) {
      if (j.type === "free") {
        const bi = clip.bodies.findIndex((x) => x.name === b.name);
        const slot = clip.frames.dynamic_index.indexOf(bi);
        const fp = clip.frames.pos[frame][slot], fq = clip.frames.quat_wxyz[frame][slot];
        p.set(fp[0], fp[1], fp[2]);
        q.copy(q3(fq[0], fq[1], fq[2], fq[3]));
        continue;
      }
      const v = jq.get(j.name);
      if (v === undefined) throw new Error(`joint ${j.name} has no recorded value`);
      const axis = new THREE.Vector3(...j.axis);
      const anchor = new THREE.Vector3(...j.pos);
      const delta = v - j.qpos0[0];
      if (j.type === "hinge") {
        const before = anchor.clone().applyQuaternion(q);
        q.multiply(new THREE.Quaternion().setFromAxisAngle(axis, delta));
        p.add(before).sub(anchor.clone().applyQuaternion(q));
      } else if (j.type === "slide") {
        p.add(axis.applyQuaternion(q).multiplyScalar(delta));
      } else throw new Error(`unsupported joint ${j.type}`);
    }
    out.set(b.name, { p, q });
  }
  return out;
}
