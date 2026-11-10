// Visual meshes of the Panda, gripper and task objects (D-090): compiled MuJoCo meshes of the
// pinned robosuite install, quantized to uint16 per mesh bounding box by `make web-assets` and
// never committed. Dequantization error is at most half a quantization step (listed per mesh in
// assets/meshes.json; at most about 7 µm for these meshes). Normals are recomputed with a 58° crease angle for shading
// only; they carry no geometric information.
import * as THREE from "three";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { loadMeshBlob, loadMeshTable } from "../lib/data";
import type { MeshTable } from "../lib/schema";

export interface MeshLibrary {
  geometry: Map<string, THREE.BufferGeometry>;
  edges: Map<string, THREE.BufferGeometry>;
  table: MeshTable;
}

let pending: Promise<MeshLibrary> | null = null;

export function decodeMesh(blob: ArrayBuffer, m: MeshTable["meshes"][string]): THREE.BufferGeometry {
  const q = new Uint16Array(blob, m.pos_offset, m.n_vert * 3);
  const pos = new Float32Array(m.n_vert * 3);
  for (let i = 0; i < m.n_vert; i++) {
    for (let a = 0; a < 3; a++) pos[3 * i + a] = m.lo[a] + (q[3 * i + a] / 65535) * m.span[a];
  }
  const idx = m.index_type === "uint16"
    ? new Uint16Array(blob, m.index_offset, m.n_face * 3)
    : new Uint32Array(blob, m.index_offset, m.n_face * 3);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

export function loadMeshLibrary(): Promise<MeshLibrary> {
  if (!pending) {
    pending = Promise.all([loadMeshTable(), loadMeshBlob()]).then(([table, blob]) => {
      if (blob.byteLength !== table.bytes) {
        throw new Error(`assets/meshes.bin has ${blob.byteLength} bytes, meshes.json says ${table.bytes}`);
      }
      const geometry = new Map<string, THREE.BufferGeometry>();
      for (const [name, m] of Object.entries(table.meshes)) {
        const raw = decodeMesh(blob, m);
        const shaded = toCreasedNormals(raw, (58 * Math.PI) / 180);
        shaded.computeBoundingSphere();
        geometry.set(name, shaded);
        geometry.set(`${name}#raw`, raw);
      }
      return { geometry, edges: new Map(), table };
    });
    pending.catch(() => { pending = null; });
  }
  return pending;
}

/** Feature edges of a mesh (angle > 38°), cached: the "industrial linework". */
export function meshEdges(lib: MeshLibrary, name: string): THREE.BufferGeometry | null {
  if (!lib.edges.has(name)) {
    const raw = lib.geometry.get(`${name}#raw`);
    if (!raw) return null;
    lib.edges.set(name, new THREE.EdgesGeometry(raw, 38));
  }
  return lib.edges.get(name)!;
}
