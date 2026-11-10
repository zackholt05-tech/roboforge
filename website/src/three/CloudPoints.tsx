// The point-cloud policy's input, drawn where the policy saw it: a 512-point world-frame cloud
// recomputed from the recorded state at each decision frame by the frozen D-083 path (two 84²
// depth images + calibration, cropped and farthest-point sampled). Placed in the MuJoCo frame;
// the cloud shown is the one of the latest decision at or before the playback frame.
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { Clock } from "../lib/playback";
import type { Clouds } from "../lib/schema";

export const CLOUD_COLOURS = [0xd9824a, 0x3cc8f0];

export function CloudPoints({ clouds, clock, size = 3.5 }: { clouds: Clouds; clock: Clock; size?: number }) {
  const invalidate = useThree((s) => s.invalidate);
  const n = clouds.spec.n_points;
  const { geo, mat, pts } = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    // Screen-space size (pixels): a 512-point cloud stays legible at any zoom.
    const m = new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: false, transparent: true,
      opacity: 0.95, depthWrite: false, toneMapped: false });
    const p = new THREE.Points(g, m);
    p.renderOrder = 4;
    p.frustumCulled = false;
    return { geo: g, mat: m, pts: p };
  }, [n, size]);
  const shown = useRef(-1);
  const colours = useMemo(() => CLOUD_COLOURS.map((c) => new THREE.Color(c)), []);
  useFrame(() => {
    const k = Math.round(clock.frame);
    let j = 0;
    for (let i = 0; i < clouds.frames.length; i++) if (clouds.frames[i] <= k) j = i;
    if (j === shown.current) return;
    shown.current = j;
    const pos = geo.getAttribute("position") as THREE.BufferAttribute;
    const col = geo.getAttribute("color") as THREE.BufferAttribute;
    clouds.points[j].forEach((p, i) => {
      pos.setXYZ(i, p[0], p[1], p[2]);
      const c = colours[clouds.source_camera[j][i]] ?? colours[0];
      col.setXYZ(i, c.r, c.g, c.b);
    });
    pos.needsUpdate = true;
    col.needsUpdate = true;
    invalidate();
  });
  useEffect(() => () => { geo.dispose(); mat.dispose(); }, [geo, mat]);
  return <primitive object={pts} />;
}
