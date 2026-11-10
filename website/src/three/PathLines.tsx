// Candidate end-effector paths drawn in the MuJoCo world frame (placed inside the viewer's
// rotated group). Every path is a clone-simulated end-effector trace from the export: the
// recorded episode is replayed bitwise to the decision frame, then the candidate chunk is run in
// the clone. Proposals other than the executed prefix never moved the recorded robot.
import { useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

export interface PathSpec { points: number[][]; color: number; width: number; opacity: number;
  dashed?: boolean }

function makeLine(p: PathSpec): Line2 | null {
  if (p.points.length < 2) return null;
  const g = new LineGeometry();
  g.setPositions(p.points.flatMap((q) => [q[0], q[1], q[2]]));
  const m = new LineMaterial({ color: p.color, linewidth: p.width, transparent: true,
    opacity: p.opacity, dashed: !!p.dashed, dashSize: 0.008, gapSize: 0.006, depthTest: false });
  const l = new Line2(g, m);
  l.computeLineDistances();
  l.renderOrder = 10; // drawn over the robot so a proposal is never hidden behind the gripper
  return l;
}

export function PathLines({ paths, markers = [] }: { paths: PathSpec[];
  markers?: { at: number[]; color: number; r: number }[] }) {
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  const lines = useMemo(() => paths.map(makeLine).filter((l): l is Line2 => l !== null), [paths]);
  useEffect(() => {
    lines.forEach((l) => (l.material as LineMaterial).resolution.set(size.width, size.height));
    invalidate();
  }, [lines, size, invalidate]);
  useEffect(() => () => lines.forEach((l) => { l.geometry.dispose(); (l.material as THREE.Material).dispose(); }), [lines]);
  return (
    <group>
      {lines.map((l, i) => <primitive key={i} object={l} />)}
      {markers.map((m, i) => (
        <mesh key={`m${i}`} position={[m.at[0], m.at[1], m.at[2]]} renderOrder={3}>
          <sphereGeometry args={[m.r, 20, 14]} />
          <meshBasicMaterial color={m.color} depthTest={false} transparent opacity={0.95} />
        </mesh>
      ))}
    </group>
  );
}
