// The articulated 3D viewer (MASTER_SPEC §8.3). It renders on demand: a frame is drawn only when
// the playback clock, the camera or an option changes, and not at all while the canvas is off
// screen, so several viewers on one page never run an unbounded render loop.
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import type { Clock } from "../lib/playback";
import type { Clip } from "../lib/schema";
import { presetPose, type PresetId } from "./cameras";
import type { MeshLibrary } from "./meshes";
import { buildScene, PALETTE, type BuiltScene } from "./robotScene";

export interface ViewerOptions {
  trail: boolean;
  labels: boolean;
  axes: boolean;
  successHeight: boolean;
  follow: boolean;
  autoOrbit?: boolean;
  /** Hide the reconstructed scene (keep the grid and overlays), e.g. to show a policy's input. */
  hideScene?: boolean;
}

export const DEFAULT_OPTIONS: ViewerOptions = { trail: true, labels: false, axes: false,
  successHeight: false, follow: false };

const LABEL_BODIES = ["robot0_link1", "robot0_link3", "robot0_link5", "robot0_link7",
  "gripper0_right_eef", "cube_main", "Can_main"];

function Stage({ clip }: { clip: Clip }) {
  const top = clip.success_rule.table_top_z_m ?? 0.8;
  const grids = useMemo(() => {
    const g = new THREE.Group();
    const floor = new THREE.GridHelper(8, 80, 0x3a3f47, 0x2a2e34);
    (floor.material as THREE.Material).transparent = true;
    (floor.material as THREE.Material).opacity = 0.9;
    g.add(floor);
    if (clip.task === "lift") {
      const fine = new THREE.GridHelper(0.8, 80, 0x4b5059, 0x41464e);
      (fine.material as THREE.Material).transparent = true;
      (fine.material as THREE.Material).opacity = 0.55;
      fine.position.y = top + 0.0006;
      g.add(fine);
      const coarse = new THREE.GridHelper(0.8, 8, 0x5c626b, 0x5c626b);
      coarse.position.y = top + 0.0008;
      g.add(coarse);
    }
    return g;
  }, [clip.task, top]);
  return (
    <>
      <color attach="background" args={[PALETTE.graphite]} />
      <fog attach="fog" args={[PALETTE.graphite, 3.2, 8]} />
      <hemisphereLight args={[0xf3ecdc, 0x15171a, 0.85]} />
      <directionalLight position={[2.2, 3.4, 1.6]} intensity={2.1} castShadow
        shadow-mapSize={[2048, 2048]} shadow-camera-left={-1.2} shadow-camera-right={1.2}
        shadow-camera-top={1.2} shadow-camera-bottom={-1.2} shadow-bias={-0.0004} />
      <directionalLight position={[-1.8, 1.6, -2.2]} intensity={0.55} color={0x9fd8f0} />
      <primitive object={grids} />
      <mesh rotation-x={-Math.PI / 2} position-y={-0.001} receiveShadow>
        <planeGeometry args={[12, 12]} />
        <meshStandardMaterial color={0x1b1e22} roughness={1} />
      </mesh>
    </>
  );
}

function Trail({ built, clip, clock, visible }: { built: BuiltScene; clip: Clip; clock: Clock;
  visible: boolean }) {
  const size = useThree((s) => s.size);
  const line = useMemo(() => {
    const pts: number[] = [];
    for (let k = 0; k < clip.frames.n; k++) {
      const p = built.bodyPos(clip.eef_body, k);
      pts.push(p.x, p.y, p.z);
    }
    const geo = new LineGeometry();
    geo.setPositions(pts);
    const mat = new LineMaterial({ color: PALETTE.signal, linewidth: 2.5, transparent: true,
      opacity: 0.95, depthTest: true });
    const l = new Line2(geo, mat);
    l.computeLineDistances();
    return l;
  }, [built, clip]);
  useEffect(() => {
    (line.material as LineMaterial).resolution.set(size.width, size.height);
  }, [line, size]);
  useFrame(() => {
    line.visible = visible;
    const k = Math.max(0, Math.floor(clock.frame));
    (line.geometry as LineGeometry).instanceCount = Math.max(0, k);
  });
  useEffect(() => () => { line.geometry.dispose(); (line.material as THREE.Material).dispose(); }, [line]);
  return <primitive object={line} />;
}

function SuccessHeight({ clip }: { clip: Clip }) {
  const z = clip.success_rule.z_threshold_m;
  if (z === undefined) return null;
  return (
    <group>
      <mesh position={[0, 0, z]}>
        <planeGeometry args={[0.36, 0.36]} />
        <meshBasicMaterial color={PALETTE.signal} transparent opacity={0.07} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <lineSegments position={[0, 0, z]}>
        <edgesGeometry args={[new THREE.PlaneGeometry(0.36, 0.36)]} />
        <lineBasicMaterial color={PALETTE.signal} transparent opacity={0.6} />
      </lineSegments>
    </group>
  );
}

function CameraRig({ clip, preset, follow, built, clock, autoOrbit, focus, focusOffset }: { clip: Clip; preset: PresetId;
  follow: boolean; built: BuiltScene; clock: Clock; autoOrbit?: boolean; focus?: number[]; focusOffset?: number[] }) {
  const { camera, gl, set, size, invalidate } = useThree();
  const controls = useRef<OrbitControls | null>(null);
  const persp = useMemo(() => new THREE.PerspectiveCamera(32, 1, 0.01, 40), []);
  const ortho = useMemo(() => new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 40), []);
  const focusRef = useRef(focus);
  focusRef.current = focus;
  const focusOffRef = useRef(focusOffset);
  focusOffRef.current = focusOffset;
  const lastFocus = useRef<THREE.Vector3 | null>(null);
  useEffect(() => {
    // A new focus point moves camera and target together, keeping the visitor's orbit angle.
    const c = controls.current;
    if (!c || !focus || !lastFocus.current) return;
    const t = new THREE.Vector3(focus[0], focus[2], -focus[1]);
    const d = t.clone().sub(lastFocus.current);
    c.target.add(d);
    c.object.position.add(d);
    c.update();
    lastFocus.current = t;
    invalidate();
  }, [focus?.[0], focus?.[1], focus?.[2], invalidate]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const pose = presetPose(preset, clip);
    const cam = pose.ortho ? ortho : persp;
    cam.position.copy(pose.position);
    cam.up.copy(pose.up);
    if (cam instanceof THREE.PerspectiveCamera) cam.fov = pose.fov;
    else cam.zoom = pose.zoom;
    set({ camera: cam });
    controls.current?.dispose();
    const c = new OrbitControls(cam, gl.domElement);
    c.target.copy(pose.target);
    c.enableDamping = false;
    c.minDistance = 0.25;
    c.maxDistance = 6;
    c.maxPolarAngle = Math.PI * 0.495;
    c.addEventListener("change", () => invalidate());
    if (focusRef.current) {
      const f = focusRef.current;
      const t = new THREE.Vector3(f[0], f[2], -f[1]);
      c.target.copy(t);
      cam.position.copy(t).add(focusOffRef.current ? new THREE.Vector3(...focusOffRef.current) : new THREE.Vector3(0.62, 0.5, 0.6).normalize().multiplyScalar(0.62));
      lastFocus.current = t.clone();
    }
    if (pose.quaternion && !focusRef.current) cam.quaternion.copy(pose.quaternion);
    else cam.lookAt(c.target);
    c.update();
    if (pose.quaternion && !focusRef.current) cam.quaternion.copy(pose.quaternion);
    controls.current = c;
    invalidate();
    return () => c.dispose();
  }, [preset, clip, gl, set, persp, ortho, invalidate, focusOffset?.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const a = size.width / Math.max(1, size.height);
    persp.aspect = a;
    persp.updateProjectionMatrix();
    ortho.left = -a; ortho.right = a; ortho.top = 1; ortho.bottom = -1;
    ortho.updateProjectionMatrix();
    invalidate();
  }, [size, persp, ortho, invalidate]);
  useFrame((_, delta) => {
    const c = controls.current;
    if (!c) return;
    if (follow) {
      const p = built.bodyPos(clip.eef_body, clock.frame);
      c.target.lerp(new THREE.Vector3(p.x, p.z, -p.y), 0.25);
      c.update();
    }
    if (autoOrbit) {
      const off = camera.position.clone().sub(c.target);
      off.applyAxisAngle(new THREE.Vector3(0, 1, 0), delta * 0.045);
      camera.position.copy(c.target).add(off);
      camera.lookAt(c.target);
    }
    camera.updateProjectionMatrix?.();
  });
  return null;
}

function Labels({ built, clock, show, layer }: { built: BuiltScene;
  clock: Clock; show: boolean; layer: React.RefObject<HTMLDivElement | null> }) {
  const { camera, size } = useThree();
  const names = useMemo(() => LABEL_BODIES.filter((n) => built.bodyIndex.has(n)), [built]);
  useEffect(() => {
    const el = layer.current;
    if (!el) return;
    el.replaceChildren(...names.map((n) => {
      const d = document.createElement("span");
      d.className = "viewer-label";
      d.textContent = n.replace("robot0_", "").replace("gripper0_right_", "").replace("_main", "");
      return d;
    }));
  }, [names, layer]);
  useFrame(() => {
    const el = layer.current;
    if (!el) return;
    el.style.display = show ? "block" : "none";
    if (!show) return;
    names.forEach((n, i) => {
      const p = built.bodyPos(n, clock.frame);
      const v = new THREE.Vector3(p.x, p.z, -p.y).project(camera);
      const s = el.children[i] as HTMLElement;
      s.style.transform = `translate(${((v.x + 1) / 2) * size.width}px, ${((1 - v.y) / 2) * size.height}px)`;
      s.style.visibility = v.z < 1 ? "visible" : "hidden";
    });
  });
  return null;
}

function ClipContent({ clip, lib, clock, preset, options, children, labelLayer, focus, focusOffset }: {
  clip: Clip; lib: MeshLibrary | null; clock: Clock; preset: PresetId; options: ViewerOptions;
  children?: ReactNode; labelLayer: React.RefObject<HTMLDivElement | null>; focus?: number[];
  focusOffset?: number[] }) {
  const built = useMemo(() => buildScene(clip, lib, "styled"), [clip, lib]);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => () => built.dispose(), [built]);
  useEffect(() => clock.subscribe(() => invalidate()), [clock, invalidate]);
  useEffect(() => { built.root.visible = !options.hideScene; invalidate(); }, [options, built, invalidate]);
  useFrame(() => built.apply(clock.frame));
  return (
    <>
      <Stage clip={clip} />
      <primitive object={built.root} />
      <group rotation-x={-Math.PI / 2}>
        <Trail built={built} clip={clip} clock={clock} visible={options.trail} />
        {options.successHeight && <SuccessHeight clip={clip} />}
        {options.axes && <axesHelper args={[0.25]} />}
        {children}
      </group>
      <CameraRig clip={clip} preset={preset} follow={options.follow} built={built} clock={clock}
        autoOrbit={options.autoOrbit} focus={focus} focusOffset={focusOffset} />
      <Labels built={built} clock={clock} show={options.labels} layer={labelLayer} />
    </>
  );
}

export function Viewer({ clip, lib, clock, preset = "orbit", options = DEFAULT_OPTIONS, children,
  ariaLabel, active = true, continuous = false, focus, focusOffset }: {
  clip: Clip; lib: MeshLibrary | null; clock: Clock; preset?: PresetId; options?: ViewerOptions;
  children?: ReactNode; ariaLabel: string; active?: boolean; continuous?: boolean;
  /** MuJoCo-frame point the camera orbits (moves with it, keeping the angle). */
  focus?: number[];
  /** three.js-frame camera offset from the focus point. */
  focusOffset?: number[] }) {
  const labelLayer = useRef<HTMLDivElement>(null);
  return (
    <div className="viewer-canvas" role="img" aria-label={ariaLabel}>
      <Canvas shadows="percentage" dpr={[1, 2]} frameloop={!active ? "never" : continuous ? "always" : "demand"}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        onCreated={({ gl }) => { gl.toneMapping = THREE.ACESFilmicToneMapping; gl.toneMappingExposure = 1.05; }}>
        <ClipContent clip={clip} lib={lib} clock={clock} preset={preset} options={options}
          labelLayer={labelLayer} focus={focus} focusOffset={focusOffset}>{children}</ClipContent>
      </Canvas>
      <div className="viewer-labels" ref={labelLayer} aria-hidden="true" />
    </div>
  );
}
