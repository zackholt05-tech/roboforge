// Viewer verification (MASTER_SPEC §8.3 "coordinate correctness"). For every exported reference
// frame, the browser renders the clip's flat-colour silhouette through the exact MuJoCo camera
// (pose and vertical field of view from the export) and compares it pixel by pixel with
// MuJoCo's own segmentation render of the same recorded state: robot IoU, task-object IoU and
// the robot silhouette's centroid offset. It also renders the styled viewer through the same
// camera for a review montage. `npm run verify` (e2e/verify.spec.ts) saves the result.
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { dataUrl, loadClip, loadMeshTable, loadRegistry } from "../lib/data";
import type { Clip } from "../lib/schema";
import { mujocoCameraPose } from "../three/cameras";
import { loadMeshLibrary, type MeshLibrary } from "../three/meshes";
import { buildScene } from "../three/robotScene";

export const THRESHOLD = { robot_iou_min: 0.95, object_iou_min: 0.85, centroid_px_max: 1.5 };

interface FrameResult { clip: string; frame: number; camera: string; size: number;
  robot_iou: number; object_iou: number | null; robot_centroid_px: number;
  pixels_mujoco: { robot: number; object: number }; pixels_viewer: { robot: number; object: number };
  pass: boolean }

async function loadMask(file: string, size: number): Promise<Uint8Array> {
  const img = new Image();
  img.src = dataUrl(`verification/${file}`);
  await img.decode();
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const g = c.getContext("2d", { willReadFrequently: true })!;
  g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, size, size).data;
  const out = new Uint8Array(size * size);
  for (let i = 0; i < out.length; i++) out[i] = d[4 * i];
  return out;
}

function classify(px: Uint8Array, size: number): Uint8Array {
  // Silhouette colours: robot white, object red, other blue, background black (exact, no AA).
  const out = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = 4 * ((size - 1 - y) * size + x); // readPixels is bottom-up
      const r = px[i], g = px[i + 1], b = px[i + 2];
      out[y * size + x] = r > 127 && g > 127 ? 1 : r > 127 ? 2 : b > 127 ? 3 : 0;
    }
  }
  return out;
}

function compare(a: Uint8Array, b: Uint8Array, size: number, cls: number) {
  let inter = 0, uni = 0, na = 0, nb = 0, ax = 0, ay = 0, bx = 0, by = 0;
  for (let i = 0; i < a.length; i++) {
    const pa = a[i] === cls, pb = b[i] === cls;
    if (pa && pb) inter++;
    if (pa || pb) uni++;
    if (pa) { na++; ax += i % size; ay += Math.floor(i / size); }
    if (pb) { nb++; bx += i % size; by += Math.floor(i / size); }
  }
  return { iou: uni ? inter / uni : 1, na, nb,
    centroid: na && nb ? Math.hypot(ax / na - bx / nb, ay / na - by / nb) : Infinity };
}

function renderTo(renderer: THREE.WebGLRenderer, clip: Clip, lib: MeshLibrary, frame: number,
  pose: Clip["cameras"][string], mode: "styled" | "silhouette") {
  const scene = new THREE.Scene();
  const built = buildScene(clip, lib, mode);
  built.apply(frame);
  scene.add(built.root);
  if (mode === "styled") {
    scene.background = new THREE.Color(0x1e2125);
    scene.add(new THREE.HemisphereLight(0xf3ecdc, 0x15171a, 1.1));
    const d = new THREE.DirectionalLight(0xffffff, 2.0);
    d.position.set(2.2, 3.4, 1.6);
    scene.add(d);
  } else scene.background = new THREE.Color(0x000000);
  const c = mujocoCameraPose(pose);
  const cam = new THREE.PerspectiveCamera(c.fov, 1, 0.005, 50);
  cam.position.copy(c.position);
  cam.quaternion.copy(c.quaternion);
  cam.updateMatrixWorld();
  renderer.render(scene, cam);
  built.dispose();
}

function Panel({ title, draw }: { title: string; draw: (ctx: CanvasRenderingContext2D) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => { const g = ref.current?.getContext("2d"); if (g) draw(g); }, [draw]);
  return <figure className="vpanel"><canvas ref={ref} width={240} height={240} /><figcaption>{title}</figcaption></figure>;
}

interface Row { res: FrameResult; mujoco: string; styled: string; diff: ImageData }

export function VerifyPage({ params }: { params: URLSearchParams }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [status, setStatus] = useState("loading");
  useEffect(() => {
    (async () => {
      const reg = await loadRegistry();
      await loadMeshTable();
      const lib = await loadMeshLibrary();
      const only = params.get("verify");
      const ids = reg.clips.filter((c) => c.has.includes("reference_frames") && (!only || only === "all" || c.id === only)).map((c) => c.id);
      if (!ids.length) throw new Error("no clip with reference frames in the exported data");
      const out: Row[] = [];
      for (const id of ids) {
        const clip = await loadClip(id);
        for (const rf of clip.reference_frames ?? []) {
          const size = rf.size;
          const canvas = document.createElement("canvas");
          canvas.width = size; canvas.height = size;
          const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
          renderer.setPixelRatio(1);
          renderer.setSize(size, size, false);
          renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
          renderTo(renderer, clip, lib, rf.frame, rf.camera_pose, "silhouette");
          const gl = renderer.getContext();
          const px = new Uint8Array(size * size * 4);
          gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, px);
          const ours = classify(px, size);
          renderer.outputColorSpace = THREE.SRGBColorSpace;
          renderer.toneMapping = THREE.ACESFilmicToneMapping;
          renderTo(renderer, clip, lib, rf.frame, rf.camera_pose, "styled");
          const styled = canvas.toDataURL("image/png");
          renderer.dispose();
          renderer.forceContextLoss();
          const theirs = await loadMask(rf.mask, size);
          const r = compare(theirs, ours, size, 1);
          const o = rf.pixels.object > 0 ? compare(theirs, ours, size, 2) : null;
          const diff = new ImageData(size, size);
          for (let i = 0; i < theirs.length; i++) {
            const a = theirs[i] === 1, b = ours[i] === 1, ao = theirs[i] === 2, bo = ours[i] === 2;
            const col = a && b ? [60, 200, 240] : a ? [255, 92, 110] : b ? [255, 214, 90] : ao && bo ? [238, 231, 215] : ao || bo ? [255, 140, 0] : [30, 33, 37];
            diff.data.set([...col, 255], 4 * i);
          }
          const res: FrameResult = { clip: id, frame: rf.frame, camera: rf.camera, size,
            robot_iou: r.iou, object_iou: o ? o.iou : null, robot_centroid_px: r.centroid,
            pixels_mujoco: { robot: r.na, object: o?.na ?? 0 }, pixels_viewer: { robot: r.nb, object: o?.nb ?? 0 },
            pass: r.iou >= THRESHOLD.robot_iou_min && r.centroid <= THRESHOLD.centroid_px_max &&
              (o === null || o.iou >= THRESHOLD.object_iou_min) };
          out.push({ res, mujoco: dataUrl(`verification/${rf.rgb}`), styled, diff });
          setRows([...out]);
        }
      }
      const doc = { schema: "roboforge.viewer_verification/1", threshold: THRESHOLD,
        method: "three.js silhouette (MeshBasicMaterial, no antialiasing, linear output) through the exported MuJoCo camera pose and fovy, vs the class mask of MuJoCo's segmentation render of the same recorded state (robosuite visual geom groups)",
        mesh_source: "website/public/assets (make web-assets, D-090)", registry_index_sha256: reg.index_sha256,
        user_agent: navigator.userAgent, frames: out.map((x) => x.res), pass: out.every((x) => x.res.pass) };
      (window as unknown as { __verification: unknown }).__verification = doc;
      setStatus(doc.pass ? "pass" : "fail");
    })().catch((e: Error) => { setStatus(`error: ${e.message}`); (window as unknown as { __verification: unknown }).__verification = { error: e.message }; });
  }, [params]);
  return (
    <main className="verify-page">
      <h1>Viewer verification: <span data-status={status}>{status}</span></h1>
      <p className="fine">Columns: MuJoCo render of the recorded state; the Observatory's styled viewer through the same camera; overlap of robot silhouettes (cyan both, red MuJoCo only, yellow viewer only; ivory and orange for the task object). Thresholds: robot IoU ≥ {THRESHOLD.robot_iou_min}, object IoU ≥ {THRESHOLD.object_iou_min}, robot centroid ≤ {THRESHOLD.centroid_px_max} px.</p>
      <div className="vgrid">
        {rows.map(({ res, mujoco, styled, diff }, i) => (
          <section key={i} className={res.pass ? "vrow" : "vrow bad"}>
            <h2>{res.clip}, frame {res.frame}, {res.camera}: robot IoU {res.robot_iou.toFixed(4)}, object IoU {res.object_iou?.toFixed(4) ?? "n/a"}, centroid {res.robot_centroid_px.toFixed(2)} px, {res.pass ? "pass" : "FAIL"}</h2>
            <div className="vpanels">
              <figure className="vpanel"><img src={mujoco} width={240} height={240} alt="MuJoCo render" /><figcaption>MuJoCo</figcaption></figure>
              <figure className="vpanel"><img src={styled} width={240} height={240} alt="Observatory viewer" /><figcaption>Observatory viewer</figcaption></figure>
              <Panel title="silhouette overlap" draw={(g) => { const c = document.createElement("canvas"); c.width = diff.width; c.height = diff.height; c.getContext("2d")!.putImageData(diff, 0, 0); g.drawImage(c, 0, 0, 240, 240); }} />
            </div>
          </section>
        ))}
      </div>
    </main>
  );
}

