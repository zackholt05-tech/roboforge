// H. Methods and credits: the experimental design in brief, how the Observatory was built and
// checked (including every visual approximation), and the third-party work it rests on.
import { Status } from "../components/ClipParts";
import { loadFindings, loadVerification } from "../lib/data";
import { short } from "../lib/format";
import { useAsync } from "../lib/hooks";
import { useRegistry } from "./registry";

export const APPROXIMATIONS = [
  "Poses between two recorded 20 Hz frames are interpolated (linear for positions, slerp for orientations) when playback runs slower or faster than real time; paused frames show the recorded pose exactly.",
  "Colours and materials are restyled (matte shells, linework); MuJoCo textures are not used.",
  "The floor and walls are replaced by a metric grid (major lines every 10 cm on the Lift table, 1 cm minor).",
  "Lighting and shadows are the viewer's own, not MuJoCo's.",
  "Robot meshes are extracted from the installed robosuite package at build time (D-090) and quantized to 16 bits per axis within each mesh's bounding box (largest vertex error about 0.007 mm, measured when the mesh file is written). If they are missing, links are drawn as bounding boxes and the viewer says so.",
  "Candidate paths in Inside the policy are end-effector traces from a cloned simulator, not kinematic guesses; intermediate sampler states are not plans the policy would execute. They are drawn on top of the robot, never hidden by it, so their depth order is not physical.",
  "Point clouds are the policy's own input, recomputed from the recorded state at each decision frame and held until the next decision; each point is drawn as a dot of fixed screen size.",
];

const REFS: [string, string, string][] = [
  ["robomimic: What Matters in Learning from Offline Human Demonstrations for Robot Manipulation", "Mandlekar et al., CoRL 2021", "https://arxiv.org/abs/2108.03298"],
  ["robosuite: A Modular Simulation Framework and Benchmark for Robot Learning", "Zhu et al., 2020", "https://arxiv.org/abs/2009.12293"],
  ["MuJoCo: A physics engine for model-based control", "Todorov, Erez and Tassa, IROS 2012", "https://doi.org/10.1109/IROS.2012.6386109"],
  ["Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", "Chi et al., RSS 2023; IJRR 2024", "https://arxiv.org/abs/2303.04137"],
  ["3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations", "Ze et al., RSS 2024", "https://arxiv.org/abs/2403.03954"],
];

function Verification() {
  const reg = useRegistry();
  const v = useAsync(loadVerification, []);
  const rec = reg.clips.map((c) => c.verification?.body_pos_max_abs_m)
    .filter((x): x is number => typeof x === "number");
  return (
    <div className="verify">
      <h3>How the 3D view was checked</h3>
      <ul className="prose checks">
        <li>Every body pose is recomputed from the saved simulator state on the canonical container and compared with the pose recorded during evaluation{rec.length ? `: largest difference ${Math.max(...rec).toExponential(1)} m over ${rec.length} clips` : ""}. Held-out episodes are re-rendered from their states, never re-run.</li>
        <li>The browser poses each link from its recorded world transform. A unit test also rebuilds the poses from the joint angles through the exported kinematic tree (three.js forward kinematics) on about 12 frames of two clips per task, and checks that they agree within 0.1 mm and 0.06°.</li>
        <li>The silhouette of the 3D reconstruction is rendered through each MuJoCo camera and compared with MuJoCo's own segmentation render of the same state:</li>
      </ul>
      {v.status === "ready" && v.value ? (
        <div className="table-scroll">
          <table className="verify-table">
            <caption>Viewer vs MuJoCo reference frames (thresholds: robot IoU ≥ {v.value.threshold.robot_iou_min}, object IoU ≥ {v.value.threshold.object_iou_min}, robot centroid within {v.value.threshold.centroid_px_max} px). Overall: {v.value.pass ? "pass" : "FAIL"}.</caption>
            <thead><tr><th scope="col">Clip</th><th scope="col">Frame</th><th scope="col">Camera</th><th scope="col">Robot IoU</th><th scope="col">Object IoU</th><th scope="col">Centroid error</th><th scope="col">Result</th></tr></thead>
            <tbody>{v.value.frames.map((f, i) => (
              <tr key={i}><td>{f.clip}</td><td className="num">{f.frame}</td><td>{f.camera}</td><td className="num">{f.robot_iou.toFixed(3)}</td>
                <td className="num">{f.object_iou === null ? "n/a" : f.object_iou.toFixed(3)}</td><td className="num">{f.robot_centroid_px.toFixed(1)} px</td>
                <td className={f.pass ? "ok" : "bad"}>{f.pass ? "pass" : "fail"}</td></tr>
            ))}</tbody>
          </table>
        </div>
      ) : v.status === "loading" ? <Status state={v} what="the viewer verification" /> : (
        <p className="status">The viewer verification has not been run for this build (`make web-verify`).</p>
      )}
      <h3>Visual approximations</h3>
      <ul className="prose checks">{APPROXIMATIONS.map((a) => <li key={a}>{a}</li>)}</ul>
    </div>
  );
}

export default function Methods() {
  const reg = useRegistry();
  const f = useAsync(loadFindings, []);
  return (
    <section id="methods" aria-labelledby="methods-title">
      <div className="wrap">
        <div className="section-head">
          <h2 id="methods-title">Methods and credits</h2>
        </div>
        <div className="grid-2 methods">
          <div className="prose">
            <h3>Design</h3>
            <p>Two robosuite tasks, Lift (pick up a cube) and Can (place a can in its bin), with a simulated
              Franka Panda under operational-space control at 20 Hz. Each policy learns from the robomimic
              proficient-human demonstrations: 180 training and 20 validation episodes per task, split by whole episode.</p>
            <p>Four learned families (diffusion U-Net, BC-MLP, BC-RNN, chunked BC-MLP) were trained with three
              seeds each, next to a random-action baseline. Checkpoints and settings were chosen on validation starts only. The 50 held-out
              starts per task were run once, under a frozen protocol, after every choice was recorded; the
              intervals resample seeds, then starts (hierarchical paired bootstrap, B = 10,000).</p>
            <p>The RGB study and the Level 2 point-cloud study followed the same pattern with their own frozen
              protocols. No result on this page was re-run for the site.</p>
            <h3>How this site was made</h3>
            <p>Every rollout is a recorded evaluation episode, reconstructed from the simulator states saved at
              evaluation time and re-derived on the canonical container ({reg.export_platform.platform_id}).
              Export commit {short(reg.export_git.sha, 10)}{reg.export_git.dirty ? " (working tree not clean: development data)" : ""},
              index {short(reg.index_sha256, 12)}. Episode choices follow written rules (DECISIONS D-089), not taste.
              The only new policy calls are the denoising traces in Inside the policy, replayed on validation
              starts and labelled as an export.</p>
            {f.status === "ready" && (
              <p className="fine">Frozen sources: {f.value.sources.map((s) => `${s.path} (${short(s.sha256, 8)})`).join("; ")}.</p>
            )}
          </div>
          <Verification />
        </div>
        <div className="credits">
          <div className="prose">
            <h3>Credits</h3>
            <p><strong>RoboForge and this Observatory: Zackary Holt</strong>, sole author.</p>
            <p>The work stands on others': the robomimic datasets and framework (Stanford Vision and Learning
              Lab), the robosuite simulation framework and its Franka Panda, cube and can assets (MIT), the MuJoCo
              physics engine (Google DeepMind, Apache-2.0), the Diffusion Policy method of Chi et al., whose
              conditional U-Net RoboForge adapts through robomimic (MIT), the diffusers schedulers re-implemented
              for recording (Apache-2.0), and the DP3 paper that inspired the point-cloud arm (no DP3 code is used).
              The site is built with React, three.js and React Three Fiber.</p>
            <p className="fine">robosuite is © 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab,
              released under the MIT License. Robot meshes are not redistributed with this repository; they are read
              from the installed package when the site is built. Full notices: THIRD_PARTY_NOTICES.md.</p>
          </div>
          <ol className="refs">
            {REFS.map(([t, a, u]) => <li key={u}><a href={u} rel="noreferrer">{t}</a><span className="dim">{a}</span></li>)}
            <li><a href="https://huggingface.co/datasets/robomimic/robomimic_datasets" rel="noreferrer">robomimic datasets (robosuite v1.5 repackaging)</a><span className="dim">Hugging Face, revision 74fa018</span></li>
          </ol>
        </div>
      </div>
    </section>
  );
}
