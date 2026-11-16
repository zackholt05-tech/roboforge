// Pieces shown around every recorded clip: provenance, the simulator's own video, the action
// channels and status messages. The MuJoCo video is ground truth rendered from the recorded
// states; the 3D view is a reconstruction of the same states.
import { useEffect, useRef } from "react";
import { dataUrl } from "../lib/data";
import { CATEGORY, FAMILY, PARTITION, short, startLabel, TASK, TASK_NOTE } from "../lib/format";
import { useClock, type Clock } from "../lib/playback";
import type { Clip } from "../lib/schema";

export function outcomeText(clip: Clip): string {
  const o = clip.outcome;
  if (o.success) return `success at step ${o.first_success_step} (${(o.first_success_step! * clip.control_dt_s).toFixed(2)} s)`;
  return `failure: ${o.stop_reason === "truncated_horizon" ? `no success within ${o.steps} steps` : o.stop_reason}`;
}

/** One line under every recorded clip: what it is, who controls the arm, the task, the outcome
 * and a short source. Everything else is in <ProvenanceDetails/>. */
export function ClipCaption({ clip, category, name, brief = false, explain = false }: { clip: Clip; category?: string | null; name?: string;
  /** Say the task in plain words after its name (where nothing else nearby does). */
  explain?: boolean;
  /** Inside a group whose header already names the policy and the outcome: only what the header lacks. */
  brief?: boolean }) {
  const run = clip.provenance.eval_run_id;
  return (
    <p className="clip-caption">
      <span className="kind recorded">recorded rollout</span>
      {!brief && (
        <>
          <span>MuJoCo simulation</span>
          <span>{name ?? FAMILY[clip.family] ?? clip.family}, seed {clip.seed}</span>
          <span>task: {TASK[clip.task]}{explain && TASK_NOTE[clip.task] ? ` (${TASK_NOTE[clip.task]})` : ""}</span>
        </>
      )}
      <span>{startLabel(clip.partition, clip.state_id)} {clip.state_id.split("_").pop()}</span>
      {!brief && (
        <>
          <span>trained checkpoint {clip.checkpoint.sha256 ? short(clip.checkpoint.sha256, 8) : "none (random)"}</span>
          <span className={clip.outcome.success ? "ok" : "bad"}>outcome: {outcomeText(clip)}</span>
        </>
      )}
      {category && category !== "success" && <span className="bad">{CATEGORY[category] ?? category}</span>}
      <span className="run-id" title={`the evaluation run this episode was recorded in: ${run}`}>source: {run.split("__")[0]}</span>
    </p>
  );
}

/** The full source of a clip, one click away: run id, checkpoint hash, start, record, physics
 * hash, simulator, platform and the recompute check (and, for X2, the recorded label, KI-065). */
export function ProvenanceDetails({ clip }: { clip: Clip }) {
  return (
    <details className="prov">
      <summary>Provenance</summary>
      <Provenance clip={clip} />
    </details>
  );
}

export function Provenance({ clip }: { clip: Clip }) {
  const p = clip.provenance;
  const rows: [string, string, string?][] = [
    ["What this is", clip.label],
    ["Policy", `${FAMILY[clip.family] ?? clip.family}, training seed ${clip.seed}`],
    ["Checkpoint", clip.checkpoint.sha256 ? `${short(clip.checkpoint.sha256, 16)} at step ${clip.checkpoint.step}` : "none (non-learned baseline)", clip.checkpoint.sha256 ?? undefined],
    ["Start state", startLabel(clip.partition, clip.state_id) === (PARTITION[clip.partition] ?? clip.partition)
      ? `${clip.state_id} (${PARTITION[clip.partition] ?? clip.partition})`
      : `${clip.state_id} (${startLabel(clip.partition, clip.state_id)}; recorded partition ${clip.partition}, D-123)`],
    ["source: run_id", p.eval_run_id, p.eval_run_id],
    ["Record", p.record, p.record],
    ["physics_sha256", short(p.physics_sha256, 16), p.physics_sha256],
    ["Simulator", `MuJoCo ${p.simulator.packages.mujoco}, robosuite ${p.simulator.packages.robosuite}, ${p.simulator.control_dt_s * 1000} ms control step`],
    ["Platform", p.platform],
    ["Recompute check", `|Δpos| ${clip.verification.recomputed_vs_recorded.body_pos_max_abs_m.toExponential(1)} m, geoms ${clip.verification.recomputed_vs_recorded.geom_pos_max_abs_m.toExponential(1)} m`],
  ];
  if (p.recorded_policy_label) {
    // KI-065: the frozen evaluator's label template names the PH demonstrations on X2 records.
    rows.splice(2, 0, ["Training data", clip.policy.label],
      ["Recorded evaluator label", `${p.recorded_policy_label} (KI-065: the label template is wrong for X2; the checkpoints were trained on Can MH)`]);
  }
  return (
    <dl className="provenance">
      {rows.map(([k, v, title]) => (
        <div key={k}><dt>{k}</dt><dd title={title}>{v}</dd></div>
      ))}
    </dl>
  );
}

export function InsetVideo({ clip, clock, large = false }: { clip: Clip; clock: Clock; large?: boolean }) {
  useClock(clock);
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const v = ref.current;
    if (!v || !Number.isFinite(v.duration)) return;
    const t = Math.min(clock.t, v.duration - 1e-3);
    if (clock.playing) {
      v.playbackRate = clock.speed;
      if (v.paused) void v.play().catch(() => {});
      if (Math.abs(v.currentTime - t) > 0.12) v.currentTime = t;
    } else {
      if (!v.paused) v.pause();
      if (Math.abs(v.currentTime - t) > 0.02) v.currentTime = t + 0.5 / clip.media.video.fps;
    }
  });
  return (
    <figure className={large ? "inset inset-large" : "inset"}>
      <video ref={ref} src={dataUrl(clip.media.video.file)} muted playsInline preload="metadata"
        width={clip.media.video.size} height={clip.media.video.size}
        aria-label={`MuJoCo ${clip.media.video.camera} render of the recorded states`} />
      <figcaption>The same episode in the simulator's own render (MuJoCo {clip.media.video.camera}, original colours)</figcaption>
    </figure>
  );
}

export const ACTIONS_NOTE = "Executed actions sent to the simulator (pose deltas and gripper). Recorded, not predicted.";

const DIM_COLOURS = ["#3cc8f0", "#3cc8f0", "#3cc8f0", "#b8b1a2", "#b8b1a2", "#b8b1a2", "#c9a86b"];

export function ActionChannels({ clip, clock, caption = true }: { clip: Clip; clock: Clock; caption?: boolean }) {
  useClock(clock);
  const W = 560, H = 18, gap = 6;
  const a = clip.actions.executed;
  const T = a.length;
  const k = Math.min(T, Math.round(clock.frame));
  const names = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"];
  return (
    <figure className="channels">
      <svg viewBox={`0 0 ${W + 64} ${7 * (H + gap)}`} role="img"
        aria-label={`Executed simulator actions over ${T} steps; 7 channels, normalized controller units in [-1, 1]`}>
        {names.map((nm, d) => {
          const y0 = d * (H + gap);
          const pts = a.map((r, t) => `${64 + (t / Math.max(1, T - 1)) * W},${y0 + H / 2 - (r[d] * H) / 2}`).join(" ");
          return (
            <g key={nm}>
              <text x={56} y={y0 + H / 2 + 4} textAnchor="end" className="axis-label">{nm}</text>
              <line x1={64} x2={64 + W} y1={y0 + H / 2} y2={y0 + H / 2} className="zero" />
              <polyline points={pts} fill="none" stroke={DIM_COLOURS[d]} strokeWidth={1.2} opacity={0.9} />
            </g>
          );
        })}
        <line x1={64 + (k / Math.max(1, T - 1)) * W} x2={64 + (k / Math.max(1, T - 1)) * W} y1={0}
          y2={7 * (H + gap)} className="playhead" />
      </svg>
      {caption && <figcaption>{ACTIONS_NOTE}</figcaption>}
    </figure>
  );
}

export function Status({ state, what }: { state: { status: string; error?: string }; what: string }) {
  if (state.status === "loading") return <div className="status" role="status">Loading {what}…</div>;
  if (state.status === "error") {
    return <div className="status status-error" role="alert"><strong>Could not load {what}.</strong> {state.error}</div>;
  }
  return null;
}
