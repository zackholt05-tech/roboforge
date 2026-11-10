// Pieces shown around every recorded clip: provenance, the simulator's own video, the action
// channels and status messages. The MuJoCo video is ground truth rendered from the recorded
// states; the 3D view is a reconstruction of the same states.
import { useEffect, useRef } from "react";
import { dataUrl } from "../lib/data";
import { CATEGORY, FAMILY, PARTITION, short, TASK } from "../lib/format";
import { useClock, type Clock } from "../lib/playback";
import type { Clip } from "../lib/schema";

export function outcomeText(clip: Clip): string {
  const o = clip.outcome;
  if (o.success) return `success at step ${o.first_success_step} (${(o.first_success_step! * clip.control_dt_s).toFixed(2)} s)`;
  return `failure: ${o.stop_reason === "truncated_horizon" ? `no success within ${o.steps} steps` : o.stop_reason}`;
}

export function ClipCaption({ clip, category }: { clip: Clip; category?: string | null }) {
  return (
    <p className="clip-caption">
      <span>MuJoCo simulation</span>
      <span>trained checkpoint {clip.checkpoint.sha256 ? short(clip.checkpoint.sha256, 8) : "none (random)"}</span>
      <span>task: {TASK[clip.task]}</span>
      <span className={clip.outcome.success ? "ok" : "bad"}>outcome: {outcomeText(clip)}</span>
      {category && category !== "success" && <span className="bad">{CATEGORY[category] ?? category}</span>}
      <span className="run-id" title="the evaluation run this episode was recorded in">source: {clip.provenance.eval_run_id}</span>
    </p>
  );
}

export function Provenance({ clip }: { clip: Clip }) {
  const p = clip.provenance;
  const rows: [string, string, string?][] = [
    ["Policy", `${FAMILY[clip.family] ?? clip.family}, training seed ${clip.seed}`],
    ["Checkpoint", clip.checkpoint.sha256 ? `${short(clip.checkpoint.sha256, 16)} at step ${clip.checkpoint.step}` : "none (non-learned baseline)", clip.checkpoint.sha256 ?? undefined],
    ["Start state", `${clip.state_id} (${PARTITION[clip.partition] ?? clip.partition})`],
    ["source: run_id", p.eval_run_id, p.eval_run_id],
    ["Record", p.record, p.record],
    ["physics_sha256", short(p.physics_sha256, 16), p.physics_sha256],
    ["Simulator", `MuJoCo ${p.simulator.packages.mujoco}, robosuite ${p.simulator.packages.robosuite}, ${p.simulator.control_dt_s * 1000} ms control step`],
    ["Platform", p.platform],
    ["Recompute check", `|Δpos| ${clip.verification.recomputed_vs_recorded.body_pos_max_abs_m.toExponential(1)} m, geoms ${clip.verification.recomputed_vs_recorded.geom_pos_max_abs_m.toExponential(1)} m`],
  ];
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
      <figcaption>MuJoCo {clip.media.video.camera} render of the recorded states</figcaption>
    </figure>
  );
}

const DIM_COLOURS = ["#3cc8f0", "#3cc8f0", "#3cc8f0", "#b8b1a2", "#b8b1a2", "#b8b1a2", "#c9a86b"];

export function ActionChannels({ clip, clock }: { clip: Clip; clock: Clock }) {
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
      <figcaption>Executed actions sent to the simulator (OSC pose deltas and gripper, clipped to [−1, 1]). Recorded, not predicted.</figcaption>
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
