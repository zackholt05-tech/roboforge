// E. Inside the policy: real DDIM-10 denoising traces of the diffusion policy, recomputed by
// replaying recorded *validation* episodes bitwise (D-089 rule 4). Never held-out starts, never a
// new result. Every candidate path is a clone simulation of a chunk the robot never executed,
// except the executed prefix, which reproduces the recorded frames bitwise.
import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Status } from "../components/ClipParts";
import { decisionFrames, useClip, ViewerOrFallback } from "../components/ClipStage";
import { Transport } from "../components/Transport";
import { loadDenoise } from "../lib/data";
import { TASK } from "../lib/format";
import { useAsync, useClockFor } from "../lib/hooks";
import { prefersReducedMotion, useClock, type Clock } from "../lib/playback";
import type { Clip, Decision, Denoise } from "../lib/schema";
import { PathLines, type PathSpec } from "../three/PathLines";
import { PALETTE } from "../three/robotScene";
import { DEFAULT_OPTIONS } from "../three/Viewer";
import { useRegistry } from "./registry";

const DIMS = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"];
// Camera offsets from the decision's path centre, in three.js axes (y up), metres.
const VIEWS: [string, string, number[] | null][] = [
  ["above", "Close, from above", [0.2, 0.62, 0.26]],
  ["side", "Close, from the side", [0.15, 0.12, 0.62]],
  ["wide", "Whole scene", null],
];

const EPISODES = [
  ["lift", "denoise:first_success", "Lift, first validation success"],
  ["can", "denoise:first_success", "Can, first validation success"],
  ["can", "denoise:first_failure", "Can, first validation failure"],
] as const;

function useDecisionFromClock(clock: Clock, frames: number[]): number {
  useClock(clock);
  const k = Math.round(clock.frame);
  let i = 0;
  for (let j = 0; j < frames.length; j++) if (frames[j] <= k) i = j;
  return i;
}

function paths(d: Decision, stage: number, showDiversity: boolean): PathSpec[] {
  const out: PathSpec[] = [];
  if (showDiversity) {
    for (const p of d.paths.diversity) out.push({ points: p, color: PALETTE.sand, width: 1.2, opacity: 0.38 });
  }
  if (stage < d.timesteps.length) {
    out.push({ points: d.paths.sample_in[stage], color: PALETTE.signal, width: 1.4, opacity: 0.55, dashed: true });
    out.push({ points: d.paths.x0[stage], color: PALETTE.signal, width: 2.4, opacity: 0.9 });
  } else {
    const ex = d.n_executed + 1;
    out.push({ points: d.paths.final.slice(ex - 1), color: PALETTE.signal, width: 2, opacity: 0.5, dashed: true });
    out.push({ points: d.paths.final.slice(0, ex), color: PALETTE.signal, width: 4, opacity: 1 });
  }
  out.push({ points: d.paths.recorded_executed, color: 0xeee7d7, width: 3, opacity: 0.95 });
  return out;
}

const ChunkChart = memo(function ChunkChart({ d, stage, horizon }: { d: Decision; stage: number; horizon: number }) {
  const W = 164, H = 74, padL = 8, padT = 16, cols = 4, gapX = 18, gapY = 18;
  const lim = 1.6;
  const final = stage >= d.timesteps.length;
  const x = (t: number) => padL + (t / (horizon - 1)) * (W - padL - 4);
  const y = (v: number) => padT + (1 - (Math.max(-lim, Math.min(lim, v)) + lim) / (2 * lim)) * (H - padT - 4);
  const div = d.diversity.chunks_normalized;
  const total = { w: cols * W + (cols - 1) * gapX, h: 2 * H + gapY };
  const exX = x(d.n_executed - 0.5);
  return (
    <figure className="chart chunk-chart">
      <svg viewBox={`0 0 ${total.w} ${total.h}`} role="img" aria-label={`Action chunk, 16 steps × 7 action dimensions, at ${final ? "the sampler output" : `denoising step ${stage + 1} of ${d.timesteps.length}`}; the first ${d.n_executed} steps were executed.`}>
        {DIMS.map((name, k) => {
          const ox = (k % cols) * (W + gapX), oy = Math.floor(k / cols) * (H + gapY);
          const lo = Array.from({ length: horizon }, (_, t) => Math.min(...div.map((c) => c[t][k])));
          const hi = Array.from({ length: horizon }, (_, t) => Math.max(...div.map((c) => c[t][k])));
          const band = [...hi.map((v, t) => `${x(t)},${y(v)}`), ...lo.map((v, t) => `${x(t)},${y(v)}`).reverse()].join(" ");
          const line = (rows: number[][]) => rows.map((r, t) => `${x(t)},${y(r[k])}`).join(" ");
          return (
            <g key={name} transform={`translate(${ox} ${oy})`}>
              <rect x={padL} y={padT} width={exX - padL} height={H - padT - 4} fill="var(--signal-soft)" />
              <line x1={padL} x2={W - 4} y1={y(0)} y2={y(0)} className="grid-line" />
              <text x={padL} y={10} className="tick-label">{name}</text>
              <polygon points={band} fill="var(--sand)" opacity={0.34} />
              {!final && (
                <>
                  <polyline points={line(d.final_normalized)} fill="none" stroke="var(--ivory-faint)" strokeWidth={1} strokeDasharray="2 3" />
                  {d.sample_in_normalized[stage].map((r, t) => (
                    <circle key={t} cx={x(t)} cy={y(r[k])} r={1.9} fill="none" stroke="var(--signal)" strokeWidth={1} opacity={Math.abs(r[k]) > lim ? 0.45 : 0.85} />
                  ))}
                  <polyline points={line(d.x0_normalized[stage])} fill="none" stroke="var(--signal)" strokeWidth={1.6} />
                </>
              )}
              {final && <polyline points={line(d.final_normalized)} fill="none" stroke="var(--signal)" strokeWidth={2.2} />}
            </g>
          );
        })}
        <g transform={`translate(${3 * (W + gapX)} ${H + gapY})`} className="chunk-key">
          <rect x={4} y={18} width={14} height={9} fill="var(--signal-soft)" /><text x={24} y={26} className="note-label">executed steps</text>
          <rect x={4} y={34} width={14} height={9} fill="var(--sand)" opacity={0.35} /><text x={24} y={42} className="note-label">16 other seeds</text>
          {!final ? (
            <>
              <circle cx={11} cy={54} r={2.4} fill="none" stroke="var(--signal)" /><text x={24} y={58} className="note-label">sampler state x_t</text>
              <line x1={4} x2={18} y1={68} y2={68} stroke="var(--signal)" strokeWidth={1.6} /><text x={24} y={72} className="note-label">predicted clean chunk</text>
            </>
          ) : (
            <><line x1={4} x2={18} y1={54} y2={54} stroke="var(--signal)" strokeWidth={2.2} /><text x={24} y={58} className="note-label">sampled chunk</text></>
          )}
        </g>
      </svg>
      <figcaption>Normalized action space, axis ±{lim}; values beyond are pinned to the edge and drawn faint. Diffusion predicts {horizon} steps at once; only the shaded first {d.n_executed} reach the simulator before the next decision replans.</figcaption>
    </figure>
  );
});

function groupSizes(names: string[], width: number): [string, number][] | null {
  const g = names.map((n): [string, number] => [n, n.endsWith("_quat") ? 4 : n.endsWith("gripper_qpos") ? 2 : 3]);
  return g.reduce((s, [, k]) => s + k, 0) === width ? g : null;
}

const OBS_LABEL = (n: string) => {
  if (n === "robot0_eef_pos") return "gripper xyz";
  if (n === "robot0_eef_quat") return "gripper quat";
  if (n === "robot0_gripper_qpos") return "fingers";
  const short = (x: string) => (x === "robot0_eef" ? "gripper" : x.replace(/_main$/, ""));
  const rel = /^(.+)_to_(.+)_pos$/.exec(n);
  if (rel) return `${short(rel[1])} to ${short(rel[2])}`;
  const obj = n.split("_")[0];
  if (n.endsWith("_quat")) return `${obj} quat`;
  if (n.endsWith("_pos")) return `${obj} xyz`;
  return n;
};

const ObsWindow = memo(function ObsWindow({ d, names }: { d: Decision; names: string[] }) {
  const width = d.obs_window_normalized[0]?.length ?? 0;
  const groups = groupSizes(names, width);
  const cell = 22, L = 74, W = L + width * cell, H = 34 + d.obs_window_normalized.length * (cell + 4);
  const colour = (v: number) => {
    const a = Math.min(1, Math.abs(v));
    return v >= 0 ? `rgba(60,200,240,${0.12 + 0.88 * a})` : `rgba(201,168,107,${0.12 + 0.88 * a})`;
  };
  let acc = 0;
  return (
    <figure className="chart obs-window">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Observation window: the last ${d.obs_window_normalized.length} observations the policy conditioned on, ${width} normalized values each.`}>
        {groups?.map(([n, k]) => {
          const x0 = L + acc * cell; acc += k;
          return (
            <g key={n}>
              <line x1={x0 + 2} x2={x0 + k * cell - 2} y1={24} y2={24} stroke="var(--rule)" />
              <text x={x0 + (k * cell) / 2} y={16} textAnchor="middle" className="tick-label obs-group">{OBS_LABEL(n)}</text>
            </g>
          );
        })}
        {d.obs_window_normalized.map((row, r) => (
          <g key={r} transform={`translate(0 ${32 + r * (cell + 4)})`}>
            <text x={L - 8} y={cell / 2 + 4} textAnchor="end" className="tick-label">{r === d.obs_window_normalized.length - 1 ? "now" : `t−${d.obs_window_normalized.length - 1 - r}`}</text>
            {row.map((v, i) => (
              <rect key={i} x={L + i * cell + 1} y={0} width={cell - 2} height={cell} rx={2}
                fill={d.obs_valid[r] ? colour(v) : "none"} stroke={d.obs_valid[r] ? "none" : "var(--rule)"} strokeDasharray="2 2">
                <title>{`${d.obs_valid[r] ? v.toFixed(3) : "padding (episode start)"}`}</title>
              </rect>
            ))}
          </g>
        ))}
      </svg>
      <figcaption>The conditioning input: the last {d.obs_window_normalized.length} observations, normalized (cyan positive, sand negative). Dashed cells are padding before the episode's first frame. These are privileged simulator states, not camera images.</figcaption>
    </figure>
  );
});

// Spread of the clone-simulated gripper positions over the 16 extra seeds plus the sampled
// chunk, after `step` actions: RMS distance from their centroid and the largest pairwise
// distance, in metres. Computed here from the exported paths (an export measurement).
function spread(d: Decision, step: number): { rms: number; max: number; n: number } {
  const ends = [d.paths.final, ...d.paths.diversity].map((p) => p[Math.min(step, p.length - 1)]);
  const c = [0, 1, 2].map((i) => ends.reduce((a, e) => a + e[i], 0) / ends.length);
  const rms = Math.sqrt(ends.reduce((a, e) => a + (e[0] - c[0]) ** 2 + (e[1] - c[1]) ** 2 + (e[2] - c[2]) ** 2, 0) / ends.length);
  let max = 0;
  for (const a of ends) for (const b of ends) max = Math.max(max, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
  return { rms, max, n: ends.length };
}

const EnvActions = memo(function EnvActions({ d }: { d: Decision }) {
  return (
    <div className="table-scroll">
      <table className="env-actions">
        <caption>The sampled chunk in the controller's units (denormalized): the first {d.n_executed} rows were sent to the simulator; the rest were discarded when the policy replanned.</caption>
        <thead><tr><th scope="col">Step</th>{DIMS.map((n) => <th key={n} scope="col" className="num">{n}</th>)}</tr></thead>
        <tbody>{d.final_env_action.map((r, t) => (
          <tr key={t} className={t < d.n_executed ? "" : "discarded"}>
            <td className="num">{d.first_action_index + t}{t < d.n_executed ? "" : " (discarded)"}</td>
            {r.map((v, k) => <td key={k} className="num">{v.toFixed(3)}</td>)}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
});

function Explorer({ clip, den }: { clip: Clip; den: Denoise }) {
  const clock = useClockFor(clip.frames.n, clip.control_dt_s);
  const frames = useMemo(() => den.decisions.map((d) => d.frame), [den]);
  const di = useDecisionFromClock(clock, frames);
  const d = den.decisions[di];
  const nSteps = d.timesteps.length;
  const [stage, setStage] = useState(nSteps);
  const [diversity, setDiversity] = useState(false);
  const [view, setView] = useState("above");
  const off = VIEWS.find((v) => v[0] === view)![2];
  const [running, setRunning] = useState(false);
  const timer = useRef<number>(0);
  useEffect(() => () => window.clearInterval(timer.current), []);
  const run = () => {
    window.clearInterval(timer.current);
    if (prefersReducedMotion()) { setStage((s) => (s >= nSteps ? 0 : s + 1)); return; }
    setStage(0); setRunning(true);
    let s = 0;
    timer.current = window.setInterval(() => {
      s += 1; setStage(s);
      if (s >= nSteps) { window.clearInterval(timer.current); setRunning(false); }
    }, 260);
  };
  const seekDecision = (i: number) => { clock.pause(); clock.seekFrame(den.decisions[i].frame); };
  const sched = den.scheduler.steps[Math.min(stage, nSteps - 1)];
  const spec = useMemo(() => paths(d, stage, diversity), [d, stage, diversity]);
  const start = d.paths.recorded_executed[0];
  const focus = useMemo(() => {
    const pts = [...d.paths.final, ...d.paths.recorded_executed];
    return [0, 1, 2].map((i) => (Math.min(...pts.map((p) => p[i])) + Math.max(...pts.map((p) => p[i]))) / 2);
  }, [d]);
  return (
    <div className="inside">
      <div className="inside-grid">
        <div className="inside-viewer">
          <ViewerOrFallback clip={clip} clock={clock} preset="orbit" options={{ ...DEFAULT_OPTIONS, trail: false, successHeight: clip.task === "lift" }}
            ariaLabel={`3D reconstruction of validation episode ${clip.state_id} at decision ${di}, with candidate end-effector paths`}
            focus={off ? focus : undefined} focusOffset={off ?? undefined}>
            <PathLines paths={spec} markers={[{ at: start, color: 0xeee7d7, r: 0.007 }]} />
          </ViewerOrFallback>
          <div className="chips" role="group" aria-label="Camera">
            {VIEWS.map(([k, l]) => <button key={k} type="button" className="chip" aria-pressed={view === k} onClick={() => setView(k)}>{l}</button>)}
          </div>
          <ul className="legend path-legend">
            <li><span className="swatch" style={{ background: "var(--ivory)" }} /><span className="kind recorded">recorded</span> executed path</li>
            {stage < nSteps ? (
              <>
                <li><span className="swatch" style={{ background: "var(--signal)" }} /><span className="kind proposal">proposal</span> predicted clean chunk, run in a clone</li>
                <li><span className="swatch dashed" /><span className="kind proposal">proposal</span> raw sampler state, run in a clone</li>
              </>
            ) : (
              <li><span className="swatch" style={{ background: "var(--signal)" }} /><span className="kind proposal">proposal</span> sampled chunk: solid part executed, dashed part discarded</li>
            )}
            {diversity && <li><span className="swatch" style={{ background: "var(--sand)" }} /><span className="kind proposal">proposal</span> 16 extra seeds, never executed</li>}
          </ul>
        </div>
        <div className="inside-panel">
          <div className="decision-picker">
            <label htmlFor="decision-range">Decision <strong className="num">{di + 1}</strong> of {den.decisions.length}, at frame {d.frame}</label>
            <input id="decision-range" type="range" min={0} max={den.decisions.length - 1} value={di}
              onChange={(e) => seekDecision(Number(e.target.value))} />
          </div>
          <div className="sampler">
            <div className="sampler-head">
              <button type="button" className="btn btn-primary" onClick={run} disabled={running}>
                {prefersReducedMotion() ? "Next denoising step" : "Run the sampler"}
              </button>
              <label className="toggle"><input type="checkbox" checked={diversity} onChange={(e) => setDiversity(e.target.checked)} /> Show 16 other seeds</label>
            </div>
            <label htmlFor="stage-range" className="stage-label">
              {stage < nSteps ? <>Denoising step <strong className="num">{stage + 1}</strong> of {nSteps}, timestep t = {d.timesteps[stage]}, ᾱ<sub>t</sub> = {sched.alpha_prod_t.toFixed(3)}</>
                : <>Sampler output after {nSteps} DDIM steps</>}
            </label>
            <input id="stage-range" type="range" min={0} max={nSteps} value={stage} onChange={(e) => setStage(Number(e.target.value))}
              aria-valuetext={stage < nSteps ? `step ${stage + 1} of ${nSteps}` : "output"} />
            <ol className="stage-ticks" aria-hidden="true">
              {[...d.timesteps.map((t) => `t=${t}`), "out"].map((l, i) => <li key={i} className={i === stage ? "on" : ""}>{l}</li>)}
            </ol>
          </div>
          <ChunkChart d={d} stage={stage} horizon={den.horizons.T_p} />
        </div>
      </div>
      <Transport clock={clock} n={clip.frames.n} label={`validation episode ${clip.state_id}`}
        successFrame={clip.outcome.first_success_step} decisions={decisionFrames(clip)} />
      <details className="env-details">
        <summary>Show this decision's actions in controller units</summary>
        <EnvActions d={d} />
      </details>
      <div className="grid-2 inside-foot">
        <ObsWindow d={d} names={den.observation_names} />
        <dl className="provenance">
          <div><dt>Episode</dt><dd>{den.state_id}, {clip.outcome.success ? `success at step ${clip.outcome.first_success_step}` : `failure (${clip.outcome.stop_reason})`}</dd></div>
          <div><dt>Checkpoint</dt><dd className="num">step {den.checkpoint.step}, sha256 {den.checkpoint.sha256.slice(0, 12)}…</dd></div>
          <div><dt>What this is</dt><dd>{den.kind}</dd></div>
          <div><dt>Checks</dt><dd>replayed chunk equals the recorded action; executed branch equals the record (bitwise); {den.n_branches} clone simulations</dd></div>
          <div><dt>Diversity</dt><dd className="num">mean per-value std across 16 seeds: {d.diversity.mean_std_normalized.toFixed(3)} (normalized units)</dd></div>
          <div><dt>Path spread</dt><dd className="num">{(() => {
            const a = spread(d, d.n_executed), b = spread(d, den.horizons.T_p);
            return `gripper positions of the ${a.n} clone-simulated candidates: after ${d.n_executed} steps, RMS ${(100 * a.rms).toFixed(2)} cm from their centre (largest pair ${(100 * a.max).toFixed(2)} cm); after ${den.horizons.T_p} steps, ${(100 * b.rms).toFixed(2)} cm (${(100 * b.max).toFixed(2)} cm). Computed in the browser from the exported paths. Distinct modes would show as separated clusters of end points; this reports spread only.`;
          })()}</dd></div>
        </dl>
      </div>
    </div>
  );
}

export default function InsidePolicy() {
  const reg = useRegistry();
  const options = EPISODES.map(([task, role, label]) => ({ label, clip: reg.clips.find((c) => c.task === task && c.roles.includes(role)) }));
  const avail = options.filter((o) => o.clip);
  const [pick, setPick] = useState(0);
  const chosen = avail[Math.min(pick, avail.length - 1)]?.clip;
  const clip = useClip(chosen?.id ?? null);
  const den = useAsync(() => (clip.status === "ready" && clip.value.denoise ? loadDenoise(clip.value.denoise)
    : Promise.reject(new Error(clip.status === "ready" ? "this clip has no denoising trace" : "waiting for the clip"))), [clip.status === "ready" ? clip.value.id : null]);
  return (
    <section id="inside" aria-labelledby="inside-title">
      <div className="wrap">
        <div className="section-head">
          <h2 id="inside-title">Inside the policy</h2>
          <p className="lede">
            Every eight steps the diffusion policy starts from noise in a 16-step × 7-dimension action
            space and removes it in ten DDIM steps, conditioned on the last two observations. Below are
            the sampler's real intermediate states, recomputed bitwise from recorded validation episodes,
            and where each candidate would have taken the gripper in a cloned simulation.
          </p>
        </div>
        <fieldset className="chips" style={{ marginBottom: 18 }}>
          <legend>Episode ({TASK.lift} and {TASK.can} validation starts, never held-out)</legend>
          {options.map((o) => (
            <button key={o.label} type="button" className="chip" disabled={!o.clip} aria-pressed={chosen?.id === o.clip?.id && !!o.clip}
              onClick={() => setPick(avail.indexOf(o))} title={o.clip ? undefined : "not in this export"}>{o.label}{o.clip ? "" : " (none)"}</button>
          ))}
        </fieldset>
        {!chosen ? <p className="status">No denoising trace in the exported data.</p>
          : clip.status !== "ready" ? <Status state={clip} what={`validation clip ${chosen.id}`} />
          : den.status !== "ready" ? <Status state={den} what="the denoising trace" />
          : <Explorer key={chosen.id} clip={clip.value} den={den.value} />}
        <p className="fine">
          Export, not a result: these sampler calls were re-run in Session 10 only to show the mechanism.
          The held-out results in Findings come from the frozen run and were not touched. Paths are the
          end effector's trajectory in a clone of the simulator; intermediate sampler states are not
          physical plans, and running them shows only where such an action sequence would lead.
        </p>
      </div>
    </section>
  );
}
