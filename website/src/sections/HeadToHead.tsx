// D. Head to head: seed 0 diffusion and seed 0 BC-MLP from the same held-out start, on one
// shared clock. The three cases are the D-089 rule 3 selection: the first start (in manifest
// order) where (a) both succeed, (b) only diffusion succeeds, (c) only BC-MLP succeeds.
import { useMemo, useState } from "react";
import { OutcomeStrip } from "../charts/Charts";
import { ClipCaption, Status } from "../components/ClipParts";
import { decisionFrames, useClip, ViewerOrFallback } from "../components/ClipStage";
import { Transport } from "../components/Transport";
import { loadFindings } from "../lib/data";
import { FAMILY, TASK } from "../lib/format";
import { useAsync, useClockFor } from "../lib/hooks";
import type { Clip } from "../lib/schema";
import { PRESETS, type PresetId } from "../three/cameras";
import { DEFAULT_OPTIONS } from "../three/Viewer";
import { useRegistry } from "./registry";

const CASES = [
  ["a_both_succeed", "Both succeed"],
  ["b_only_diffusion", "Only diffusion succeeds"],
  ["c_only_bc_mlp", "Only BC-MLP succeeds"],
] as const;

function Pair({ a, b }: { a: Clip; b: Clip }) {
  const n = Math.max(a.frames.n, b.frames.n);
  const clock = useClockFor(n, a.control_dt_s);
  const [preset, setPreset] = useState<PresetId>("frontview");
  const opts = { ...DEFAULT_OPTIONS, successHeight: a.task === "lift" };
  return (
    <div className="h2h">
      <div className="h2h-pair">
        {[a, b].map((c) => (
          <figure key={c.id} className={`h2h-side ${c.family === "diffusion_unet" ? "focus" : "base"}`}>
            <figcaption>
              <span className="h2h-name">{FAMILY[c.family]}</span>
              <span className={c.outcome.success ? "ok" : "bad"}>
                {c.outcome.success ? `success at step ${c.outcome.first_success_step}` : `failure, ${c.outcome.steps} steps`}
              </span>
            </figcaption>
            <ViewerOrFallback clip={c} clock={clock} preset={preset} options={opts}
              ariaLabel={`3D reconstruction of ${FAMILY[c.family]} on ${c.state_id}`} />
            <ClipCaption clip={c} />
          </figure>
        ))}
      </div>
      <Transport clock={clock} n={n} label={`${a.state_id}, both policies`}
        successFrame={a.outcome.first_success_step ?? b.outcome.first_success_step}
        decisions={decisionFrames(a.family === "diffusion_unet" ? a : b)} />
      <fieldset className="chips">
        <legend>Camera (both views)</legend>
        {PRESETS.filter((p) => !p.mujoco || a.cameras[p.id]).map((p) => (
          <button key={p.id} type="button" className="chip" aria-pressed={preset === p.id}
            onClick={() => setPreset(p.id)}>{p.label}</button>
        ))}
      </fieldset>
      <p className="fine">One clock drives both reconstructions; the shorter episode holds its last recorded frame. Scrubber ticks mark diffusion's decisions (a new 16-step plan every 8 steps).</p>
    </div>
  );
}

export default function HeadToHead() {
  const reg = useRegistry();
  const findings = useAsync(loadFindings, []);
  const [task, setTask] = useState<"can" | "lift">("can");
  const [kase, setKase] = useState<string>("b_only_diffusion");
  const role = `h2h:${task}:${kase}`;
  const pair = useMemo(() => {
    const d = reg.clips.find((c) => c.roles.includes(role) && c.family === "diffusion_unet");
    const m = reg.clips.find((c) => c.roles.includes(role) && c.family === "bc_mlp");
    return d && m ? [d, m] as const : null;
  }, [reg, role]);
  const A = useClip(pair?.[0].id ?? null);
  const B = useClip(pair?.[1].id ?? null);
  const strips = findings.status === "ready"
    ? findings.value.level1.tasks[task].strips.filter((s) => s.seed === 0 && (s.family === "diffusion_unet" || s.family === "bc_mlp"))
      .sort((x, y) => Number(y.family === "diffusion_unet") - Number(x.family === "diffusion_unet"))
      .map((s) => ({ name: `${FAMILY[s.family]}, seed 0`, outcomes: s.outcomes }))
    : null;
  return (
    <section id="compare" aria-labelledby="compare-title">
      <div className="wrap">
        <div className="section-head">
          <h2 id="compare-title">Same start, two policies</h2>
          <p className="lede">
            Seed 0 of the diffusion policy and seed 0 of the one-step BC-MLP, replayed from the same
            held-out start. Which start is shown is fixed by rule: the first one, in held-out order,
            where both succeed, where only diffusion does, and where only BC-MLP does.
          </p>
        </div>
        <div className="stage-controls" style={{ marginBottom: 18 }}>
          <fieldset className="chips">
            <legend>Task</legend>
            {(["can", "lift"] as const).map((t) => (
              <button key={t} type="button" className="chip" aria-pressed={task === t} onClick={() => setTask(t)}>{TASK[t]}</button>
            ))}
          </fieldset>
          <fieldset className="chips">
            <legend>Case</legend>
            {CASES.map(([k, l]) => (
              <button key={k} type="button" className="chip" aria-pressed={kase === k} onClick={() => setKase(k)}>{l}</button>
            ))}
          </fieldset>
        </div>
        {!pair ? (
          <p className="status">No start in the frozen evidence fits this case for {TASK[task]} (seed 0 vs seed 0), so there is nothing to show.</p>
        ) : A.status !== "ready" || B.status !== "ready" ? (
          <Status state={A.status !== "ready" ? A : B} what={`the ${TASK[task]} pair at ${pair[0].state_id}`} />
        ) : (
          <Pair key={role} a={A.value} b={B.value} />
        )}
        {strips && (
          <OutcomeStrip label={`${TASK[task]}: all 50 held-out starts for the two seed-0 checkpoints (outlined square: the start shown)`}
            rows={strips} highlight={pair?.[0].state_id ?? null} />
        )}
      </div>
    </section>
  );
}
