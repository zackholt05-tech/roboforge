// B. Why diffusion for actions? Two figures that are kept visibly apart: an explanatory
// schematic (drawn by hand, labelled illustrative) and a strip of the policy's real sampler
// tensors from a recorded validation decision (D-089 rule 4), unaltered.
import { useMemo, useState } from "react";
import { Status } from "../components/ClipParts";
import { loadDenoise } from "../lib/data";
import { TASK } from "../lib/format";
import { useAsync } from "../lib/hooks";
import type { Decision } from "../lib/schema";
import { useRegistry } from "./registry";

function Schematic() {
  const [mode, setMode] = useState<"bc" | "diffusion">("bc");
  // Hand-placed coordinates of a toy 2D scene: two equally good ways around an obstacle.
  const left = "M60,250 C70,180 90,120 150,96 C200,78 250,70 300,60";
  const right = "M60,250 C130,250 210,240 250,190 C280,150 292,100 300,60";
  const mean = "M60,250 C100,215 140,170 175,138 C215,110 260,84 300,60";
  return (
    <figure className="schematic chart">
      <div className="fig-kind"><span className="kind illustrative">illustrative schematic, not data</span></div>
      <svg viewBox="0 0 360 300" role="img" aria-label={mode === "bc"
        ? "Schematic: two demonstrated paths go around an obstacle on either side; a regression policy averages them into a path through the obstacle."
        : "Schematic: a diffusion policy samples whole paths, each following one of the demonstrated ways around the obstacle."}>
        <rect x={150} y={112} width={70} height={58} rx={4} fill="var(--slate-2)" stroke="var(--rule)" />
        <text x={185} y={186} textAnchor="middle" className="tick-label">obstacle</text>
        <circle cx={60} cy={250} r={6} fill="var(--ivory)" />
        <text x={60} y={276} textAnchor="middle" className="tick-label">gripper</text>
        <circle cx={300} cy={60} r={7} fill="none" stroke="var(--ivory)" strokeWidth={1.5} />
        <text x={300} y={40} textAnchor="middle" className="tick-label">goal</text>
        <path d={left} fill="none" stroke="var(--ivory-dim)" strokeWidth={1.4} strokeDasharray="3 4" />
        <path d={right} fill="none" stroke="var(--ivory-dim)" strokeWidth={1.4} strokeDasharray="3 4" />
        {mode === "bc" ? (
          <path d={mean} fill="none" stroke="var(--sand)" strokeWidth={3} />
        ) : (
          [-6, -2, 3].map((d, i) => <path key={i} d={left} transform={`translate(${d} ${-d})`} fill="none" stroke="var(--signal)" strokeWidth={2} opacity={0.85} />)
            .concat([4, -3].map((d, i) => <path key={`r${i}`} d={right} transform={`translate(${d} ${d / 2})`} fill="none" stroke="var(--signal)" strokeWidth={2} opacity={0.85} />))
        )}
      </svg>
      <div className="chips" role="group" aria-label="Policy type in the schematic">
        <button type="button" className="chip" aria-pressed={mode === "bc"} onClick={() => setMode("bc")}>Regression (one answer)</button>
        <button type="button" className="chip" aria-pressed={mode === "diffusion"} onClick={() => setMode("diffusion")}>Diffusion (samples)</button>
      </div>
      <figcaption>
        {mode === "bc"
          ? "A policy trained to predict the average action blends two valid demonstrations into one that neither demonstrator took."
          : "A diffusion policy learns the distribution of whole action sequences and draws one sample, so each plan follows a single demonstrated way."}{" "}
        Dashed: demonstrations. This is the textbook motivation; whether it matters on Lift and Can is what the findings test.
      </figcaption>
    </figure>
  );
}

const ROWS: [number, string][] = [[0, "Δx"], [1, "Δy"], [2, "Δz"], [6, "gripper"]];

function TensorStrip({ d, horizon }: { d: Decision; horizon: number }) {
  const steps = d.timesteps.length;
  const cw = 64, ch = 40, gx = 8, gy = 10, L = 56, T = 22;
  const W = L + (steps + 1) * (cw + gx), H = T + ROWS.length * (ch + gy) + 8;
  const lim = 1.6;
  const x = (t: number) => 3 + (t / (horizon - 1)) * (cw - 6);
  const y = (v: number) => ch / 2 - (Math.max(-lim, Math.min(lim, v)) / lim) * (ch / 2 - 2);
  const poly = (rows: number[][], k: number) => rows.map((r, t) => `${x(t)},${y(r[k])}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" className="tensor-strip"
      aria-label={`Recorded sampler tensors: ${steps} DDIM steps from Gaussian noise to the final 16-step action chunk, for 4 of the 7 action dimensions; the last column overlays 16 extra seeds.`}>
      {[...d.timesteps.map((t) => `t = ${t}`), "output"].map((l, i) => (
        <text key={l} x={L + i * (cw + gx) + cw / 2} y={12} textAnchor="middle" className="tick-label">{l}</text>
      ))}
      {ROWS.map(([k, name], r) => (
        <g key={name} transform={`translate(0 ${T + r * (ch + gy)})`}>
          <text x={L - 8} y={ch / 2 + 4} textAnchor="end" className="tick-label">{name}</text>
          {Array.from({ length: steps + 1 }, (_, i) => (
            <g key={i} transform={`translate(${L + i * (cw + gx)} 0)`}>
              <rect width={cw} height={ch} rx={3} fill="var(--slate)" />
              <line x1={2} x2={cw - 2} y1={ch / 2} y2={ch / 2} className="grid-line" />
              {i < steps ? (
                <>
                  {d.sample_in_normalized[i].map((row, t) => <circle key={t} cx={x(t)} cy={y(row[k])} r={1.3} fill="var(--signal)" opacity={0.75} />)}
                  <polyline points={poly(d.x0_normalized[i], k)} fill="none" stroke="var(--ivory)" strokeWidth={1} opacity={0.55} />
                </>
              ) : (
                <>
                  {d.diversity.chunks_normalized.map((c, j) => <polyline key={j} points={poly(c, k)} fill="none" stroke="var(--sand)" strokeWidth={0.8} opacity={0.45} />)}
                  <polyline points={poly(d.final_normalized, k)} fill="none" stroke="var(--signal)" strokeWidth={1.8} />
                </>
              )}
            </g>
          ))}
        </g>
      ))}
    </svg>
  );
}

export default function WhyDiffusion() {
  const reg = useRegistry();
  const pick = useMemo(() => {
    const cands = reg.clips.filter((c) => c.roles.includes("denoise:first_success"));
    return cands.find((c) => c.task === "lift") ?? cands[0];
  }, [reg]);
  const file = pick ? `denoise/${pick.id}.json.gz` : null;
  const den = useAsync(() => (file && reg.files[file] ? loadDenoise(file) : Promise.reject(new Error("no denoising trace exported"))), [file]);
  return (
    <section id="why" aria-labelledby="why-title">
      <div className="wrap">
        <div className="section-head">
          <h2 id="why-title">Why diffusion for actions?</h2>
          <p className="lede">
            Behaviour cloning usually regresses one action from the current state. A diffusion policy
            instead learns to turn noise into a whole sequence of future actions, so it can represent
            several good ways to move, and it commits to a short plan at a time.
          </p>
        </div>
        <div className="why-grid">
          <Schematic />
          <figure className="recorded-fig chart">
            <div className="fig-kind"><span className="kind recorded">recorded sampler tensors</span></div>
            {den.status !== "ready" ? <Status state={den} what="the recorded denoising trace" /> : (
              <>
                <TensorStrip d={den.value.decisions[0]} horizon={den.value.horizons.T_p} />
                <figcaption>
                  The trained policy's first decision on {TASK[den.value.task]} validation start {den.value.state_id.split("_").pop()}: dots are the sampler's
                  state x<sub>t</sub> at each of the 10 DDIM steps, starting as Gaussian noise; the ivory line is the clean chunk the network predicts from it;
                  the last column is the output (cyan) and 16 more samples from other noise seeds (sand). Normalized units, 16 future steps per cell.
                  Replayed bitwise from the recorded episode; not a held-out result.
                </figcaption>
                <a className="btn" href="#inside">Explore every decision</a>
              </>
            )}
          </figure>
        </div>
      </div>
    </section>
  );
}
