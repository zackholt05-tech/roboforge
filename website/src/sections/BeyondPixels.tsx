// G. Beyond pixels: the frozen RGB study (D-081) and the Level 2 study (D-086, approval A-04),
// DP3-inspired point clouds against the frozen RGB arm. Numbers are findings.json fields (copied
// verbatim from the frozen results); episodes are the D-089 rule 5 placement pairs.
import { useMemo, useState } from "react";
import { DifferenceChart, EstimateChart, OutcomeStrip, TaxonomyBars } from "../charts/Charts";
import { Status } from "../components/ClipParts";
import { decisionFrames, useClip, ViewerOrFallback } from "../components/ClipStage";
import { Transport } from "../components/Transport";
import { dataUrl, loadClouds, loadFindings } from "../lib/data";
import { FAMILY, FAMILY_NOTE, ciPp, pp, sepLabel } from "../lib/format";
import { useAsync, useClockFor } from "../lib/hooks";
import { useClock, type Clock } from "../lib/playback";
import type { Clip, Findings } from "../lib/schema";
import { CloudPoints } from "../three/CloudPoints";
import { DEFAULT_OPTIONS } from "../three/Viewer";
import { useRegistry } from "./registry";

const cap = (t: string) => t.charAt(0).toUpperCase() + t.slice(1);
const SUITE: Record<string, string> = { robustness_base: "training-like placement", ood_cube_shift: "cube shifted out of the training square", heldout: "held-out test" };
const ARM: Record<string, string> = { pc_diffusion: "Point cloud", rgb_diffusion: "RGB" };
const SET: Record<string, string> = { robustness_base: "training-like", ood_cube_shift: "shifted cube", heldout: "held-out" };
const ARM_SET = (k: string) => { const [a, s] = k.split(":"); return `${ARM[a] ?? a}, ${SET[s] ?? s}`; };
const CONTRAST = (kind: string) => kind.startsWith("PRIMARY H5") ? "H5: point cloud − RGB, shifted"
  : kind.startsWith("drop (base - OOD), pc") ? "Drop, point cloud (training-like − shifted)"
  : kind.startsWith("drop (base - OOD), rgb") ? "Drop, RGB (training-like − shifted)"
  : kind.startsWith("between-arm difference in base") ? "Point cloud − RGB, training-like (context)" : kind;
const ARM_COLOUR = (k: string) => (k.startsWith("pc_") ? "var(--signal)" : "var(--sand)");
const CASES = [
  ["pixels:a_pc_fails_rgb_succeeds", "Shifted cube: point cloud fails, RGB succeeds"],
  ["pixels:b_both_succeed", "Shifted cube: both succeed"],
  ["pixels:base_first", "Training-like placement"],
] as const;

function RgbContext({ f }: { f: Findings }) {
  const ctx = (f.rgb as unknown as { level1_context?: { label: string; families: Record<string, { estimate: number }> } }).level1_context;
  if (!ctx) return null;
  const d = ctx.families.diffusion_unet;
  return <>{d && `For context only, the state-based ${FAMILY.diffusion_unet} scored ${(100 * d.estimate).toFixed(1)} % on the same 50 starts`} ({ctx.label.replace(/^CONTEXT ONLY: /, "").replace(/^Level 1 low-dim policies, same 50 Lift held-out starts; /, "")}).</>;
}

function PlacementMap({ f, arm }: { f: Findings; arm: "pc_diffusion" | "rgb_diffusion" }) {
  const m = f.level2.placement_map;
  const S = 300, lim = 0.085, c = S / 2;
  const sx = (v: number) => c + ((v - m.center_xy[0]) / lim) * c;
  const sy = (v: number) => c - ((v - m.center_xy[1]) / lim) * c;
  const r = 5;
  const wedge = (x: number, y: number, k: number) => {
    if (k >= 3) return <circle cx={x} cy={y} r={r} fill={ARM_COLOUR(arm)} />;
    if (k <= 0) return null;
    const a = (2 * Math.PI * k) / 3;
    const x1 = x + r * Math.sin(a), y1 = y - r * Math.cos(a);
    return <path d={`M${x},${y} L${x},${y - r} A${r},${r} 0 ${a > Math.PI ? 1 : 0} 1 ${x1},${y1} Z`} fill={ARM_COLOUR(arm)} />;
  };
  const ood = m.states.filter((s) => s.suite === "ood_cube_shift");
  const won = ood.reduce((a, s) => a + s.successes_of_3_seeds[arm], 0);
  const h = m.training_square_half_m;
  return (
    <figure className="chart placement-map">
      <svg viewBox={`0 0 ${S} ${S + 22}`} role="img" aria-label={`${FAMILY[arm]}: cube start positions seen from above; ${won} of ${3 * ood.length} shifted-cube episodes succeeded across 3 seeds.`}>
        <circle cx={c} cy={c} r={(m.ood_radius_m[1] / lim) * c} fill="var(--signal-soft)" opacity={0.5} />
        <circle cx={c} cy={c} r={(m.ood_radius_m[0] / lim) * c} fill="var(--graphite)" />
        <rect x={sx(m.center_xy[0] - h)} y={sy(m.center_xy[1] + h)} width={(2 * h / lim) * c} height={(2 * h / lim) * c}
          fill="none" stroke="var(--ivory-dim)" strokeDasharray="4 3" />
        {[-0.05, 0, 0.05].map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={0} y2={S} className="grid-line" opacity={0.5} />
            <line x1={0} x2={S} y1={sy(t)} y2={sy(t)} className="grid-line" opacity={0.5} />
            <text x={sx(t)} y={S + 14} textAnchor="middle" className="tick-label">{(100 * t).toFixed(0)} cm</text>
          </g>
        ))}
        <text x={6} y={14} className="tick-label">← robot base, at x = −56 cm</text>
        <text x={S - 4} y={c - 6} textAnchor="end" className="tick-label">+x</text>
        <text x={c + 6} y={12} className="tick-label">+y</text>
        {m.states.map((s) => {
          const x = sx(s.cube_initial_pos_m[0]), y = sy(s.cube_initial_pos_m[1]);
          const k = s.successes_of_3_seeds[arm];
          return (
            <g key={s.state_id}>
              <circle cx={x} cy={y} r={r} fill="none" stroke={k === 0 ? "var(--error)" : ARM_COLOUR(arm)} strokeWidth={1.3} />
              {wedge(x, y, k)}
              <title>{`${s.state_id}: ${k} of 3 seeds succeeded`}</title>
            </g>
          );
        })}
      </svg>
      <figcaption><strong>{FAMILY[arm]}</strong>: {won} of {3 * ood.length} shifted-cube episodes succeed. Each disc is one start, filled by how many of 3 seeds succeeded (red ring: none). Dashed square: the training placement range (±{100 * h} cm); shaded ring: the shifted placements ({100 * m.ood_radius_m[0]}–{100 * m.ood_radius_m[1]} cm from centre). Top view.</figcaption>
    </figure>
  );
}

function PolicyView({ clip, clock }: { clip: Clip; clock: Clock }) {
  useClock(clock);
  const pv = clip.policy_views;
  if (!pv) return null;
  const k = Math.round(clock.frame);
  let j = 0;
  for (let i = 0; i < pv.frames.length; i++) if (pv.frames[i] <= k) j = i;
  return (
    <figure className="policy-view">
      <div>{pv.cameras.map((c) => (
        <img key={c} src={dataUrl(`policy_views/${pv.files[c][j]}`)} width={pv.size} height={pv.size}
          alt={`The RGB policy's ${c} input (${pv.size}×${pv.size}) at decision frame ${pv.frames[j]}`} />
      ))}</div>
      <figcaption>What the RGB policy saw at frame {pv.frames[j]}: its two {pv.size}² inputs, rendered from the recorded state.</figcaption>
    </figure>
  );
}

function PcStage({ pc, rgb }: { pc: Clip; rgb: Clip }) {
  const n = Math.max(pc.frames.n, rgb.frames.n);
  const clock = useClockFor(n, pc.control_dt_s);
  const clouds = useAsync(() => (pc.clouds ? loadClouds(pc.clouds) : Promise.reject(new Error("no cloud file"))), [pc.id]);
  const [onlyCloud, setOnlyCloud] = useState(false);
  const pcOpts = { ...DEFAULT_OPTIONS, trail: !onlyCloud, successHeight: !onlyCloud, hideScene: onlyCloud };
  return (
    <div className="h2h">
      <div className="h2h-pair">
        <figure className="h2h-side focus">
          <figcaption><span className="h2h-name">{FAMILY[pc.family]}</span>
            <span className={pc.outcome.success ? "ok" : "bad"}>{pc.outcome.success ? `success at step ${pc.outcome.first_success_step}` : `failure, ${pc.outcome.steps} steps`}</span></figcaption>
          <ViewerOrFallback clip={pc} clock={clock} preset="orbit" options={pcOpts}
            ariaLabel={`3D reconstruction of the point-cloud policy on ${pc.state_id}, with its input cloud`}>
            {clouds.status === "ready" && <CloudPoints clouds={clouds.value} clock={clock} />}
          </ViewerOrFallback>
          {clouds.status !== "ready" && <Status state={clouds} what="the input point clouds" />}
          <ul className="legend"><li><span className="swatch" style={{ background: "#d9824a" }} />cloud points from agentview depth</li>
            <li><span className="swatch" style={{ background: "var(--signal)" }} />from the wrist camera's depth</li></ul>
        </figure>
        <figure className="h2h-side base">
          <figcaption><span className="h2h-name">{FAMILY[rgb.family]}</span>
            <span className={rgb.outcome.success ? "ok" : "bad"}>{rgb.outcome.success ? `success at step ${rgb.outcome.first_success_step}` : `failure, ${rgb.outcome.steps} steps`}</span></figcaption>
          <ViewerOrFallback clip={rgb} clock={clock} preset="orbit" options={{ ...DEFAULT_OPTIONS, successHeight: true }}
            ariaLabel={`3D reconstruction of the RGB policy on ${rgb.state_id}`} />
          <PolicyView clip={rgb} clock={clock} />
        </figure>
      </div>
      <Transport clock={clock} n={n} label={`${pc.state_id}, both arms`} successFrame={pc.outcome.first_success_step ?? rgb.outcome.first_success_step}
        decisions={decisionFrames(pc)} />
      <label className="toggle"><input type="checkbox" checked={onlyCloud} onChange={(e) => setOnlyCloud(e.target.checked)} /> Show only the point cloud, what the point-cloud policy sees</label>
      <p className="fine">Seed 0 of each arm (point cloud step {pc.checkpoint.step}, RGB step {rgb.checkpoint.step}) from the same start, one shared clock. {pc.clouds ? "The cloud is the policy's own input, recomputed from the recorded state; it updates at each decision (every 8 steps)." : ""}</p>
    </div>
  );
}

function Level2({ f }: { f: Findings }) {
  const reg = useRegistry();
  const l2 = f.level2;
  const P = l2.placement.bootstrap;
  const h5 = P.differences.find((d) => (d.kind ?? "").startsWith("PRIMARY H5"))!;
  const order = ["pc_diffusion:robustness_base", "rgb_diffusion:robustness_base", "pc_diffusion:ood_cube_shift", "rgb_diffusion:ood_cube_shift"];
  const [kase, setKase] = useState<string>(CASES[0][0]);
  const pair = useMemo(() => {
    const pc = reg.clips.find((c) => c.roles.includes(kase) && c.family === "pc_diffusion");
    const rgb = reg.clips.find((c) => c.roles.includes(kase) && c.family === "rgb_diffusion");
    return pc && rgb ? [pc, rgb] as const : null;
  }, [reg, kase]);
  const A = useClip(pair?.[0].id ?? null);
  const B = useClip(pair?.[1].id ?? null);
  const strips = l2.strips.filter((s) => s.suite === (pair?.[0].suite ?? "ood_cube_shift"))
    .sort((a, b) => a.arm.localeCompare(b.arm) || a.seed - b.seed)
    .map((s) => ({ name: `${FAMILY[s.arm]}, seed ${s.seed}`, outcomes: s.outcomes }));
  const tax = Object.entries(l2.taxonomy_by_arm_and_set).filter(([k]) => k.endsWith("ood_cube_shift"));
  const er = l2.training_erratum;
  return (
    <>
      <div className="h5">
        <p className="big-number num bad">{pp(h5.difference)} <span className="unit">points</span></p>
        <div>
          <h3>Point clouds generalize worse when the cube moves</h3>
          <p className="prose">
            The pre-registered primary test (H5): with the cube placed 5 to 7 cm from the centre, outside
            the ±3 cm training square, point-cloud diffusion succeeds {pp(h5.difference).replace("−", "")} points less often than RGB
            diffusion, 95 % interval {ciPp(h5.ci95)}: {sepLabel(h5.separation)}. On training-like
            placements and the held-out test, both arms succeed every time.
          </p>
          <p className="fine">{cap(l2.privilege)}. {cap(l2.power)}.</p>
        </div>
      </div>
      <div className="grid-2">
        <EstimateChart title="Lift success by arm and placement" rows={order.filter((k) => P.families[k]).map((k) => [k, P.families[k]])}
          label={ARM_SET} colour={ARM_COLOUR} />
        <DifferenceChart title="Placement contrasts" rows={P.differences} label={(d) => CONTRAST(d.kind ?? "")} />
      </div>
      <div className="grid-2 maps">
        <PlacementMap f={f} arm="pc_diffusion" />
        <PlacementMap f={f} arm="rgb_diffusion" />
      </div>
      <h3 className="sub">Watch a shifted start</h3>
      <fieldset className="chips" style={{ margin: "10px 0 18px" }}>
        <legend>Episode pair (rule-selected)</legend>
        {CASES.map(([k, l]) => {
          const has = reg.clips.some((c) => c.roles.includes(k));
          return <button key={k} type="button" className="chip" disabled={!has} aria-pressed={kase === k} onClick={() => setKase(k)}>{l}{has ? "" : " (none)"}</button>;
        })}
      </fieldset>
      {!pair ? <p className="status">No start in the frozen evidence fits this case.</p>
        : A.status !== "ready" || B.status !== "ready" ? <Status state={A.status !== "ready" ? A : B} what="the placement pair" />
        : <PcStage key={kase} pc={A.value} rgb={B.value} />}
      <OutcomeStrip label={`Every ${SUITE[pair?.[0].suite ?? "ood_cube_shift"]} episode, by arm and seed (outlined: the start shown)`}
        rows={strips} highlight={pair?.[0].state_id ?? null} />
      <div className="grid-2">
        <TaxonomyBars title="How the shifted-cube failures failed" rows={tax} label={ARM_SET} />
        <figure className="overlays">
          <div className="overlay-strip">
            {Object.keys(reg.files).filter((k) => k.startsWith("overlays/")).map((k) => (
              <img key={k} src={dataUrl(k)} alt={`Session 9 calibration overlay ${k.replace("overlays/", "")}: sampled cloud points projected onto MuJoCo renders`} loading="lazy" />
            ))}
          </div>
          <figcaption>Calibration check from the Session 9 feasibility run, copied unmodified: 512 sampled points (orange from agentview, cyan from the wrist camera) projected back onto 256² renders of a training demonstration.</figcaption>
        </figure>
      </div>
      <div className="table-scroll">
        <table className="training">
          <caption>Level 2 point-cloud training runs as generated by the frozen analysis: paid remote GPU (Modal, one NVIDIA L4), approval A-04, ${Object.values(l2.training_generated).reduce((a, t) => a + t.billed_usd, 0).toFixed(2)} billed in all.</caption>
          <thead><tr><th scope="col">Run</th><th scope="col">ms per step</th><th scope="col">Train wall time</th><th scope="col">Segments</th><th scope="col">Billed</th></tr></thead>
          <tbody>{Object.entries(l2.training_generated).map(([id, t]) => (
            <tr key={id} className={id === er.run_id ? "erratum-row" : ""}>
              <td>{id}</td>
              <td className="num">{id === er.run_id ? <><s>{t.ms_per_step.toFixed(2)}</s> {er.corrected_ms_per_step.toFixed(1)}*</> : t.ms_per_step.toFixed(2)}</td>
              <td className="num">{(t.train_wall_s / 3600).toFixed(2)} h{id === er.run_id ? "*" : ""}</td>
              <td className="num">{t.segments}</td>
              <td className="num">${t.billed_usd.toFixed(2)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <p className="erratum"><strong>* Erratum (KI-049).</strong> {f.notes.ki_049} The generated row's wall time ({(l2.training_generated[er.run_id].train_wall_s / 3600).toFixed(2)} h) is the completed segment only ({er.completed_segment_wall_h} h); the run took {er.total_wall_h} h in total.</p>
    </>
  );
}

export default function BeyondPixels() {
  const s = useAsync(loadFindings, []);
  return (
    <section id="pixels" aria-labelledby="pixels-title">
      <div className="wrap">
        <div className="section-head">
          <h2 id="pixels-title">Beyond privileged state</h2>
          <p className="lede">
            The Level 1 policies read the cube's pose straight from the simulator. Two later studies
            took that away: first two 84-pixel cameras, then a 3D point cloud built from depth, both
            with the robot's own joint readings.
          </p>
        </div>
        {s.status !== "ready" ? <Status state={s} what="the frozen findings" /> : (
          <>
            <h3 className="sub">From cameras alone (RGB study)</h3>
            <div className="grid-2">
              <EstimateChart title="Lift held-out success from two 84² cameras" rows={["rgb_diffusion", "rgb_bc_chunk"].filter((k) => s.value.rgb.success.families[k]).map((k) => [k, s.value.rgb.success.families[k]])} />
              <div>
                <DifferenceChart title="Visual diffusion vs visual BC-chunk" rows={s.value.rgb.success.differences} />
                <p className="fine">{s.value.notes.privilege_rgb} {FAMILY.rgb_diffusion}: {FAMILY_NOTE.rgb_diffusion}. Lift is saturated (KI-034), so success alone cannot rank these two. <RgbContext f={s.value} /></p>
              </div>
            </div>
            <h3 className="sub">From a point cloud (Level 2)</h3>
            <p className="prose dim">{s.value.notes.privilege_level2}</p>
            <Level2 f={s.value} />
          </>
        )}
      </div>
    </section>
  );
}
