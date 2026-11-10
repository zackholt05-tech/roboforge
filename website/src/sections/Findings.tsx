// F. Research findings: the Level 1 held-out results and, kept apart, its validation-regime
// robustness (H4) and ablations, drawn from findings.json, which copies the frozen results,
// taxonomy and robustness analysis blocks verbatim. Nothing here is recomputed; summed counts
// are the frozen taxonomy's per-category counts added over seeds, and the callouts only pick
// out values (the highest estimate, the drops whose interval includes 0).
import { useState } from "react";
import { DifferenceChart, EstimateChart, LatencyChart, OutcomeStrip, TaxonomyBars } from "../charts/Charts";
import { Status } from "../components/ClipParts";
import { loadFindings } from "../lib/data";
import { FAMILY, TASK, pct, pp, short } from "../lib/format";
import { useAsync } from "../lib/hooks";
import type { Findings as F } from "../lib/schema";

export const ORDER = ["diffusion_unet", "bc_mlp_chunk", "bc_rnn", "bc_mlp", "random_uniform"];

const total = (t: Record<string, number>) => Object.values(t).reduce((a, b) => a + b, 0);

export function SourceTable({ f }: { f: F }) {
  return (
    <div className="table-scroll">
      <table className="sources">
        <caption>Every number in this section is read from these files (SHA-256 checked when the site data is built).</caption>
        <thead><tr><th scope="col">Frozen file</th><th scope="col">SHA-256</th></tr></thead>
        <tbody>{f.sources.map((s) => (
          <tr key={s.path}><td>{s.path}</td><td className="num" title={s.sha256}>{short(s.sha256, 16)}…</td></tr>
        ))}</tbody>
      </table>
    </div>
  );
}

const SET: Record<string, string> = { robustness_base: "base", ood_cube_shift: "cube shift",
  ood_arm_offset: "arm offset" };
const cm = (m: number) => String(Math.round(m * 1000) / 10); // metres -> cm, at most 1 decimal
const sepd = (s: string) => s.startsWith("separated");
const famSet = (k: string) => { const [fam, set] = k.split(":"); return [fam, set] as const; };
const CONTRAST = /^drop (\S+) - drop (\S+) on (\S+)$/;

function Robustness({ f, task }: { f: F; task: string }) {
  const R = f.level1_robustness.tasks[task];
  if (!R) return null;
  const B = R.h4.bootstrap;
  const P = f.level1_robustness.perturbations;
  const fams = ORDER.filter((k) => k !== "random_uniform");
  const keys = R.h4.sets.flatMap((s) => fams.map((k) => `${k}:${s}`)).filter((k) => B.families[k]);
  const counts = Object.fromEntries(keys.map((k) => {
    const t = R.h4.taxonomy[k];
    return [k, t ? `${t.success ?? 0}/${total(t)}` : ""];
  }));
  const drops = B.differences.filter((d) => d.kind?.startsWith("drop"));
  const between = B.differences.filter((d) => d.kind?.startsWith("between"));
  const contrasts = B.contrasts.map((c) => ({ a: c.name, b: "", difference: c.estimate, ci95: c.ci95, separation: c.separation }));
  const unsep = drops.filter((d) => !sepd(d.separation));
  const ood = R.h4.sets.filter((s) => s !== "robustness_base");
  const best = ood.map((s) => {
    const top = keys.filter((k) => k.endsWith(`:${s}`)).reduce((a, b) => (B.families[b].estimate > B.families[a].estimate ? b : a));
    return `${SET[s]}: ${FAMILY[famSet(top)[0]]}, ${pct(B.families[top].estimate)}`;
  });
  const abl = Object.entries(R.ablations).flatMap(([name, a]) => a.bootstrap.differences.map((d) => ({ ...d, hypothesis: name })));
  const ARM: Record<string, string> = { primary: "primary", "trained-weights": "raw weights", ddpm100: "DDPM-100",
    "T_o=2": "T_o = 2", "T_o=1": "T_o = 1" };
  const evals = [...R.h4.evaluations, ...Object.values(R.ablations).flatMap((a) => a.evaluations)];
  return (
    <div className="robustness">
      <div className="robustness-head">
        <h3>{TASK[task]} under perturbation</h3>
        <span className="kind measured">validation regime, not held-out</span>
      </div>
      <p className="fine">{f.notes.robustness} Each set has {R.h4.n_states} starts.{" "}
        {ood.includes("ood_cube_shift") && <>Cube shift: the cube starts {cm(P.ood_cube_shift.radius_m[0])}–{cm(P.ood_cube_shift.radius_m[1])} cm from the placement centre, outside the ±{cm(P.ood_cube_shift.training_half_range_m)} cm square the demonstrations used. </>}
        Arm offset: every arm joint starts ±{P.ood_arm_offset.offset_rad} rad from its usual pose; training resets add only {P.ood_arm_offset.training_noise_sigma_rad} rad of noise.</p>
      <div className="grid-2 findings-top">
        <EstimateChart title={`${TASK[task]}: success by policy on the base and perturbed sets`} rows={keys.map((k) => [k, B.families[k]])}
          counts={counts} label={(k) => `${FAMILY[famSet(k)[0]]}, ${SET[famSet(k)[1]]}`} />
        <div className="findings-read">
          <p className="callout">
            {drops.every((d) => d.difference > 0)
              ? <><strong>Every family loses success under {ood.length > 1 ? "every perturbation" : `the ${SET[ood[0]]}`}.</strong>{" "}</>
              : <><strong>Success changes under perturbation.</strong>{" "}</>}
            {unsep.length === 0 ? "Every drop is separated." :
              `Every drop is separated except ${unsep.map((d) => `${FAMILY[famSet(d.a)[0]]} on the ${SET[famSet(d.b)[1]]} (${pp(d.difference)} points)`).join(", ")}.`}{" "}
            Highest success: {best.join("; ")}. Which family holds up best depends on the perturbation, and an order of estimates is not a ranking: the separations are in the charts below.
          </p>
          <p className="fine">{f.notes.not_separated}</p>
        </div>
      </div>
      <div className="grid-2">
        <DifferenceChart title="Between families: success on the perturbed set (H4)" rows={between} labelWidth={318}
          label={(d) => `${FAMILY[famSet(d.a)[0]]} − ${FAMILY[famSet(d.b)[0]]}, ${SET[famSet(d.a)[1]]}`} />
        <DifferenceChart title="Declared contrasts: difference in drop (base − perturbed)" rows={contrasts} labelWidth={318} lowerIsBetter
          label={(d) => { const m = CONTRAST.exec(d.a); return m ? `${FAMILY[m[1]]} − ${FAMILY[m[2]]}, ${SET[m[3]]}` : d.a; }}
          note="Here a negative difference means the first family drops less, so cyan marks the first family degrading less." />
      </div>
      <div className="grid-2">
        <DifferenceChart title={`${TASK[task]} ablations of the diffusion policy, on the base set`} rows={abl} labelWidth={220}
          label={(d) => `${ARM[d.a] ?? d.a} − ${ARM[d.b] ?? d.b}`}
          note={`Primary: EMA weights with the DDIM-10 sampler, T_o = 2. ${Object.entries(R.ablations).map(([n, a]) => { const t = a.bias_disclosure ?? a.note ?? ""; return t.startsWith(n) ? t : `${n}: ${t}`; }).join(". ")}.`} />
        <details className="manifests">
          <summary>Evaluation manifests behind these numbers ({evals.length})</summary>
          <p className="fine">Paths in the repository, with the SHA-256 of each evaluation summary.</p>
          <ul>
            {evals.map((e) => (
              <li key={e.eval_run_id}><code>{e.summary}</code> <span className="num" title={e.summary_sha256}>{short(e.summary_sha256, 10)}</span>{" "}
                {FAMILY[e.family] ?? e.family}, seed {e.seed}, {SET[e.suite] ?? e.suite}: {e.successes}/{e.n_episodes}</li>
            ))}
          </ul>
        </details>
      </div>
    </div>
  );
}

function TaskFindings({ f, task }: { f: F; task: string }) {
  const T = f.level1.tasks[task];
  const fams = ORDER.filter((k) => T.success.families[k]);
  const counts = Object.fromEntries(fams.map((k) => {
    const t = T.taxonomy_by_family[k];
    return [k, t ? `${t.success ?? 0}/${total(t)}` : ""];
  }));
  const diffs = T.success.differences;
  const h1 = diffs.filter((d) => d.hypothesis === "H1");
  const rest = diffs.filter((d) => d.hypothesis !== "H1");
  const strips = [...T.strips].sort((a, b) => ORDER.indexOf(a.family) - ORDER.indexOf(b.family) || (a.seed ?? 0) - (b.seed ?? 0))
    .map((s) => ({ name: `${FAMILY[s.family]}${s.seed !== null ? `, seed ${s.seed}` : ""}`, outcomes: s.outcomes }));
  const tax = fams.filter((k) => k !== "random_uniform" || T.taxonomy_by_family[k])
    .map((k) => [k, T.taxonomy_by_family[k]] as [string, Record<string, number>]).filter(([, t]) => t);
  return (
    <div className="findings-task">
      <div className="grid-2 findings-top">
        <EstimateChart title={`${TASK[task]}: held-out success by policy`} rows={fams.map((k) => [k, T.success.families[k]])} counts={counts} />
        <div className="findings-read">
          {task === "lift" ? (
            <p className="callout"><strong>Lift is saturated.</strong> {f.notes.ki_034}</p>
          ) : (
            <p className="callout"><strong>Can separates the families.</strong> Diffusion beats the one-step MLP by{" "}
              {(100 * (rest.find((d) => d.hypothesis === "H2" && d.b === "bc_mlp")?.difference ?? 0)).toFixed(1)} points, an interval that excludes 0,
              but it is not separated from the recurrent (BC-RNN) or the chunked (BC-MLP-chunk) baseline.</p>
          )}
          <p className="fine">{f.notes.not_separated} {f.notes.privilege_level1}</p>
        </div>
      </div>
      <div className="grid-2">
        <DifferenceChart title="Every learned family against random actions (H1)" rows={h1} />
        <DifferenceChart title="Between learned families (H2, H2c, H3)" rows={rest} />
      </div>
      <OutcomeStrip label={`${TASK[task]}: every held-out episode, by checkpoint`} rows={strips} />
      <div className="grid-2">
        <TaxonomyBars title={`${TASK[task]}: how the failures failed`} rows={tax.filter(([k]) => k !== "random_uniform")} />
        <LatencyChart title="Cost of a decision vs held-out success, both tasks" points={Object.entries(f.level1.tasks).flatMap(([tk, TT]) =>
          ORDER.filter((k) => k !== "random_uniform" && TT.cost_latency[k]?.latency_ms_median && TT.success.families[k])
            .map((k) => ({ family: k, task: tk, latency: TT.cost_latency[k].latency_ms_median!, success: TT.success.families[k].estimate })))} />
      </div>
      <Robustness f={f} task={task} />
    </div>
  );
}

export default function Findings() {
  const s = useAsync(loadFindings, []);
  const [task, setTask] = useState("can");
  return (
    <section id="findings" aria-labelledby="findings-title">
      <div className="wrap">
        <div className="section-head">
          <h2 id="findings-title">Research findings</h2>
          <p className="lede">
            Four learned policy families and a random baseline, three training seeds each, and fifty
            held-out starts per task that no model or choice saw before the single frozen run. Intervals are 95 % hierarchical
            paired bootstraps that resample seeds, then starts.
          </p>
        </div>
        {s.status !== "ready" ? <Status state={s} what="the frozen findings" /> : (
          <>
            <div className="tabs" role="group" aria-label="Task">
              {["can", "lift"].map((t) => (
                <button key={t} type="button" className="chip" aria-pressed={task === t} onClick={() => setTask(t)}>{TASK[t]}</button>
              ))}
            </div>
            <TaskFindings key={task} f={s.value} task={task} />
            <SourceTable f={s.value} />
          </>
        )}
      </div>
    </section>
  );
}
