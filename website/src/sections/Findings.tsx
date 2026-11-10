// F. Research findings: the Level 1 held-out results, drawn from findings.json, which copies the
// frozen results.json and taxonomy.json blocks verbatim. Nothing here is recomputed; summed
// counts are the frozen taxonomy's per-category counts added over seeds.
import { useState } from "react";
import { DifferenceChart, EstimateChart, LatencyChart, OutcomeStrip, TaxonomyBars } from "../charts/Charts";
import { Status } from "../components/ClipParts";
import { loadFindings } from "../lib/data";
import { FAMILY, TASK, short } from "../lib/format";
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
