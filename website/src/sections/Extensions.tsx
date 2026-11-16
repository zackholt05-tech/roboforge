// G. After v1.0.0: the three follow-up studies (v1.1; DECISIONS D-133, D-134). Every number is a
// field of findings_ext.json, which copies blocks of the three hash-pinned results (X1, X3, X2)
// verbatim; nothing is recomputed here (the counts below add frozen per-category counts or
// count verdicts). Each study keeps its own label and is never pooled with the frozen Level 1
// result or with another study. The recorded episodes are X1 and X2 records re-derived from
// their recorded MuJoCo states; which start is shown is fixed by the D-134 rules (selection.json).
import { useMemo, useState } from "react";
import { DifferenceChart, EstimateChart, OutcomeStrip, TaxonomyBars } from "../charts/Charts";
import { ActionChannels, ClipCaption, Provenance, Status } from "../components/ClipParts";
import { decisionFrames, ViewerOrFallback } from "../components/ClipStage";
import { Transport } from "../components/Transport";
import { loadExtClip, loadExtFindings, loadExtSelection, loadExtVerification } from "../lib/data";
import { ARM, ciPp, FAMILY, pct, pp, SET, sepLabel, short, steps1, TASK } from "../lib/format";
import { useAsync, useClockFor } from "../lib/hooks";
import type { Clip, ExtContrast, ExtSelection, FindingsExt } from "../lib/schema";
import { PRESETS, type PresetId } from "../three/cameras";
import { DEFAULT_OPTIONS } from "../three/Viewer";

const BC = ["bc_mlp", "bc_rnn", "bc_mlp_chunk"] as const;
const FAMS = ["diffusion_unet", "bc_mlp_chunk", "bc_rnn", "bc_mlp"];
const X1_ARMS = ["diffusion_unet", "bc_mlp_chunk", "bc_mlp_chunk+clamp"];
const total = (t: Record<string, number>) => Object.values(t).reduce((a, b) => a + b, 0);
const hyp = (c: ExtContrast) => (c.hypothesis === "H2c" ? "H2-c" : c.hypothesis);
const bare = (c: ExtContrast) => c.name.replace(/^\S+\s/, "");
const verdict = (c: ExtContrast) => (c.sign ? `${c.verdict} (${c.sign})` : c.verdict);

function strip(s: { order: string[]; outcomes: boolean[] }) {
  return s.order.map((state_id, i) => ({ state_id, category: s.outcomes[i] ? "success" : "failure" }));
}

const STRIP_NOTE = "One square per start, in the summary's start order: filled, success; outlined, failure (these strips record success only, not the failure type; hover for the start).";

/** Recorded episodes of one rule-selected start, side by side on one clock. */
function Group({ clips, names, study }: { clips: Clip[]; names: string[]; study: string }) {
  const n = Math.max(...clips.map((c) => c.frames.n));
  const clock = useClockFor(n, clips[0].control_dt_s);
  const [preset, setPreset] = useState<PresetId>("frontview");
  const opts = { ...DEFAULT_OPTIONS, successHeight: clips[0].task === "lift" };
  const diff = clips.find((c) => c.family === "diffusion_unet");
  const first = clips.map((c) => c.outcome.first_success_step).find((s) => s !== null) ?? null;
  return (
    <div className="h2h">
      <div className={`h2h-pair n${clips.length}`}>
        {clips.map((c, i) => (
          <figure key={c.id} className={`h2h-side ${c.family === "diffusion_unet" ? "focus" : "base"}`}>
            <figcaption>
              <span className="h2h-name">{names[i]}</span>
              <span className={c.outcome.success ? "ok" : "bad"}>
                {c.outcome.success ? `success at step ${c.outcome.first_success_step}` : `failure, ${c.outcome.steps} steps`}
              </span>
            </figcaption>
            <ViewerOrFallback clip={c} clock={clock} preset={preset} options={opts}
              ariaLabel={`3D reconstruction of ${names[i]} (${study}) on ${c.state_id}`} />
            <ClipCaption clip={c} />
            <ActionChannels clip={c} clock={clock} />
            <details className="manifests">
              <summary>Provenance and training data</summary>
              <Provenance clip={c} />
            </details>
          </figure>
        ))}
      </div>
      <Transport clock={clock} n={n} label={`${clips[0].state_id}, ${clips.length} recorded episodes`}
        successFrame={first} decisions={diff ? decisionFrames(diff) : undefined} />
      <fieldset className="chips">
        <legend>Camera (all views)</legend>
        {PRESETS.filter((p) => !p.mujoco || clips[0].cameras[p.id]).map((p) => (
          <button key={p.id} type="button" className="chip" aria-pressed={preset === p.id}
            onClick={() => setPreset(p.id)}>{p.label}</button>
        ))}
      </fieldset>
      <p className="fine">One clock drives every reconstruction; a shorter episode holds its last recorded frame.{diff ? " Scrubber ticks mark diffusion's decisions." : ""} Each episode was re-derived from its recorded MuJoCo states; nothing was simulated again.</p>
    </div>
  );
}

function CaseView({ ids, nameOf, study, what }: { ids: string[]; nameOf: (c: Clip) => string;
  study: string; what: string }) {
  const s = useAsync(() => Promise.all(ids.map(loadExtClip)), [ids.join("|")]);
  if (s.status !== "ready") return <Status state={s} what={what} />;
  return <Group key={ids.join("|")} clips={s.value} names={s.value.map(nameOf)} study={study} />;
}

function Chips<T extends string>({ legend, items, value, set }: { legend: string;
  items: readonly (readonly [T, string])[]; value: T; set: (v: T) => void }) {
  return (
    <fieldset className="chips">
      <legend>{legend}</legend>
      {items.map(([k, l]) => (
        <button key={k} type="button" className="chip" aria-pressed={value === k} onClick={() => set(k)}>{l}</button>
      ))}
    </fieldset>
  );
}

function ContrastTable({ cs, study }: { cs: ExtContrast[]; study: string }) {
  return (
    <div className="table-scroll">
      <table className="sources ext-contrasts">
        <caption>The four declared Can contrasts, in percentage points: the frozen Level 1 Can PH value and the {study} value, each with its hierarchical paired bootstrap and seed-level (Welch) interval.</caption>
        <thead><tr><th scope="col">Contrast</th><th scope="col">Level 1 Can PH (frozen)</th><th scope="col">seed-level</th>
          <th scope="col">{study}</th><th scope="col">seed-level</th><th scope="col">verdict</th><th scope="col">relation to Level 1</th></tr></thead>
        <tbody>{cs.map((c) => (
          <tr key={c.name}>
            <th scope="row">{c.name}</th>
            <td className="num">{pp(c.frozen.difference)} {ciPp(c.frozen.bootstrap_ci95)}</td>
            <td className="num">{ciPp(c.frozen.seed_level_ci95)}</td>
            <td className="num">{pp(c.difference)} {ciPp(c.bootstrap_ci95)}</td>
            <td className="num">{ciPp(c.seed_level_ci95)}</td>
            <td>{sepLabel(verdict(c))}</td>
            <td>{c.relation}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function contrastRows(cs: ExtContrast[], study: string) {
  return cs.flatMap((c) => [
    { a: bare(c), b: "Level 1", hypothesis: hyp(c), difference: c.frozen.difference,
      ci95: c.frozen.bootstrap_ci95, separation: c.frozen.bootstrap_separation },
    { a: bare(c), b: study, hypothesis: hyp(c), difference: c.difference, ci95: c.bootstrap_ci95,
      separation: c.bootstrap_separation },
  ]);
}

function StudyHead({ id, title, kinds }: { id: string; title: string; kinds: string[] }) {
  return (
    <div className="robustness-head">
      <h3 id={id}>{title}</h3>
      {kinds.map((k) => <span key={k} className="kind measured">{k}</span>)}
    </div>
  );
}

const X1_CASES = [
  ["a_only_diffusion", "Only diffusion succeeds"],
  ["b_all_succeed", "All three succeed"],
  ["c_clamp_changed_outcome", "The bound changed the outcome"],
] as const;

function X1({ f, sel }: { f: FindingsExt; sel: ExtSelection }) {
  const R = f.x1.readings;
  const L = f.x1.tasks.lift;
  const [kase, setKase] = useState<(typeof X1_CASES)[number][0]>("a_only_diffusion");
  const c = sel.cases.find((x) => x.role === `x1:${kase}`);
  const keys = L.sets.flatMap((s) => X1_ARMS.map((a) => `${a}:${s}`)).filter((k) => L.families[k]);
  // Clamped − unclamped success on the same checkpoints, both tasks; the label is baked in.
  const cu = Object.entries(f.x1.tasks).flatMap(([task, t]) => t.x1_differences
    .filter((d) => d.kind?.startsWith("clamped - unclamped"))
    .map((d) => { const [a, s] = d.a.split(":"); return { ...d, a: `${ARM[a.replace("+clamp", "")] ?? a}, ${SET[s] ?? s}, ${TASK[task] ?? task}` }; }));
  const cuSep = cu.filter((d) => d.separation !== "not separated").length;
  const holds = R.primary.separation.startsWith("separated") && R.primary.separation === R.primary.frozen.separation;
  const changed = L.clamp_activity["bc_mlp_chunk+clamp:ood_arm_offset"]?.fraction_actions_changed;
  const drops = [
    { a: "prim", b: "", difference: R.primary.estimate, ci95: R.primary.ci95, separation: R.primary.separation },
    { a: "prim.f", b: "", difference: R.primary.frozen.estimate, ci95: R.primary.frozen.ci95, separation: R.primary.frozen.separation },
    { a: "sup", b: "", difference: R.support.estimate, ci95: R.support.ci95, separation: R.support.separation },
  ];
  const DROP: Record<string, string> = {
    prim: "diffusion − BC-MLP-chunk, clamped (X1)",
    "prim.f": "diffusion − BC-MLP-chunk, unclamped (frozen)",
    sup: "BC-MLP-chunk − BC-MLP-chunk, clamped",
  };
  const rows = X1_ARMS.filter((a) => sel.strips.x1[a]).map((a) => ({ name: `${ARM[a]}, seed ${sel.seed}`, outcomes: strip(sel.strips.x1[a]) }));
  return (
    <div className="robustness ext-study" aria-labelledby="x1-title">
      <StudyHead id="x1-title" title="X1: does the missing action bound explain BC's robustness gap?"
        kinds={[f.x1.label, "validation regime, not held-out"]} />
      <p className="fine">
        The diffusion sampler bounds every action it decodes to the training range; the v1.0.0 BC heads had no such bound (KI-052).
        X1 re-ran the frozen BC checkpoints on the Session 7 robustness starts with the same bound and nothing else changed ({f.x1.decision}).
        {L.families[keys[0]] && ` Each set has ${L.n_states} starts; ${L.families[keys[0]].n_seeds} training seeds per arm.`}
      </p>
      <div className="grid-2 findings-top">
        <EstimateChart title={`${TASK.lift}: success of diffusion and BC-MLP-chunk, with and without the bound`}
          rows={keys.map((k) => [k, L.families[k]])}
          label={(k) => { const [a, s] = k.split(":"); return `${ARM[a] ?? a}, ${SET[s] ?? s}`; }} />
        <div className="findings-read">
          <p className="callout">
            {holds && <><strong>The bound does not close the gap.</strong>{" "}</>}{R.primary.reading}
          </p>
          <p className="fine">
            {changed !== undefined && <>Under the {TASK.lift} arm offset the bound changed {pct(changed)} of BC-MLP-chunk's actions; </>}
            {cuSep} of {cu.length} clamped − unclamped success differences (same checkpoints, both tasks) separate. {R.support.reading}
            {" "}The diffusion side of KI-052, clipping inside the sampler, was not tested.
          </p>
          <p className="fine">Method: {f.x1.method}.</p>
        </div>
      </div>
      <div className="grid-2">
        <DifferenceChart title={`${TASK.lift}, arm offset: difference in drop (base − arm offset)`} rows={drops} labelWidth={318} lowerIsBetter
          label={(d) => DROP[d.a] ?? d.a}
          note="A negative difference means the first arm drops less. The frozen row is the v1.0.0 value, shown for comparison." />
        <DifferenceChart title="Clamped − unclamped success, same checkpoints" rows={cu} labelWidth={318}
          label={(d) => d.a} />
      </div>
      <h4 className="ext-sub">Three arms, one start ({TASK.lift}, arm offset, seed {sel.seed})</h4>
      <p className="fine">Which start is shown is fixed by rule (D-134): the first one, in the summary's start order, where only diffusion succeeds, where all three succeed, and where the bound changed BC-MLP-chunk's outcome. Rule-selected cases, not representative samples.</p>
      <div className="stage-controls"><Chips legend="Case" items={X1_CASES} value={kase} set={setKase} /></div>
      {!c?.clips ? (
        <p className="status">No start fits this case: {c?.reason ?? "not in the selection"}.</p>
      ) : (
        <CaseView ids={c.clips} study="X1" what={`the X1 episodes at ${c.state_id}`} nameOf={(k) => ARM[k.arm ?? ""] ?? k.family} />
      )}
      <OutcomeStrip label={`${TASK.lift}, arm offset: every start for the three seed-${sel.seed} arms (outlined square: the start shown)`}
        rows={rows} highlight={c?.state_id ?? null} caption={STRIP_NOTE} labelWidth={200} />
    </div>
  );
}

function X3({ f }: { f: FindingsExt }) {
  const x = f.x3;
  const frozenSeeds = x.contrasts[0].frozen.n_seeds;
  const added = x.seeds.filter((s) => s >= frozenSeeds);
  const fams = FAMS.filter((k) => x.families[k]);
  return (
    <div className="robustness ext-study" aria-labelledby="x3-title">
      <StudyHead id="x3-title" title="X3: more seeds on Can" kinds={[x.label, "held-out, fresh starts"]} />
      <p className="fine">
        {x.seeds.length} training seeds per family (the frozen {frozenSeeds} plus seeds {added[0]}–{added[added.length - 1]}), evaluated once on {x.suite.n_states} fresh held-out {TASK[x.suite.task] ?? x.suite.task} starts ({x.decision}).
        The frozen Level 1 Can PH values ({frozenSeeds} seeds × {x.contrasts[0].frozen.n_states} starts) stand as reported; X3 is a separate replication beside them.
      </p>
      <div className="grid-2">
        <DifferenceChart title="The four declared Can contrasts: Level 1 (frozen) and X3" rows={contrastRows(x.contrasts, "X3")} labelWidth={330}
          label={(d) => `${d.a}, ${d.b}`} note="Bootstrap intervals; the seed-level intervals are in the table below." />
        <EstimateChart title={`X3: ${TASK.can} success by family`} rows={fams.map((k) => [k, x.families[k]])} />
      </div>
      <ContrastTable cs={x.contrasts} study="X3" />
      <details className="manifests">
        <summary>Readings recorded with the X3 results</summary>
        <ul>{x.contrasts.filter((c) => c.reading).map((c) => <li key={c.name}>{c.reading}</li>)}</ul>
      </details>
    </div>
  );
}

function X2({ f, sel }: { f: FindingsExt; sel: ExtSelection }) {
  const x = f.x2;
  const horizon = x.contrasts[0].frozen.horizon;
  const [fam, setFam] = useState<(typeof BC)[number]>("bc_mlp");
  const [kase, setKase] = useState<"a" | "b" | "c">("a");
  const role = { a: `x2:${fam}:a_only_diffusion`, b: `x2:${fam}:b_only_${fam}`, c: `x2:${fam}:c_both_succeed` }[kase];
  const c = sel.cases.find((k) => k.role === role);
  const fams = FAMS.filter((k) => x.families[k]);
  const counts = Object.fromEntries(fams.map((k) => [k, x.taxonomy[k] ? `${x.taxonomy[k].success ?? 0}/${total(x.taxonomy[k])}` : ""]));
  const tax = fams.filter((k) => x.taxonomy[k]).map((k) => [k, x.taxonomy[k]] as [string, Record<string, number>]);
  const rows = useMemo(() => FAMS.filter((k) => sel.strips.x2[k]).map((k) => ({ name: `${FAMILY[k]}, seed ${sel.seed}`,
    outcomes: strip(sel.strips.x2[k]) })), [sel]);
  const cases = [["a", "Only diffusion succeeds"], ["b", `Only ${FAMILY[fam]} succeeds`], ["c", "Both succeed"]] as const;
  return (
    <div className="robustness ext-study" aria-labelledby="x2-title">
      <StudyHead id="x2-title" title={`${x.study}: demonstrations from several operators`} kinds={["held-out, once", "within-operator only"]} />
      <p className="fine">
        The four families were trained again on the robomimic Can MH demonstrations (mixed quality, several human operators) and evaluated once on {x.suite.n_states} held-out {TASK[x.suite.task] ?? x.suite.task} starts.
        {" "}{x.label}. Outcome: {x.outcome}{horizon !== undefined ? `; Level 1 episodes end at ${horizon} steps, so the two success scales differ` : ""}.
      </p>
      <div className="grid-2 findings-top">
        <EstimateChart title={`${x.study}: success by family`} rows={fams.map((k) => [k, x.families[k]])} counts={counts} />
        <DifferenceChart title={`The four declared Can contrasts: Level 1 Can PH (frozen) and ${x.study}`} rows={contrastRows(x.contrasts, "X2")} labelWidth={300}
          ticksPerSide={2} label={(d) => `${d.a}, ${d.b}`} note="Rows marked Level 1 are the frozen Can PH values; rows marked X2 are X2 (Can MH). Bootstrap intervals; seed-level intervals and each contrast's relation to Level 1 are in the table below." />
      </div>
      <ContrastTable cs={x.contrasts} study={x.study} />
      <div className="grid-2">
        <TaxonomyBars title={`${x.study}: how the failures failed`} rows={tax} />
        <div className="table-scroll">
          <table className="sources">
            <caption>Declared secondary: episode length in steps (steps to success; a failure counts as the full horizon), paired differences with {Math.round(100 * x.censored_episode_length.level)} % hierarchical bootstrap intervals.</caption>
            <thead><tr><th scope="col">Difference</th><th scope="col">steps</th><th scope="col">interval</th><th scope="col">verdict</th></tr></thead>
            <tbody>{x.censored_episode_length.differences.map((d) => (
              <tr key={`${d.a}-${d.b}`}><th scope="row">{FAMILY[d.a] ?? d.a} − {FAMILY[d.b] ?? d.b}</th>
                <td className="num">{steps1(d.difference)}</td><td className="num">[{steps1(d.ci95[0])}, {steps1(d.ci95[1])}]</td>
                <td>{sepLabel(d.separation)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </div>
      <h4 className="ext-sub">Same start, diffusion against one BC family ({x.study}, seed {sel.seed})</h4>
      <p className="fine">Which start is shown is fixed by rule (D-134): the first one, in the summary's start order, where only diffusion succeeds, where only the BC family succeeds, and where both do. Rule-selected cases, not representative samples.</p>
      <div className="stage-controls">
        <Chips legend="BC family" items={BC.map((k) => [k, FAMILY[k]] as const)} value={fam} set={setFam} />
        <Chips legend="Case" items={cases} value={kase} set={setKase} />
      </div>
      {!c?.clips ? (
        <p className="status">No start fits this case: {c?.reason ?? "not in the selection"}.</p>
      ) : (
        <CaseView ids={c.clips} study={x.study} what={`the ${x.study} pair at ${c.state_id}`} nameOf={(k) => FAMILY[k.family] ?? k.family} />
      )}
      <OutcomeStrip label={`${x.study}: every held-out start for the four seed-${sel.seed} checkpoints (outlined square: the start shown)`}
        rows={rows} highlight={c?.state_id ?? null} caption={STRIP_NOTE} />
    </div>
  );
}

/** The common reading, stated only when all three H2-c results are "not separated". */
function H2c({ f }: { f: FindingsExt }) {
  const pick = (cs: ExtContrast[]) => cs.find((c) => c.hypothesis === "H2c");
  const x3 = pick(f.x3.contrasts), x2 = pick(f.x2.contrasts);
  const none = !!x3 && !!x2 && [x3.frozen.bootstrap_separation, x3.frozen.seed_level_separation,
    x3.verdict, x2.verdict].every((v) => v === "not separated");
  if (!none) return null;
  return (
    <p className="callout">
      <strong>H2-c on Can.</strong> In the frozen result, in X3 and in X2 alike, no primary analysis separates the diffusion
      policy from the deterministic MLP that sees the same history and predicts the same action chunks
      ({x3.name}: Level 1 {pp(x3.frozen.difference)}, X3 {pp(x3.difference)}, X2 (Can MH) {pp(x2.difference)} points; each interval includes 0).
    </p>
  );
}

/** The committed viewer check of the extension clips (make web-verify-ext), when present. */
function ViewerCheck() {
  const v = useAsync(loadExtVerification, []);
  if (v.status !== "ready" || v.value === null) return null;
  const fr = v.value.frames;
  const minIou = Math.min(...fr.map((x) => x.robot_iou));
  const maxPx = Math.max(...fr.map((x) => x.robot_centroid_px));
  return (
    <p className="fine">
      Viewer check ({v.value.pass ? "pass" : "FAIL"}): on {fr.length} reference frames of these clips, the 3D reconstruction's
      robot silhouette matches MuJoCo's own segmentation render through the same camera with IoU ≥ {minIou.toFixed(4)} and a
      centroid offset ≤ {maxPx.toFixed(2)} px (thresholds: IoU ≥ {v.value.threshold.robot_iou_min}, ≤ {v.value.threshold.centroid_px_max} px).
    </p>
  );
}

export default function Extensions() {
  const f = useAsync(loadExtFindings, []);
  const sel = useAsync(loadExtSelection, []);
  return (
    <section id="followups" aria-labelledby="followups-title">
      <div className="wrap">
        <div className="section-head">
          <h2 id="followups-title">After v1.0.0: three follow-up studies</h2>
          <p className="lede">
            Three studies ran after the v1.0.0 report, each preregistered and run once. None of them
            changes a frozen number above. Each keeps its own label, and none is pooled with the frozen
            result or with another study. &ldquo;Not separated&rdquo; is never read as equality.
          </p>
        </div>
        {f.status !== "ready" ? <Status state={f} what="the v1.1 follow-up findings" /> :
          sel.status !== "ready" ? <Status state={sel} what="the v1.1 clip selection" /> : (
            <>
              <X1 f={f.value} sel={sel.value} />
              <X3 f={f.value} />
              <X2 f={f.value} sel={sel.value} />
              <H2c f={f.value} />
              <div className="table-scroll">
                <table className="sources">
                  <caption>Every number in this section is read from these files (SHA-256 checked when the site data is built). The v1.1 addendum (paper/addendum/addendum.pdf in the repository) has the tables and caveats.</caption>
                  <thead><tr><th scope="col">Study</th><th scope="col">Pinned results</th><th scope="col">SHA-256</th></tr></thead>
                  <tbody>{f.value.sources.map((s) => (
                    <tr key={s.key}><td>{s.key.toUpperCase()}: {s.label}</td><td>{s.path}</td><td className="num" title={s.sha256}>{short(s.sha256, 16)}…</td></tr>
                  ))}</tbody>
                </table>
              </div>
              <ViewerCheck />
            </>
          )}
      </div>
    </section>
  );
}
