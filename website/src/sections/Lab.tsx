// C. The robotic lab: every Level 1 held-out episode that the Level 1 freeze's example rule
// selects (first success and first failure per task and family, plus the first episode of each
// failure category), in the articulated viewer.
import { useMemo, useState } from "react";
import { ClipStage } from "../components/ClipStage";
import { CATEGORY, FAMILY, FAMILY_NOTE, TASK } from "../lib/format";
import { roleText, useRegistry } from "./registry";

const ORDER = ["diffusion_unet", "bc_mlp_chunk", "bc_rnn", "bc_mlp", "random_uniform"];

export function Lab() {
  const reg = useRegistry();
  const lab = useMemo(() => reg.clips.filter((c) => c.roles.includes("lab")), [reg]);
  const tasks = [...new Set(lab.map((c) => c.task))].sort();
  const [task, setTask] = useState(tasks[0] ?? "lift");
  const families = ORDER.filter((f) => lab.some((c) => c.task === task && c.family === f));
  const [family, setFamily] = useState("diffusion_unet");
  const fam = families.includes(family) ? family : families[0];
  const eps = lab.filter((c) => c.task === task && c.family === fam)
    .sort((a, b) => Number(b.outcome.success) - Number(a.outcome.success) || a.state_id.localeCompare(b.state_id));
  const [pick, setPick] = useState<string | null>(null);
  const chosen = eps.find((c) => c.id === pick) ?? eps[0];
  const category = (c: (typeof eps)[number]) =>
    c.outcome.success ? "success" : c.roles.find((r) => r.startsWith("gallery:category:"))?.replace("gallery:category:", "") ?? null;
  return (
    <section id="lab" aria-labelledby="lab-title">
      <div className="wrap">
        <div className="section-head">
          <h2 id="lab-title">The robotic lab</h2>
          <p className="lede">
            Recorded held-out episodes, reconstructed in 3D from the simulator's saved states. The
            episodes are not picked by eye: the frozen protocol names, for each task and policy, the
            first success and the first failure by start ID, and the first episode of each failure type.
          </p>
        </div>
        {lab.length === 0 ? <p className="status">No lab episodes in the exported data.</p> : (
          <>
            <div className="stage-controls" style={{ marginBottom: 18 }}>
              <fieldset className="chips">
                <legend>Task</legend>
                {tasks.map((t) => (
                  <button key={t} type="button" className="chip" aria-pressed={task === t}
                    onClick={() => { setTask(t); setPick(null); }}>{TASK[t]}</button>
                ))}
              </fieldset>
              <fieldset className="chips">
                <legend>Policy (trained checkpoint)</legend>
                {families.map((f) => (
                  <button key={f} type="button" className="chip" aria-pressed={fam === f}
                    onClick={() => { setFamily(f); setPick(null); }} title={FAMILY_NOTE[f]}>{FAMILY[f]}</button>
                ))}
              </fieldset>
              <fieldset className="chips">
                <legend>Episode</legend>
                {eps.map((c) => (
                  <button key={c.id} type="button" className="chip" aria-pressed={chosen?.id === c.id}
                    onClick={() => setPick(c.id)}>
                    <span className={c.outcome.success ? "ok" : "bad"}>{c.outcome.success ? "●" : "○"}</span>{" "}
                    {c.state_id.replace(`${task}_final_test_`, "start ")}, seed {c.seed}, {roleText(c)}
                  </button>
                ))}
              </fieldset>
            </div>
            {chosen && (
              <ClipStage key={chosen.id} clipId={chosen.id} category={category(chosen)}
                title={`${FAMILY[chosen.family]} on ${TASK[chosen.task]}: ${chosen.outcome.success ? "success" : CATEGORY[category(chosen) ?? ""] ?? "failure"}`} />
            )}
            <p className="fine">{FAMILY[fam]}: {FAMILY_NOTE[fam]}. Filled dot: success; open dot: failure.</p>
          </>
        )}
      </div>
    </section>
  );
}
