// A. Hero, "Learning to move": the recorded held-out Lift rollout of the diffusion policy chosen by
// D-089 rule 2 (the Level 1 freeze's first success of Lift diffusion_unet), looping with a slow
// camera drift. With reduced motion it stays on the first frame until played.
import { useEffect, useState } from "react";
import { ClipCaption, ProvenanceDetails, Status } from "../components/ClipParts";
import { useClip, ViewerOrFallback } from "../components/ClipStage";
import { loadFindings } from "../lib/data";
import { ciPp, FAMILY, pp, sepLabel } from "../lib/format";
import { useAsync, useClockFor } from "../lib/hooks";
import { prefersReducedMotion, useClock } from "../lib/playback";
import { DEFAULT_OPTIONS } from "../three/Viewer";
import { byRole, useRegistry } from "./registry";

type Diff = { a: string; b: string; difference: number; ci95: [number, number]; separation: string;
  hypothesis?: string; kind?: string };

/** What the research found, in three recorded differences. Every value and verdict is a field of
 * findings.json; the sentence under each number is chosen by the recorded verdict, so a changed
 * verdict changes the sentence. */
function Glance() {
  const f = useAsync(loadFindings, []);
  if (f.status !== "ready") return null;
  const can: Diff[] = f.value.level1.tasks.can?.success.differences ?? [];
  const h5 = (f.value.level2.placement.bootstrap.differences as Diff[]).find((d) => (d.kind ?? "").startsWith("PRIMARY H5"));
  const cards: { where: string; a: string; b: string; d: Diff | undefined }[] = [
    { where: "Can, held-out starts", a: FAMILY.diffusion_unet, b: FAMILY.bc_mlp, d: can.find((d) => d.hypothesis === "H2" && d.b === "bc_mlp") },
    { where: "Can, held-out starts", a: FAMILY.diffusion_unet, b: FAMILY.bc_mlp_chunk, d: can.find((d) => d.hypothesis === "H2c") },
    { where: "Lift, cube moved off the training square", a: "Point-cloud diffusion (DP3-inspired)", b: "visual diffusion", d: h5 },
  ];
  const read = (a: string, b: string, d: Diff) => (!d.separation.startsWith("separated")
    ? "Not separated: the interval includes 0, which is not evidence that the two are equal."
    : `${a} succeeds ${d.difference > 0 ? "more" : "less"} often than ${b}.`);
  return (
    <div className="glance" aria-labelledby="glance-title">
      <div className="wrap">
        <h2 id="glance-title" className="glance-title">What the research found</h2>
        <ul className="glance-cards">
          {cards.map(({ where, a, b, d }) => d && (
            <li key={`${a}-${b}`}>
              <span className="glance-where">{where}</span>
              <span className="glance-what">{a} − {b}</span>
              <span className={`glance-number num ${!d.separation.startsWith("separated") ? "dim" : d.difference > 0 ? "ok" : "bad"}`}>
                {pp(d.difference)}<span className="unit"> points</span></span>
              <span className="glance-ci num">95 % interval {ciPp(d.ci95)}, {sepLabel(d.separation)}</span>
              <span className="glance-read">{read(a, b, d)}</span>
            </li>
          ))}
        </ul>
        <p className="fine">
          Simulation only. The first two results come from policies that read exact simulator state (privileged information),
          and include a confound: the diffusion sampler bounds its actions, the BC heads do not (KI-052). Lift is saturated
          (KI-034), so the family comparison rests on Can.
          {" "}<a href="#findings">All findings</a>
        </p>
      </div>
    </div>
  );
}

export function Hero() {
  const reg = useRegistry();
  const hero = byRole(reg, "hero") ?? reg.clips[0];
  const s = useClip(hero.id);
  const clip = s.status === "ready" ? s.value : null;
  const clock = useClockFor(clip?.frames.n, clip?.control_dt_s);
  const reduced = prefersReducedMotion();
  const [moving, setMoving] = useState(!reduced);
  useClock(clock);
  useEffect(() => {
    if (!clip) return;
    clock.loop = true;
    if (!reduced) clock.play();
  }, [clip, clock, reduced]);
  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <div className="hero-stage">
      {clip ? (
        <>
          <ViewerOrFallback clip={clip} clock={clock} preset="hero" continuous={moving && clock.playing} tag={false}
            options={{ ...DEFAULT_OPTIONS, trail: true, autoOrbit: moving }}
            ariaLabel={`Recorded MuJoCo rollout: ${clip.policy.label}, start ${clip.state_id}`} />
          <div className="hero-meta">
            <ClipCaption clip={clip} explain />
            <ProvenanceDetails clip={clip} />
          </div>
        </>
      ) : <div className="wrap" style={{ padding: 40 }}><Status state={s} what="the hero rollout" /></div>}
      <div className="hero-shade" />
      <div className="hero-copy">
        <h1 id="hero-title">Learning to move</h1>
        <p className="lede">
          A diffusion policy and three behaviour-cloning baselines learned two manipulation tasks
          from human demonstrations, then drove a simulated robot arm from starts they had never
          seen. Every rollout on this page is a recorded evaluation episode.
        </p>
        <div className="hero-actions">
          <a className="btn btn-primary" href="#lab">Open the lab</a>
          <a className="btn" href="#findings">Read the research</a>
        </div>
      </div>
      {clip && (
        <div className="hero-controls">
          <button type="button" className="btn" onClick={() => { clock.toggle(); }}
            aria-label={clock.playing ? "Pause the hero rollout" : "Play the hero rollout"}>
            {clock.playing ? "Pause" : "Play"}
          </button>
          <button type="button" className="btn" aria-pressed={moving} onClick={() => setMoving((m) => !m)}>
            Camera drift {moving ? "on" : "off"}
          </button>
        </div>
      )}
      </div>
      <Glance />
    </section>
  );
}
