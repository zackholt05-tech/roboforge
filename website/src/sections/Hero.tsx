// A. Hero, "Learning to move": the recorded held-out Lift rollout of the diffusion policy chosen by
// D-089 rule 2 (the Level 1 freeze's first success of Lift diffusion_unet), looping with a slow
// camera drift. With reduced motion it stays on the first frame until played.
import { useEffect, useState } from "react";
import { ClipCaption, Status } from "../components/ClipParts";
import { useClip, ViewerOrFallback } from "../components/ClipStage";
import { useClockFor } from "../lib/hooks";
import { prefersReducedMotion, useClock } from "../lib/playback";
import { DEFAULT_OPTIONS } from "../three/Viewer";
import { byRole, useRegistry } from "./registry";

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
      {clip ? (
        <>
          <ViewerOrFallback clip={clip} clock={clock} preset="hero" continuous={moving && clock.playing}
            options={{ ...DEFAULT_OPTIONS, trail: true, autoOrbit: moving }}
            ariaLabel={`Recorded MuJoCo rollout: ${clip.policy.label}, start ${clip.state_id}`} />
          <ClipCaption clip={clip} />
        </>
      ) : <div className="wrap" style={{ padding: 40 }}><Status state={s} what="the hero rollout" /></div>}
      <div className="hero-shade" />
      <div className="hero-copy">
        <h1 id="hero-title">Learning to move</h1>
        <p className="lede">
          A diffusion policy and three behaviour-cloning baselines learned from 180 human
          demonstrations per task, then drove a simulated Franka Panda arm from 50 starts they had
          never seen. Every rollout on this page was recorded in RoboForge's frozen evaluations;
          proposals and schematics are labelled as such.
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
    </section>
  );
}
