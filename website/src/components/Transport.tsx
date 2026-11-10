// Playback controls: play/pause, frame step, restart, speed and a timeline scrubber with the
// success frame and the policy's decision frames marked. Keyboard: Space toggles, ←/→ step one
// frame, Home restarts, [ and ] change speed (when the control group has focus).
import { useId } from "react";
import { SPEEDS, useClock, type Clock } from "../lib/playback";

export function Transport({ clock, n, successFrame, decisions, label }: { clock: Clock; n: number;
  successFrame: number | null; decisions?: number[]; label: string }) {
  useClock(clock);
  const id = useId();
  const k = Math.round(clock.frame);
  const onKey = (e: React.KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement && e.target.type === "range" &&
      (e.key === "ArrowLeft" || e.key === "ArrowRight")) return;
    const i = SPEEDS.indexOf(clock.speed as (typeof SPEEDS)[number]);
    if (e.key === " " && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); clock.toggle(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); clock.step(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); clock.step(1); }
    else if (e.key === "Home") { e.preventDefault(); clock.restart(); }
    else if (e.key === "[") clock.setSpeed(SPEEDS[Math.max(0, i - 1)]);
    else if (e.key === "]") clock.setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, i + 1)]);
  };
  return (
    <div className="transport" role="group" aria-label={`Playback: ${label}`} onKeyDown={onKey}>
      <div className="transport-buttons">
        <button type="button" className="btn btn-primary" onClick={() => clock.toggle()}
          aria-label={clock.playing ? "Pause" : "Play"}>
          {clock.playing ? <PauseIcon /> : <PlayIcon />}<span>{clock.playing ? "Pause" : "Play"}</span>
        </button>
        <button type="button" className="btn btn-icon" onClick={() => clock.step(-1)} aria-label="Previous frame">
          <StepIcon dir={-1} />
        </button>
        <button type="button" className="btn btn-icon" onClick={() => clock.step(1)} aria-label="Next frame">
          <StepIcon dir={1} />
        </button>
        <button type="button" className="btn btn-icon" onClick={() => clock.restart()} aria-label="Restart">
          <RestartIcon />
        </button>
        <label className="speed" htmlFor={`${id}-speed`}>
          <span className="visually-hidden">Playback speed</span>
          <select id={`${id}-speed`} value={clock.speed} onChange={(e) => clock.setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
        </label>
      </div>
      <div className="scrubber">
        <label htmlFor={`${id}-seek`} className="visually-hidden">Frame</label>
        <div className="scrubber-track">
          {decisions?.map((d) => <span key={d} className="tick" style={{ left: `${(100 * d) / Math.max(1, n - 1)}%` }} />)}
          {successFrame !== null && (
            <span className="tick tick-success" title={`success at frame ${successFrame}`}
              style={{ left: `${(100 * successFrame) / Math.max(1, n - 1)}%` }} />
          )}
          <input id={`${id}-seek`} type="range" min={0} max={n - 1} step={1} value={k}
            aria-valuetext={`frame ${k} of ${n - 1}, ${(k * clock.dt).toFixed(2)} seconds`}
            onChange={(e) => { clock.pause(); clock.seekFrame(Number(e.target.value)); }} />
        </div>
        <output className="readout" htmlFor={`${id}-seek`}>
          <span className="num">{(k * clock.dt).toFixed(2)} s</span>
          <span className="dim"> frame {k} / {n - 1}</span>
        </output>
      </div>
    </div>
  );
}

const PlayIcon = () => (<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor" /></svg>);
const PauseIcon = () => (<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor" /></svg>);
const StepIcon = ({ dir }: { dir: 1 | -1 }) => (
  <svg viewBox="0 0 16 16" aria-hidden="true" style={{ transform: dir < 0 ? "scaleX(-1)" : undefined }}>
    <path d="M3 3v10l7-5zM11 3h2v10h-2z" fill="currentColor" /></svg>);
const RestartIcon = () => (
  <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 2v4.5h4.5M3.6 6A5 5 0 1 1 3 9" fill="none" stroke="currentColor" strokeWidth="1.6" /></svg>);
