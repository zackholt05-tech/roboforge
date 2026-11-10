// One recorded clip in the 3D viewer with its controls, video, provenance and channels. On a
// phone, without WebGL, or with reduced motion the simulator's own video leads and the 3D
// reconstruction loads on request (MASTER_SPEC §8.3 fallback).
import { useEffect, useState, type ReactNode } from "react";
import { loadClip } from "../lib/data";
import { useAsync, useClockFor, useMedia, useMeshes, useNearScreen, webglAvailable } from "../lib/hooks";
import { prefersReducedMotion, type Clock } from "../lib/playback";
import type { Clip } from "../lib/schema";
import { PRESETS, type PresetId } from "../three/cameras";
import { DEFAULT_OPTIONS, Viewer, type ViewerOptions } from "../three/Viewer";
import { ActionChannels, ClipCaption, InsetVideo, Provenance, Status } from "./ClipParts";
import { Transport } from "./Transport";

export function decisionFrames(clip: Clip): number[] {
  return clip.actions.chunk_offset.flatMap((o, t) => (o === 0 ? [t] : []));
}

export function useClip(id: string | null) {
  return useAsync(() => (id ? loadClip(id) : Promise.reject(new Error("no clip selected"))), [id]);
}

export function ViewerOrFallback({ clip, clock, preset, options, ariaLabel, children, continuous, focus, focusOffset }: {
  clip: Clip; clock: Clock; preset: PresetId; options: ViewerOptions; ariaLabel: string;
  children?: ReactNode; continuous?: boolean; focus?: number[]; focusOffset?: number[] }) {
  const meshes = useMeshes();
  const small = useMedia("(max-width: 720px)");
  const [want3d, setWant3d] = useState(false);
  const [ref, near] = useNearScreen<HTMLDivElement>();
  const gl = typeof document !== "undefined" && webglAvailable();
  const show3d = gl && (!small || want3d);
  return (
    <div className="stage" ref={ref}>
      {show3d ? (
        <>
          {near && <Viewer clip={clip} lib={meshes.lib} clock={clock} preset={preset} options={options}
            ariaLabel={ariaLabel} continuous={continuous} focus={focus} focusOffset={focusOffset}>{children}</Viewer>}
          {meshes.error && (
            <p className="stage-note" role="note">Robot meshes are not available ({meshes.error}). Links are drawn as their bounding boxes, an approximation.</p>
          )}
          {meshes.loading && <p className="stage-note" role="status">Loading robot meshes (3 MB)…</p>}
        </>
      ) : (
        <div className="stage-fallback">
          <InsetVideo clip={clip} clock={clock} large />
          {gl ? (
            <button type="button" className="btn" onClick={() => setWant3d(true)}>Load the 3D reconstruction</button>
          ) : <p className="stage-note">WebGL is not available, so the simulator video is shown instead of the 3D view.</p>}
        </div>
      )}
    </div>
  );
}

export function ClipStage({ clipId, title, category, compact = false, autoplay = false,
  initialPreset = "orbit", extra }: { clipId: string; title?: string; category?: string | null;
  compact?: boolean; autoplay?: boolean; initialPreset?: PresetId; extra?: ReactNode }) {
  const s = useClip(clipId);
  const clip = s.status === "ready" ? s.value : null;
  const clock = useClockFor(clip?.frames.n, clip?.control_dt_s);
  const [preset, setPreset] = useState<PresetId>(initialPreset);
  const [opt, setOpt] = useState<ViewerOptions>({ ...DEFAULT_OPTIONS, successHeight: true });
  const [viewNonce, setViewNonce] = useState(0);
  useEffect(() => {
    if (clip && autoplay && !prefersReducedMotion()) clock.play();
  }, [clip, autoplay, clock]);
  if (!clip) return <Status state={s} what={`recorded clip ${clipId}`} />;
  const toggle = (k: keyof ViewerOptions) => setOpt((o) => ({ ...o, [k]: !o[k] }));
  return (
    <div className={compact ? "clip-stage compact" : "clip-stage"}>
      {title && <h3 className="clip-title">{title}</h3>}
      <ClipCaption clip={clip} category={category} />
      <div className="stage-wrap">
        <ViewerOrFallback key={viewNonce} clip={clip} clock={clock} preset={preset} options={opt}
          ariaLabel={`3D reconstruction of ${clip.id}: ${clip.label}`}>{extra}</ViewerOrFallback>
        {!compact && <InsetVideo clip={clip} clock={clock} />}
      </div>
      <Transport clock={clock} n={clip.frames.n} label={clip.id}
        successFrame={clip.outcome.first_success_step} decisions={decisionFrames(clip)} />
      {!compact && (
        <div className="stage-controls">
          <fieldset className="chips">
            <legend>Camera</legend>
            {PRESETS.filter((p) => !p.mujoco || clip.cameras[p.id]).map((p) => (
              <button key={p.id} type="button" className="chip" aria-pressed={preset === p.id}
                onClick={() => setPreset(p.id)}>{p.label}</button>
            ))}
            <button type="button" className="chip" onClick={() => setViewNonce((n) => n + 1)}>Reset view</button>
          </fieldset>
          <fieldset className="chips">
            <legend>Overlays</legend>
            {([["trail", "End-effector trail"], ["follow", "Follow gripper"], ["labels", "Link labels"],
              ["axes", "MuJoCo axes"], ...(clip.task === "lift" ? [["successHeight", "Success height"]] : [])] as
              [keyof ViewerOptions, string][]).map(([k, l]) => (
              <button key={k} type="button" className="chip" aria-pressed={!!opt[k]} onClick={() => toggle(k)}>{l}</button>
            ))}
          </fieldset>
        </div>
      )}
      {!compact && (
        <div className="stage-detail">
          <ActionChannels clip={clip} clock={clock} />
          <Provenance clip={clip} />
        </div>
      )}
      <p className="fine">{clip.label}. {clip.success_rule.text && `Success rule: ${clip.success_rule.text}.`}</p>
    </div>
  );
}
