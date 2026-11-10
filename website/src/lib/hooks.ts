// React hooks around the data loaders: explicit loading / error / ready states, so a missing
// artifact always shows a message with the fix instead of an empty canvas.
import { useEffect, useMemo, useRef, useState } from "react";
import { loadMeshLibrary, type MeshLibrary } from "../three/meshes";
import { Clock } from "./playback";

export type Async<T> = { status: "loading" } | { status: "error"; error: string } |
  { status: "ready"; value: T };

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): Async<T> {
  const [s, setS] = useState<Async<T>>({ status: "loading" });
  useEffect(() => {
    let live = true;
    setS({ status: "loading" });
    fn().then((value) => live && setS({ status: "ready", value }),
      (e: Error) => live && setS({ status: "error", error: e.message }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return s;
}

/** Meshes are optional: without them the viewer draws bounding boxes and says so. */
export function useMeshes(): { lib: MeshLibrary | null; error: string | null; loading: boolean } {
  const s = useAsync(loadMeshLibrary, []);
  return { lib: s.status === "ready" ? s.value : null, error: s.status === "error" ? s.error : null,
    loading: s.status === "loading" };
}

export function useClockFor(nFrames: number | undefined, dt: number | undefined): Clock {
  const clock = useMemo(() => new Clock(1, 0.05), []);
  useEffect(() => {
    if (nFrames && dt) clock.setDuration((nFrames - 1) * dt, dt);
  }, [clock, nFrames, dt]);
  useEffect(() => () => clock.dispose(), [clock]);
  return clock;
}

/** True while the element is within one viewport of the screen. */
export function useNearScreen<T extends Element>(margin = "100% 0px"): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") { setNear(true); return; }
    const io = new IntersectionObserver(([e]) => setNear(e.isIntersecting), { rootMargin: margin });
    io.observe(el);
    return () => io.disconnect();
  }, [margin]);
  return [ref, near];
}

let webgl: boolean | null = null;
/** Probed once per page; the probe context is released immediately. */
export function webglAvailable(): boolean {
  if (webgl !== null) return webgl;
  try {
    const c = document.createElement("canvas");
    const gl = (c.getContext("webgl2") || c.getContext("webgl")) as WebGLRenderingContext | null;
    webgl = !!gl;
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    webgl = false;
  }
  if (new URLSearchParams(window.location.search).has("nowebgl")) webgl = false;
  return webgl;
}

export function useMedia(query: string): boolean {
  const [m, setM] = useState(() => typeof window !== "undefined" && window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setM(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return m;
}
