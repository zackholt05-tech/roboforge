// Playback clock shared by viewers. Time is simulation time in seconds (frame k is at k·dt, the
// 20 Hz control rate). The 3D scenes read `clock.t` every animation frame without React renders;
// controls subscribe through useSyncExternalStore. A paused clock never schedules frames.
import { useSyncExternalStore } from "react";

export const SPEEDS = [0.25, 0.5, 1, 1.5, 2] as const;

export class Clock {
  t = 0;
  playing = false;
  speed = 1;
  loop = false;
  private listeners = new Set<() => void>();
  private raf = 0;
  private last = 0;
  private version = 0;

  constructor(public duration: number, public dt: number) {}

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
  getVersion = (): number => this.version;

  private emit(): void {
    this.version++;
    this.listeners.forEach((f) => f());
  }

  setDuration(duration: number, dt: number): void {
    this.duration = duration;
    this.dt = dt;
    this.t = Math.min(this.t, duration);
    this.emit();
  }

  get frame(): number {
    return this.t / this.dt;
  }

  seekFrame(k: number): void {
    this.seek(k * this.dt);
  }

  seek(t: number): void {
    this.t = Math.max(0, Math.min(this.duration, t));
    this.emit();
  }

  step(dk: number): void {
    this.pause();
    this.seekFrame(Math.round(this.frame) + dk);
  }

  play(): void {
    if (this.playing) return;
    if (this.t >= this.duration) this.t = 0;
    this.playing = true;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.tick);
    this.emit();
  }

  pause(): void {
    if (!this.playing) return;
    this.playing = false;
    cancelAnimationFrame(this.raf);
    this.emit();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  restart(): void {
    this.seek(0);
  }

  setSpeed(s: number): void {
    this.speed = s;
    this.emit();
  }

  private tick = (now: number): void => {
    const dtWall = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    this.t += dtWall * this.speed;
    if (this.t >= this.duration) {
      if (this.loop) this.t = 0;
      else {
        this.t = this.duration;
        this.playing = false;
        this.emit();
        return;
      }
    }
    this.emit();
    this.raf = requestAnimationFrame(this.tick);
  };

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.listeners.clear();
  }
}

export function useClock(clock: Clock): Clock {
  useSyncExternalStore(clock.subscribe, clock.getVersion);
  return clock;
}

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
