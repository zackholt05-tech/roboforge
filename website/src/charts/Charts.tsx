// Findings charts. Every mark is a field of findings.json (the frozen results, copied verbatim);
// nothing is computed here except screen positions. Each chart has a visible title above it, a
// text alternative, and keeps its numbers visible as labels, so no value is encoded by colour
// alone. What the marks mean is said once per group of charts by <ChartKey/>, not under each one.
import { useSyncExternalStore, type ReactNode } from "react";
import { CATEGORY, FAMILY, pct, pp } from "../lib/format";

type Est = { estimate: number; ci95: [number, number]; per_seed: number[]; n_seeds: number; n_states: number };
type Diff = { a: string; b: string; difference: number; ci95: [number, number]; separation: string;
  hypothesis?: string; kind?: string };

const FOCUS = new Set(["diffusion_unet", "rgb_diffusion", "pc_diffusion"]);
const colourOf = (k: string) => (FOCUS.has(k.split(":")[0]) ? "var(--signal)" : k.startsWith("random") ? "var(--ivory-faint)" : "var(--sand)");

const PHONE = "(max-width: 720px)";
const media = () => (typeof window !== "undefined" && typeof window.matchMedia === "function" ? window.matchMedia(PHONE) : null);
/** True on a phone-width screen: row charts then put each label above its row instead of beside it. */
function useNarrow(): boolean {
  return useSyncExternalStore(
    (cb) => { const m = media(); m?.addEventListener("change", cb); return () => m?.removeEventListener("change", cb); },
    () => media()?.matches ?? false, () => false);
}

/** A chart with its title above the marks and an optional short note below. On narrow screens
 * row charts switch to a phone layout (`fit`); the others keep their width and scroll in their box. */
function Frame({ title, sub, note, className = "", children }: { title: string; sub?: string; note?: ReactNode;
  className?: string; children: ReactNode }) {
  return (
    <figure className={`chart ${className}`.trim()}>
      <figcaption className="chart-title">{title}{sub && <span className="chart-sub">{sub}</span>}</figcaption>
      <div className="chart-scroll">{children}</div>
      {note && <p className="chart-note">{note}</p>}
    </figure>
  );
}

/** The shared reading key. `estimate`: success-rate rows; `difference`: paired differences. */
export function ChartKey({ estimate = false, difference = false }: { estimate?: boolean; difference?: boolean }) {
  const glyph = (child: ReactNode) => <svg viewBox="0 0 22 12" width={22} height={12} aria-hidden="true">{child}</svg>;
  return (
    <ul className="chart-key" aria-label="How to read these charts">
      {estimate && (
        <>
          <li>{glyph(<circle cx={11} cy={6} r={4.5} fill="var(--ivory)" />)}rate over all seeds and starts</li>
          <li>{glyph(<circle cx={11} cy={6} r={3} fill="none" stroke="var(--ivory)" strokeWidth={1.2} />)}one seed</li>
        </>
      )}
      {(estimate || difference) && (
        <li>{glyph(<line x1={2} x2={20} y1={6} y2={6} stroke="var(--ivory)" strokeWidth={3} strokeLinecap="round" />)}95 % bootstrap interval</li>
      )}
      {difference && (
        <>
          <li>{glyph(<rect x={7} y={2} width={8} height={8} fill="var(--signal)" transform="rotate(45 11 6)" />)}
            separated: the interval excludes 0</li>
          <li>{glyph(<rect x={7} y={2} width={8} height={8} fill="none" stroke="var(--ivory-dim)" strokeWidth={1.5} transform="rotate(45 11 6)" />)}
            not separated: it includes 0, which is not evidence of equivalence</li>
        </>
      )}
    </ul>
  );
}

export function EstimateChart({ rows, title, counts, label, colour }: { rows: [string, Est][];
  title: string; unit?: string; counts?: Record<string, string>; label?: (k: string) => string;
  colour?: (k: string) => string }) {
  const narrow = useNarrow();
  const W = narrow ? 360 : 640, L = narrow ? 12 : label ? 240 : 168, R = narrow ? 126 : 150;
  const rowH = narrow ? 48 : 34, top = narrow ? 30 : 18, H = rows.length * rowH + 34;
  const nm = label ?? ((k: string) => FAMILY[k] ?? k);
  const col = colour ?? colourOf;
  const x = (v: number) => L + v * (W - L - R);
  const first = rows[0]?.[1];
  return (
    <Frame title={title} className={narrow ? "fit" : ""} sub={first ? `${first.n_seeds} training seeds × ${first.n_states} starts` : undefined}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${rows.map(([k, e]) =>
        `${nm(k)} ${pct(e.estimate)} (95 % CI ${pct(e.ci95[0])} to ${pct(e.ci95[1])})`).join("; ")}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={6} y2={H - 24} className="grid-line" />
            <text x={x(t)} y={H - 8} textAnchor="middle" className="tick-label">{t * 100} %</text>
          </g>
        ))}
        {rows.map(([k, e], i) => {
          const y = top + i * rowH;
          return (
            <g key={k}>
              {narrow
                ? <text x={L - 8} y={y - 14} className="row-label">{nm(k)}</text>
                : <text x={L - 12} y={y + 4} textAnchor="end" className="row-label">{nm(k)}</text>}
              <line x1={x(e.ci95[0])} x2={x(e.ci95[1])} y1={y} y2={y} stroke={col(k)} strokeWidth={3} strokeLinecap="round" />
              {e.per_seed.map((s, j) => (
                <circle key={j} cx={x(s)} cy={y + 9} r={2.6} fill="none" stroke={col(k)} strokeWidth={1.2} opacity={0.85} />
              ))}
              <circle cx={x(e.estimate)} cy={y} r={5.5} fill={col(k)} stroke="var(--graphite)" strokeWidth={1.5} />
              <text x={W - R + 12} y={y + 4} className="value-label">{pct(e.estimate)}
                <tspan className="note-label" dx={6}>{counts?.[k] ?? ""}</tspan></text>
            </g>
          );
        })}
      </svg>
    </Frame>
  );
}

export function DifferenceChart({ rows, title, label, labelWidth = 250, lowerIsBetter = false, note, ticksPerSide, paired = false }: {
  rows: Diff[]; title: string; label?: (d: Diff) => string; labelWidth?: number;
  lowerIsBetter?: boolean; note?: string; ticksPerSide?: number;
  /** Rows come in pairs (a reference row, then the row compared with it): draw each pair as a group. */
  paired?: boolean }) {
  // Colour follows which side the interval favours: cyan for the first-named arm, red for the
  // second. For a difference in drops (lowerIsBetter) a negative value favours the first arm.
  const narrow = useNarrow();
  const W = narrow ? 360 : 640, L = narrow ? 12 : labelWidth, R = narrow ? 122 : 128;
  const rowH = narrow ? 48 : 34, top = narrow ? 30 : 18, pairGap = paired ? 10 : 0;
  const H = rows.length * rowH + Math.floor(Math.max(0, rows.length - 1) / 2) * pairGap + 48;
  const raw = Math.max(0.1, ...rows.flatMap((d) => [Math.abs(d.ci95[0]), Math.abs(d.ci95[1])]));
  const perSide = ticksPerSide ?? (W - L - R < 220 ? 2 : W - L - R < 300 ? 3 : 4); // fewer ticks when the plot is narrow
  const step = [0.05, 0.1, 0.25, 0.5].find((st) => raw / st <= perSide) ?? 0.5;
  // A difference of two rates cannot leave [−100, +100] points, so the axis never goes past it.
  const lim = Math.min(Math.ceil((raw * 1.04) / step) * step, Math.max(1, Math.ceil(raw / step) * step));
  const x = (v: number) => L + ((v + lim) / (2 * lim)) * (W - L - R);
  const ticks = Array.from({ length: Math.round((2 * lim) / step) + 1 }, (_, i) => Math.round((-lim + i * step) * 1000) / 1000);
  const name = label ?? ((d: Diff) => `${FAMILY[d.a] ?? d.a} − ${FAMILY[d.b] ?? d.b}`);
  const rowY = (i: number) => top + i * rowH + Math.floor(i / 2) * pairGap;
  return (
    <Frame title={title} note={note} className={narrow ? "fit" : ""}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${rows.map((d) =>
        `${name(d)} ${pp(d.difference)} points, 95 % CI ${pp(d.ci95[0])} to ${pp(d.ci95[1])}, ${d.separation}`).join("; ")}`}>
        {paired && rows.map((_, i) => (i % 2 === 0 ? (
          <rect key={i} x={0} y={rowY(i) - top + 2} width={W} height={2 * rowH - 2} rx={4} className="pair-band" />
        ) : null))}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={6} y2={H - 38} className={t === 0 ? "axis" : "grid-line"} strokeWidth={t === 0 ? 1.5 : 1} />
            <text x={x(t)} y={H - 22} textAnchor="middle" className="tick-label">{t === 0 ? "0" : pp(t, 0)}</text>
          </g>
        ))}
        <text x={x(0)} y={H - 6} textAnchor="middle" className="note-label">difference, percentage points</text>
        {rows.map((d, i) => {
          const y = rowY(i);
          const sep = d.separation.startsWith("separated");
          const favoursFirst = lowerIsBetter ? d.difference < 0 : d.difference > 0;
          const c = !sep ? "var(--ivory-dim)" : favoursFirst ? "var(--signal)" : "var(--error)";
          return (
            <g key={i}>
              <text x={narrow ? L - 8 : L - 12} y={narrow ? y - 14 : y + 4} textAnchor={narrow ? "start" : "end"} className="row-label">
                {d.hypothesis && <tspan className="note-label">{d.hypothesis}  </tspan>}{name(d)}</text>
              <line x1={x(d.ci95[0])} x2={x(d.ci95[1])} y1={y} y2={y} stroke={c} strokeWidth={3} strokeLinecap="round" />
              <rect x={x(d.difference) - 5} y={y - 5} width={10} height={10} fill={sep ? c : "var(--graphite)"} stroke={c} strokeWidth={1.6} transform={`rotate(45 ${x(d.difference)} ${y})`} />
              <text x={W - R + 12} y={y + 4} className="value-label">{pp(d.difference)} pp
                <tspan className="note-label" dx={6}>{sep ? (d.difference > 0 ? "sep. +" : "sep. −") : "not sep."}</tspan></text>
            </g>
          );
        })}
      </svg>
    </Frame>
  );
}

export const CAT_COLOUR: Record<string, string> = {
  success: "var(--signal)",
  engineering_error: "#ffffff",
  gripper_never_closed: "#ff5c6e",
  no_grasp_drift: "#d9824a",
  no_grasp_near: "#e7b04f",
  timeout_holding: "#9d8cff",
  in_goal_bin_not_success: "#7fd1a8",
  dropped: "#f08cc0",
};

export function OutcomeStrip({ rows, highlight, label, caption, labelWidth = 150 }: { rows: { name: string; outcomes:
  { state_id: string; category: string }[] }[]; highlight?: string | null; label: string;
  caption?: string; labelWidth?: number }) {
  const n = Math.max(...rows.map((r) => r.outcomes.length));
  // Wide enough for the longest row name at the label size (a screen position, not a result).
  const L = Math.max(labelWidth, Math.ceil(6.6 * Math.max(...rows.map((r) => r.name.length))) + 16);
  // A long row (the X2 suite has twice the starts of the others) continues on a second line, so
  // every strip on the page is drawn at the same cell size.
  const perLine = n > 60 ? Math.ceil(n / 2) : n, lines = Math.ceil(n / perLine);
  const cell = 11, gap = 2, lineH = cell + 2, rowH = lines * lineH + 4;
  const W = L + perLine * (cell + gap), H = rows.length * rowH + 4;
  const at = (j: number) => ({ x: L + (j % perLine) * (cell + gap), y: Math.floor(j / perLine) * lineH });
  const seen = new Set(rows.flatMap((r) => r.outcomes.map((o) => o.category)));
  const types = Object.keys(CAT_COLOUR).filter((c) => c !== "success" && seen.has(c));
  return (
    <Frame title={label} className="strip" note={`${caption ?? "One square per start, in start order. Filled: success. Outlined: failure, by type."}${lines > 1 ? " Each row continues on a second line." : ""}`}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: Math.round(W * 1.22) }} role="img" aria-label={`${label}: ${rows.map((r) =>
        `${r.name} ${r.outcomes.filter((o) => o.category === "success").length} of ${r.outcomes.length} successes`).join("; ")}`}>
        {rows.map((r, i) => (
          <g key={r.name} transform={`translate(0 ${i * rowH})`}>
            <text x={L - 10} y={cell - 1} textAnchor="end" className="tick-label">{r.name}</text>
            {r.outcomes.map((o, j) => {
              const ok = o.category === "success";
              const hl = o.state_id === highlight;
              return (
                <rect key={o.state_id} x={at(j).x} y={at(j).y} width={cell} height={cell} rx={1.5}
                  fill={ok ? "var(--signal)" : "none"} fillOpacity={ok ? 0.85 : 0}
                  stroke={ok ? "none" : CAT_COLOUR[o.category] ?? "var(--error)"} strokeWidth={ok ? 0 : 1.6}
                  data-highlight={hl || undefined}>
                  <title>{`${o.state_id}: ${CATEGORY[o.category] ?? o.category}`}</title>
                </rect>
              );
            })}
            {highlight && r.outcomes.findIndex((o) => o.state_id === highlight) >= 0 && (
              <rect x={at(r.outcomes.findIndex((o) => o.state_id === highlight)).x - 2.5} y={at(r.outcomes.findIndex((o) => o.state_id === highlight)).y - 2.5}
                width={cell + 5} height={cell + 5} fill="none" stroke="var(--ivory)" strokeWidth={1.2} rx={2} />
            )}
          </g>
        ))}
      </svg>
      {types.length > 0 && (
        <ul className="legend">{types.map((c) => (
          <li key={c}><span className="swatch outline" style={{ borderColor: CAT_COLOUR[c] }} />{CATEGORY[c] ?? c}</li>
        ))}</ul>
      )}
    </Frame>
  );
}

export function TaxonomyBars({ rows, title, label }: { rows: [string, Record<string, number>][]; title: string;
  label?: (k: string) => string }) {
  const nm = label ?? ((k: string) => FAMILY[k] ?? k);
  const cats = Object.keys(CAT_COLOUR).filter((c) => c !== "success" && rows.some(([, t]) => (t[c] ?? 0) > 0));
  const max = Math.max(1, ...rows.map(([, t]) => cats.reduce((s, c) => s + (t[c] ?? 0), 0)));
  const narrow = useNarrow();
  const W = narrow ? 360 : 640, L = narrow ? 4 : label ? 240 : 168, R = narrow ? 44 : 60;
  const rowH = narrow ? 42 : 26, off = narrow ? 18 : 0, H = rows.length * rowH + 8;
  const x = (v: number) => (v / max) * (W - L - R);
  return (
    <Frame title={title} sub="failed episodes by type, all seeds" className={narrow ? "fit" : ""}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${rows.map(([k, t]) =>
        `${nm(k)}: ${cats.map((c) => `${t[c] ?? 0} ${CATEGORY[c]}`).join(", ")}`).join("; ")}`}>
        {rows.map(([k, t], i) => {
          let acc = 0;
          const total = cats.reduce((s, c) => s + (t[c] ?? 0), 0);
          return (
            <g key={k} transform={`translate(0 ${6 + i * rowH})`}>
              {narrow
                ? <text x={L} y={12} className="row-label">{nm(k)}</text>
                : <text x={L - 12} y={12} textAnchor="end" className="row-label">{nm(k)}</text>}
              {cats.map((c) => {
                const v = t[c] ?? 0;
                const r = <rect key={c} x={L + x(acc)} y={2 + off} width={x(v)} height={14} fill={CAT_COLOUR[c]} opacity={0.9}><title>{`${CATEGORY[c]}: ${v}`}</title></rect>;
                acc += v;
                return r;
              })}
              <text x={L + x(total) + 8} y={13 + off} className="value-label">{total}</text>
            </g>
          );
        })}
      </svg>
      <ul className="legend">{cats.map((c) => <li key={c}><span className="swatch" style={{ background: CAT_COLOUR[c] }} />{CATEGORY[c]}</li>)}</ul>
    </Frame>
  );
}

export function LatencyChart({ points, title }: { points: { family: string; task: string; latency: number; success: number }[]; title: string }) {
  const W = 640, H = 280, L = 54, B = 44, T = 16, R = 24;
  const lo = Math.min(0.7, ...points.map((p) => p.success)) - 0.02;
  const lx = (v: number) => L + ((Math.log10(v) + 0.3) / 2.6) * (W - L - R);
  const y = (v: number) => T + (1 - (v - lo) / (1 - lo)) * (H - T - B);
  const fams = [...new Set(points.map((p) => p.family))];
  const yt = [0.7, 0.8, 0.9, 1].filter((t) => t >= lo);
  return (
    <Frame title={title} sub="filled circle: Lift; open square: Can"
      note="Diffusion runs its network once per DDIM step but decides only once per executed chunk, so its cost per control step is lower than this.">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${points.map((p) =>
        `${FAMILY[p.family]} on ${p.task}: ${p.latency.toFixed(1)} ms median, ${pct(p.success)} success`).join("; ")}`}>
        {[1, 10, 100].map((t) => (
          <g key={t}><line x1={lx(t)} x2={lx(t)} y1={T} y2={H - B} className="grid-line" />
            <text x={lx(t)} y={H - B + 16} textAnchor="middle" className="tick-label">{t} ms</text></g>
        ))}
        {yt.map((t) => (
          <g key={t}><line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="grid-line" />
            <text x={L - 8} y={y(t) + 4} textAnchor="end" className="tick-label">{Math.round(t * 100)} %</text></g>
        ))}
        <text x={(W + L) / 2} y={H - 6} textAnchor="middle" className="note-label">median time per decision, observe + act (canonical container, 1 CPU thread, log scale)</text>
        {fams.map((f) => {
          const ps = points.filter((p) => p.family === f);
          const top = ps.reduce((a, b) => (a.success >= b.success ? a : b));
          return (
            <g key={f}>
              {ps.length > 1 && <line x1={lx(ps[0].latency)} x2={lx(ps[1].latency)} y1={y(ps[0].success)} y2={y(ps[1].success)} stroke={colourOf(f)} strokeWidth={1.2} opacity={0.6} />}
              {ps.map((p) => p.task === "lift"
                ? <circle key={p.task} cx={lx(p.latency)} cy={y(p.success)} r={5.5} fill={colourOf(f)} />
                : <rect key={p.task} x={lx(p.latency) - 5} y={y(p.success) - 5} width={10} height={10} fill="var(--graphite)" stroke={colourOf(f)} strokeWidth={2} />)}
              <text x={lx(top.latency) + 10} y={y(top.success) - 8} className="note-label">{FAMILY[f]}</text>
            </g>
          );
        })}
      </svg>
    </Frame>
  );
}
