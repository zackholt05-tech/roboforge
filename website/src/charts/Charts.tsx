// Findings charts. Every mark is a field of findings.json (the frozen results, copied verbatim);
// nothing is computed here except screen positions. Each chart has a text alternative and keeps
// its numbers visible as labels, so no value is encoded by colour alone.
import { CATEGORY, FAMILY, pct, pp } from "../lib/format";

type Est = { estimate: number; ci95: [number, number]; per_seed: number[]; n_seeds: number; n_states: number };
type Diff = { a: string; b: string; difference: number; ci95: [number, number]; separation: string;
  hypothesis?: string; kind?: string };

const FOCUS = new Set(["diffusion_unet", "rgb_diffusion", "pc_diffusion"]);
const colourOf = (k: string) => (FOCUS.has(k.split(":")[0]) ? "var(--signal)" : k.startsWith("random") ? "var(--ivory-faint)" : "var(--sand)");

export function EstimateChart({ rows, title, unit = "success", counts, label, colour }: { rows: [string, Est][];
  title: string; unit?: string; counts?: Record<string, string>; label?: (k: string) => string;
  colour?: (k: string) => string }) {
  const W = 640, L = label ? 240 : 168, R = 150, rowH = 34, H = rows.length * rowH + 34;
  const nm = label ?? ((k: string) => FAMILY[k] ?? k);
  const col = colour ?? colourOf;
  const x = (v: number) => L + v * (W - L - R);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${rows.map(([k, e]) =>
        `${nm(k)} ${pct(e.estimate)} (95 % CI ${pct(e.ci95[0])} to ${pct(e.ci95[1])})`).join("; ")}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={6} y2={H - 24} className="grid-line" />
            <text x={x(t)} y={H - 8} textAnchor="middle" className="tick-label">{t * 100} %</text>
          </g>
        ))}
        {rows.map(([k, e], i) => {
          const y = 18 + i * rowH;
          return (
            <g key={k}>
              <text x={L - 12} y={y + 4} textAnchor="end" className="row-label">{nm(k)}</text>
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
      <figcaption>{title}. Dot: estimate over {rows[0]?.[1].n_seeds} training seeds × {rows[0]?.[1].n_states} starts; bar: 95 % hierarchical paired bootstrap interval; rings: each seed's {unit} rate.</figcaption>
    </figure>
  );
}

export function DifferenceChart({ rows, title, label }: { rows: Diff[]; title: string;
  label?: (d: Diff) => string }) {
  const W = 640, L = 250, R = 128, rowH = 34, H = rows.length * rowH + 34;
  const raw = Math.max(0.1, ...rows.flatMap((d) => [Math.abs(d.ci95[0]), Math.abs(d.ci95[1])]));
  const step = [0.05, 0.1, 0.25, 0.5].find((st) => raw / st <= 4) ?? 0.5;
  const lim = Math.ceil((raw * 1.04) / step) * step;
  const x = (v: number) => L + ((v + lim) / (2 * lim)) * (W - L - R);
  const ticks = Array.from({ length: Math.round((2 * lim) / step) + 1 }, (_, i) => Math.round((-lim + i * step) * 1000) / 1000);
  const name = label ?? ((d: Diff) => `${FAMILY[d.a] ?? d.a} − ${FAMILY[d.b] ?? d.b}`);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${rows.map((d) =>
        `${name(d)} ${pp(d.difference)} points, 95 % CI ${pp(d.ci95[0])} to ${pp(d.ci95[1])}, ${d.separation}`).join("; ")}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={6} y2={H - 24} className={t === 0 ? "axis" : "grid-line"} strokeWidth={t === 0 ? 1.5 : 1} />
            <text x={x(t)} y={H - 8} textAnchor="middle" className="tick-label">{t === 0 ? "0" : pp(t, 0)}</text>
          </g>
        ))}
        {rows.map((d, i) => {
          const y = 18 + i * rowH;
          const sep = d.separation.startsWith("separated");
          const c = !sep ? "var(--ivory-dim)" : d.difference > 0 ? "var(--signal)" : "var(--error)";
          return (
            <g key={i}>
              <text x={L - 12} y={y + 4} textAnchor="end" className="row-label">
                {d.hypothesis && <tspan className="note-label">{d.hypothesis}  </tspan>}{name(d)}</text>
              <line x1={x(d.ci95[0])} x2={x(d.ci95[1])} y1={y} y2={y} stroke={c} strokeWidth={3} strokeLinecap="round" />
              <rect x={x(d.difference) - 5} y={y - 5} width={10} height={10} fill={sep ? c : "var(--graphite)"} stroke={c} strokeWidth={1.6} transform={`rotate(45 ${x(d.difference)} ${y})`} />
              <text x={W - R + 12} y={y + 4} className="value-label">{pp(d.difference)} pp
                <tspan className="note-label" dx={6}>{sep ? (d.difference > 0 ? "sep. +" : "sep. −") : "not sep."}</tspan></text>
            </g>
          );
        })}
      </svg>
      <figcaption>{title}. Diamond: paired difference in percentage points; bar: 95 % interval. Filled: the interval excludes 0 ("separated"); hollow: it includes 0 ("not separated", which is not evidence of equivalence).</figcaption>
    </figure>
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

export function OutcomeStrip({ rows, highlight, label }: { rows: { name: string; outcomes:
  { state_id: string; category: string }[] }[]; highlight?: string | null; label: string }) {
  const n = Math.max(...rows.map((r) => r.outcomes.length));
  const cell = 11, gap = 2, L = 150, W = L + n * (cell + gap), H = rows.length * (cell + 6) + 4;
  return (
    <figure className="chart strip">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}: ${rows.map((r) =>
        `${r.name} ${r.outcomes.filter((o) => o.category === "success").length} of ${r.outcomes.length} successes`).join("; ")}`}>
        {rows.map((r, i) => (
          <g key={r.name} transform={`translate(0 ${i * (cell + 6)})`}>
            <text x={L - 10} y={cell - 1} textAnchor="end" className="tick-label">{r.name}</text>
            {r.outcomes.map((o, j) => {
              const ok = o.category === "success";
              const hl = o.state_id === highlight;
              return (
                <rect key={o.state_id} x={L + j * (cell + gap)} y={0} width={cell} height={cell} rx={1.5}
                  fill={ok ? "var(--signal)" : "none"} fillOpacity={ok ? 0.85 : 0}
                  stroke={ok ? "none" : CAT_COLOUR[o.category] ?? "var(--error)"} strokeWidth={ok ? 0 : 1.6}
                  data-highlight={hl || undefined}>
                  <title>{`${o.state_id}: ${CATEGORY[o.category] ?? o.category}`}</title>
                </rect>
              );
            })}
            {highlight && r.outcomes.findIndex((o) => o.state_id === highlight) >= 0 && (
              <rect x={L + r.outcomes.findIndex((o) => o.state_id === highlight) * (cell + gap) - 2.5} y={-2.5}
                width={cell + 5} height={cell + 5} fill="none" stroke="var(--ivory)" strokeWidth={1.2} rx={2} />
            )}
          </g>
        ))}
      </svg>
      <figcaption>{label}. One square per held-out start, in start order: filled, success; outlined, failure (colour = failure type; hover for the start and type).</figcaption>
    </figure>
  );
}

export function TaxonomyBars({ rows, title, label }: { rows: [string, Record<string, number>][]; title: string;
  label?: (k: string) => string }) {
  const nm = label ?? ((k: string) => FAMILY[k] ?? k);
  const cats = Object.keys(CAT_COLOUR).filter((c) => c !== "success" && rows.some(([, t]) => (t[c] ?? 0) > 0));
  const max = Math.max(1, ...rows.map(([, t]) => cats.reduce((s, c) => s + (t[c] ?? 0), 0)));
  const W = 640, L = label ? 240 : 168, R = 60, rowH = 26, H = rows.length * rowH + 8;
  const x = (v: number) => (v / max) * (W - L - R);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${rows.map(([k, t]) =>
        `${nm(k)}: ${cats.map((c) => `${t[c] ?? 0} ${CATEGORY[c]}`).join(", ")}`).join("; ")}`}>
        {rows.map(([k, t], i) => {
          let acc = 0;
          const total = cats.reduce((s, c) => s + (t[c] ?? 0), 0);
          return (
            <g key={k} transform={`translate(0 ${6 + i * rowH})`}>
              <text x={L - 12} y={12} textAnchor="end" className="row-label">{nm(k)}</text>
              {cats.map((c) => {
                const v = t[c] ?? 0;
                const r = <rect key={c} x={L + x(acc)} y={2} width={x(v)} height={14} fill={CAT_COLOUR[c]} opacity={0.9}><title>{`${CATEGORY[c]}: ${v}`}</title></rect>;
                acc += v;
                return r;
              })}
              <text x={L + x(total) + 8} y={13} className="value-label">{total}</text>
            </g>
          );
        })}
      </svg>
      <ul className="legend">{cats.map((c) => <li key={c}><span className="swatch" style={{ background: CAT_COLOUR[c] }} />{CATEGORY[c]}</li>)}</ul>
      <figcaption>{title}. Failed episodes by failure type, all seeds (the frozen taxonomy's counts).</figcaption>
    </figure>
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
    <figure className="chart">
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
      <figcaption>{title}. Filled circle: Lift; open square: Can; a line joins one family's two tasks. Diffusion runs its network 10 times per decision (DDIM-10) but decides only once every 8 control steps, so its cost per control step is lower than this.</figcaption>
    </figure>
  );
}
