"""LaTeX tables of the paper, generated from the frozen sources (DECISIONS D-094).

Each builder returns the body of one ``tabular`` (booktabs); the section files wrap it in a
``table`` with its caption. Values are copied at the frozen tables' rounding (``fmt``), and the
cross-check test compares them with the frozen ``tables.md`` files.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from roboforge.paper import figures, fmt, secondary, sensitivity
from roboforge.paper.numbers import L1_FAMILIES, LEARNED, TASKS
from roboforge.paper.sources import SOURCES, Sources

TASK_LABEL = {"lift": "Lift", "can": "Can"}
HYP_LABEL = {"H1": "H1", "H2": "H2", "H2c": "H2-c", "H3": "H3"}


def _tabular(spec: str, header: list[str], rows: list[list[str] | str]) -> str:
    """A booktabs tabular; a string row is inserted verbatim (rules, group headings)."""
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def _fam_cell(v: dict[str, Any]) -> str:
    return f"{fmt.pct(v['estimate'])} {fmt.ci(*v['ci95'])}"


def _seeds(v: dict[str, Any], whole: bool = True) -> str:
    f = fmt.pct0 if whole else fmt.pct
    return ", ".join(f(x) for x in v["per_seed"])


def heldout_success(src: Sources) -> str:
    """Level 1 held-out family success per task (core performance)."""
    r = src.json("l1_results")
    rows: list[list[str] | str] = []
    for f in L1_FAMILIES:
        cells = [fmt.FAMILY_LABEL[f]]
        for t in TASKS:
            td = r["tasks"][t]
            v = td["success"]["families"][f]
            k = sum(c["successes"] for c in td["per_checkpoint"] if c["family"] == f)
            n = sum(c["n_episodes"] for c in td["per_checkpoint"] if c["family"] == f)
            cells += [f"{k}/{n}", _fam_cell(v), _seeds(v)]
        rows.append(cells)
    hdr = ["Family"] + [f"{TASK_LABEL[t]} {c}" for t in TASKS for c in ("$k/n$", "success \\%",
                                                                         "per seed")]
    return _tabular("l" + "rrr" * len(TASKS), hdr, rows)


def heldout_differences(src: Sources) -> str:
    """Declared paired differences (success and censored episode length) per task."""
    r = src.json("l1_results")
    rows: list[list[str] | str] = ["\\multicolumn{5}{l}{\\emph{Success (percentage points)}} \\\\"]
    succ = {t: r["tasks"][t]["success"]["differences"] for t in TASKS}
    for i, d in enumerate(succ["lift"]):
        cells = [HYP_LABEL[d["hypothesis"]],
                 f"{fmt.FAMILY_LABEL[d['a']]} $-$ {fmt.FAMILY_LABEL[d['b']]}"]
        for t in TASKS:
            e = succ[t][i]
            assert (e["a"], e["b"]) == (d["a"], d["b"])
            cells.append(f"{fmt.spct(e['difference'])} {fmt.sci(*e['ci95'])}, "
                         f"{fmt.sep(e['separation'])}")
        rows.append(cells[:2] + [cells[2], cells[3]])
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Censored episode length (steps; failures count "
                f"{src.freeze_block()['rollouts']['horizon']})}}}} \\\\")
    lens = {t: r["tasks"][t]["censored_episode_length"]["differences"] for t in TASKS}
    for i, d in enumerate(lens["lift"]):
        cells = [HYP_LABEL[d["hypothesis"]],
                 f"{fmt.FAMILY_LABEL[d['a']]} $-$ {fmt.FAMILY_LABEL[d['b']]}"]
        for t in TASKS:
            e = lens[t][i]
            assert (e["a"], e["b"]) == (d["a"], d["b"])
            cells.append(f"{fmt.snum1(e['difference'])} {fmt.sci1(*e['ci95'])}, "
                         f"{fmt.sep(e['separation'])}")
        rows.append(cells)
    # Two value columns; the group headings span all of them.
    body = _tabular("llll", ["", "Difference $A - B$", "Lift", "Can"],
                    [r_ if isinstance(r_, str) else r_ for r_ in rows])
    return body.replace("\\multicolumn{5}", "\\multicolumn{4}")


def heldout_checkpoints(src: Sources) -> str:
    """Per-checkpoint held-out results (appendix): step, k/50, Wilson, first-success step."""
    r = src.json("l1_results")
    rows: list[list[str] | str] = []
    for t in TASKS:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{TASK_LABEL[t]}}}}} \\\\")
        for c in r["tasks"][t]["per_checkpoint"]:
            fs = c["first_success_step_mean"]
            rows.append([fmt.FAMILY_LABEL[c["family"]], "--" if c["family"] == "random_uniform"
                         else str(c["seed"]),
                         "--" if c["step"] is None else fmt.thousands(c["step"]),
                         f"{c['successes']}/{c['n_episodes']}", fmt.ci(*c["wilson95"]),
                         "--" if fs is None else fmt.num1(fs), str(c["n_errors"])])
    return _tabular("llrrrrr", ["Family", "Seed", "Step", "$k/n$", "Wilson 95\\,\\%",
                                "First success", "Errors"], rows)


def robustness_success(src: Sources) -> str:
    """H4: success on the fresh base starts and each OOD set (validation regime)."""
    a = src.json("l1_robustness")
    cols = [("lift", s) for s in a["tasks"]["lift"]["h4"]["sets"]] + \
           [("can", s) for s in a["tasks"]["can"]["h4"]["sets"]]
    rows: list[list[str] | str] = []
    for f in LEARNED:
        cells = [fmt.FAMILY_LABEL[f]]
        for t, s in cols:
            v = a["tasks"][t]["h4"]["bootstrap"]["families"][f"{f}:{s}"]
            cells.append(f"{fmt.pct(v['estimate'])} {fmt.ci(*v['ci95'])}")
        rows.append(cells)
    hdr = ["Family"] + [f"{TASK_LABEL[t]} {fmt.SET_LABEL[s]}" for t, s in cols]
    return _tabular("l" + "r" * len(cols), hdr, rows)


def robustness_contrasts(src: Sources) -> str:
    """H4: every reported drop, between-family difference and difference in drop."""
    a = src.json("l1_robustness")
    rows: list[list[str] | str] = []
    for t in TASKS:
        b = a["tasks"][t]["h4"]["bootstrap"]
        rows.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{TASK_LABEL[t]}}}}} \\\\")
        for d in b["differences"]:
            fa, sa = d["a"].split(":")
            fb, sb = d["b"].split(":")
            if fa == fb:
                name = f"drop of {fmt.FAMILY_LABEL[fa]}, base $-$ {fmt.SET_LABEL[sb]}"
            else:
                name = (f"{fmt.FAMILY_LABEL[fa]} $-$ {fmt.FAMILY_LABEL[fb]} on "
                        f"{fmt.SET_LABEL[sa]}")
            rows.append([name, f"{fmt.spct(d['difference'])} {fmt.sci(*d['ci95'])}",
                         fmt.sep(d["separation"])])
        for c in b["contrasts"]:
            pos = [k for k, w in c["weights"].items() if w == 1]
            fa = next(k.split(":")[0] for k in pos if k.endswith("robustness_base"))
            fb = next(k.split(":")[0] for k in pos if not k.endswith("robustness_base"))
            st = next(k.split(":")[1] for k in pos if not k.endswith("robustness_base"))
            name = (f"drop {fmt.FAMILY_LABEL[fa]} $-$ drop {fmt.FAMILY_LABEL[fb]} on "
                    f"{fmt.SET_LABEL[st]}")
            rows.append([name, f"{fmt.spct(c['estimate'])} {fmt.sci(*c['ci95'])}",
                         fmt.sep(c["separation"])])
    return _tabular("lrl", ["Contrast", "Estimate (pp) [95\\,\\% CI]", "Outcome"], rows)


ABL_LABEL = {
    "A1": "EMA (primary) vs trained weights",
    "A2": "DDIM-10 (primary) vs DDPM-100",
    "A3": "$T_o = 2$ (primary) vs $T_o = 1$",
}
ARM_LABEL = {"primary": "primary", "trained-weights": "trained weights", "ddpm100": "DDPM-100",
             "T_o=2": "$T_o = 2$", "T_o=1": "$T_o = 1$"}


def ablations(src: Sources) -> str:
    """A1-A3 on the fresh base starts (validation regime)."""
    a = src.json("l1_robustness")
    rows: list[list[str] | str] = []
    for ab in ("A1", "A2", "A3"):
        for t in TASKS:
            blk = a["tasks"][t].get(ab)
            if blk is None:
                continue
            fams = blk["bootstrap"]["families"]
            d = blk["bootstrap"]["differences"][0]
            arm_a, arm_b = d["a"], d["b"]
            rows.append([ab, TASK_LABEL[t],
                         f"{ARM_LABEL[arm_a]} {_fam_cell(fams[arm_a])}",
                         f"{ARM_LABEL[arm_b]} {_fam_cell(fams[arm_b])}",
                         f"{fmt.spct(d['difference'])} {fmt.sci(*d['ci95'])}",
                         fmt.sep(d["separation"])])
    return _tabular("llllll", ["", "Task", "Arm $A$ (success \\%)", "Arm $B$ (success \\%)",
                               "$A - B$ (pp)", "Outcome"], rows)


def selection_optimism(src: Sources) -> str:
    """Validation (selection) vs fresh base vs held-out success, descriptive."""
    a = src.json("l1_robustness")
    r = src.json("l1_results")
    rows: list[list[str] | str] = []
    for f in LEARNED:
        cells = [fmt.FAMILY_LABEL[f]]
        for t in TASKS:
            suite = src.json(f"suite_{t}")
            fam = next(x for x in suite["families"] if x["family"] == f)
            chosen = [x for x in fam["runs"] if x["trial"] == fam["chosen_trial"]]
            k = sum(x["selection"]["successes"] for x in chosen)
            n = sum(x["selection"]["n_episodes"] for x in chosen)
            fresh = a["tasks"][t]["h4"]["bootstrap"]["families"][f"{f}:robustness_base"]
            held = r["tasks"][t]["success"]["families"][f]
            cells += [fmt.pct(k / n), fmt.pct(fresh["estimate"]), fmt.pct(held["estimate"])]
        rows.append(cells)
    hdr = ["Family"] + [f"{TASK_LABEL[t]} {c}" for t in TASKS
                        for c in ("validation", "fresh base", "held-out")]
    return _tabular("l" + "rrr" * len(TASKS), hdr, rows)


def families(src: Sources) -> str:
    """Model families: network, horizons and exact parameter counts."""
    pc = src.json("param_counts")["primary"]
    cfg = src.yaml("cfg_diffusion")["model"]
    net = {
        "bc_mlp": f"MLP {len(pc['bc_mlp']['model']['hidden'])} $\\times$ "
                  f"{pc['bc_mlp']['model']['hidden'][0]}, ReLU",
        "bc_rnn": f"LSTM {pc['bc_rnn']['model']['layers']} $\\times$ "
                  f"{pc['bc_rnn']['model']['hidden']}",
        "bc_mlp_chunk": f"MLP {len(pc['bc_mlp_chunk']['model']['hidden'])} $\\times$ "
                        f"{pc['bc_mlp_chunk']['model']['hidden'][0]}, ReLU",
        "diffusion_unet": f"1D U-Net ({cfg['down_dims'][0]}, {cfg['down_dims'][1]}), FiLM; "
                          f"DDPM $K = {cfg['num_train_timesteps']}$, "
                          f"DDIM-{cfg['inference_steps']}, EMA",
    }
    rows: list[list[str] | str] = []
    for f in LEARNED:
        h = pc[f]["horizons"]
        hor = (f"context {h['obs']}, $T_p = {h['pred']}$, $T_a = {h['exec']}$" if f == "bc_rnn"
               else f"$T_o = {h['obs']}$, $T_p = {h['pred']}$, $T_a = {h['exec']}$")
        rows.append([fmt.FAMILY_LABEL[f], net[f], hor, fmt.thousands(pc[f]["params"])])
    return _tabular("llll", ["Family", "Network", "Horizons", "Parameters"], rows)


def _segments(k: int) -> str:
    return f"{k} segment" if k == 1 else f"{k} segments"


def cost_latency(src: Sources) -> str:
    """Hardware, training cost and inference latency of every study, with the KI-049 erratum."""
    r = src.json("l1_results")
    rgb = src.json("rgb_results")
    l2 = src.json("l2_results")
    rec = src.json("l2_seed0_launch")
    errata_sha = SOURCES["l2_errata"].sha256[:8]
    src.text("l2_errata")
    rows: list[list[str] | str] = [
        "\\multicolumn{5}{l}{\\emph{Level 1: training on local CPU (one torch thread); latency: "
        "dedicated benchmark, canonical container}} \\\\"]
    for t in TASKS:
        for f in LEARNED:
            cl = r["tasks"][t]["cost_latency"][f]
            cpu = ", ".join(fmt.num2(x) for x in cl["training_cpu_h_per_run"])
            rows.append([f"{TASK_LABEL[t]} {fmt.FAMILY_LABEL[f]}", f"CPU {cpu} CPU-h",
                         fmt.num2(cl["latency_ms_median"]), "--", f"\\${cl['cost_usd']:.2f}"])
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{RGB study (Lift): training on one NVIDIA L4; latency: "
                "median of episode medians during the held-out run}} \\\\")
    preempted = False
    for c in rgb["per_checkpoint"]:
        tr = c["training"]
        # More segments than launches: one launch restarted after a preemption, and the
        # preempted attempt has no recorded training time (audit C-6, KI-044).
        mark = ""
        if tr["segments"] > tr["launches"]:
            mark, preempted = "\\textsuperscript{b}", True
        rows.append([f"{fmt.FAMILY_LABEL[c['family']]} seed {c['seed']}",
                     f"L4 {fmt.num2(tr['train_wall_s'] / 3600)}{mark} h, "
                     f"{_segments(tr['segments'])}",
                     fmt.num1(c["latency_ms_median_of_episode_medians"]), "--",
                     f"\\${tr['billed_usd']:.2f}"])
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Level 2 (Lift): training on one NVIDIA L4; latency: "
                "median of episode medians during the held-out run}} \\\\")
    lat = {e["train_run_id"]: e["latency_ms_median_of_episode_medians"]
           for e in l2["heldout"]["evaluations"] if e["arm"] == "pc_diffusion"}
    for i, (run, v) in enumerate(sorted(l2["training"].items())):
        ms = f"{v['ms_per_step']:.2f}"
        wall = f"{v['train_wall_s'] / 3600:.2f}"
        if run == rec["run_id"]:
            steps = rec["end_step"] - rec["resumed_from_step"]
            fixed = 1000 * rec["segment"]["train_wall_s"] / steps
            ms = (f"{ms} $\\to$ \\textbf{{{fixed:.1f}}}\\textsuperscript{{a}}")
            wall = f"{wall}\\textsuperscript{{a}}"
        rows.append([f"{fmt.FAMILY_LABEL['pc_diffusion']} seed {i}",
                     f"L4 {wall} h, {_segments(v['segments'])}", fmt.num1(lat[run]), ms,
                     f"\\${v['billed_usd']:.2f}"])
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{p{0.95\\linewidth}}{\\footnotesize\\textsuperscript{a} Erratum "
                "(KI-049, \\texttt{ERRATA.md}, SHA-256 \\texttt{" + errata_sha + "}): the frozen "
                "analysis printed the left value. The corrected step time is the completing "
                "launch's training time over the steps it trained, from its committed launch "
                "record; the hours are that segment only and leave out the cut first segment "
                "(\\texttt{ERRATA.md} gives the total).} \\\\")
    if preempted:
        rows.append("\\multicolumn{5}{p{0.95\\linewidth}}{\\footnotesize\\textsuperscript{b} "
                    "One launch of this run was preempted and restarted; the preempted attempt "
                    "recorded no training time, so the hours leave it out, while the billed "
                    "amount includes it (KI-044; audit C-6).} \\\\")
    return _tabular("llrrr", ["Run", "Hardware and training time", "Latency ms", "ms/step",
                              "Billed"], rows)


def spend(src: Sources) -> str:
    """Billed spend per owner approval, from the cost ledger's billing events."""
    ev = src.jsonl("cost_ledger")
    launch = {e["launch_id"]: e for e in ev if e["event"] == "launch"}
    agg: dict[str, dict[str, Any]] = {}
    for e in ev:
        if e["event"] != "billing":
            continue
        lt = launch[e["launch_id"]]
        a = lt["approval_id"]
        g = agg.setdefault(a, {"gate": lt["decision"]["gate"], "cap": lt["decision"]["cap_usd"],
                               "functions": set(), "usd": 0.0, "n": 0})
        g["functions"].add(lt["function"])
        g["usd"] += e["billed_usd"]
        g["n"] += 1
    rows: list[list[str] | str] = []
    for a in sorted(agg):
        g = agg[a]
        rows.append([a, fmt.tex_escape(g["gate"]),
                     ", ".join(fmt.path(x) for x in sorted(g["functions"])),
                     str(g["n"]), f"\\${g['cap']:.2f}", f"\\${g['usd']:.4f}"])
    rows.append("\\midrule")
    total = sum(g["usd"] for g in agg.values())
    rows.append(["Total", "", "", str(sum(g["n"] for g in agg.values())), "", f"\\${total:.4f}"])
    return _tabular("lllrrr", ["Approval", "Gate", "Functions", "Billed launches", "Cap",
                               "Billed"], rows)


def taxonomy(src: Sources) -> str:
    """Failure taxonomy: held-out episodes of every study and the Level 2 placement sets."""
    r = src.json("l1_results")
    rgb = src.json("rgb_results")
    l2 = src.json("l2_results")
    cats = fmt.TAXONOMY_ORDER
    rows: list[list[str] | str] = []

    def row(name: str, d: dict[str, int]) -> list[str]:
        return [name] + [str(d[c]) for c in cats]

    for t in TASKS:
        rows.append(f"\\multicolumn{{{len(cats) + 1}}}{{l}}{{\\emph{{Level 1 held-out, "
                    f"{TASK_LABEL[t]}}}}} \\\\")
        for f in L1_FAMILIES:
            rows.append(row(fmt.FAMILY_LABEL[f], r["tasks"][t]["taxonomy_by_family"][f]))
    rows.append(f"\\multicolumn{{{len(cats) + 1}}}{{l}}{{\\emph{{RGB study held-out, Lift}}}} \\\\")
    for f in ("rgb_diffusion", "rgb_bc_chunk"):
        rows.append(row(fmt.FAMILY_LABEL[f], rgb["taxonomy_by_family"][f]))
    rows.append(f"\\multicolumn{{{len(cats) + 1}}}{{l}}{{\\emph{{Level 2, Lift}}}} \\\\")
    for k, d in l2["taxonomy_by_arm_and_set"].items():
        arm, st = k.split(":")
        label = "held-out" if st == "heldout" else fmt.SET_LABEL[st]
        rows.append(row(f"{fmt.FAMILY_LABEL[arm]}, {label}", d))
    hdr = ["Policy"] + [f"\\rotatebox{{90}}{{{fmt.TAXONOMY_LABEL[c]}}}" for c in cats]
    return _tabular("l" + "r" * len(cats), hdr, rows)


def taxonomy_robustness(src: Sources) -> str:
    """Failure taxonomy of the H4 evaluations (base and perturbation sets; validation regime)."""
    a = src.json("l1_robustness")
    cats = fmt.TAXONOMY_ORDER
    rows: list[list[str] | str] = []
    for t in TASKS:
        tax = a["tasks"][t]["h4"]["taxonomy"]
        sets = list(dict.fromkeys(k.split(":")[1] for k in tax))
        for st in sets:
            rows.append(f"\\multicolumn{{{len(cats) + 1}}}{{l}}{{\\emph{{{TASK_LABEL[t]}, "
                        f"{fmt.SET_LABEL[st]}}}}} \\\\")
            for f in LEARNED:
                d = tax[f"{f}:{st}"]
                rows.append([fmt.FAMILY_LABEL[f]] + [str(d[c]) for c in cats])
    hdr = ["Policy"] + [f"\\rotatebox{{90}}{{{fmt.TAXONOMY_LABEL[c]}}}" for c in cats]
    return _tabular("l" + "r" * len(cats), hdr, rows)


def rgb_table(src: Sources) -> str:
    """RGB study per checkpoint and family results (Lift held-out)."""
    r = src.json("rgb_results")
    rows: list[list[str] | str] = []
    for c in r["per_checkpoint"]:
        rows.append([fmt.FAMILY_LABEL[c["family"]], str(c["seed"]), fmt.thousands(c["step"]),
                     f"{c['successes']}/{c['n']}", fmt.ci(*c["wilson95"]),
                     fmt.num1(c["first_success_step"]["mean"])])
    rows.append("\\midrule")
    for f, v in r["success"]["families"].items():
        rows.append([fmt.FAMILY_LABEL[f], "all", "", "", _fam_cell(v), _seeds(v, whole=False)])
    d = r["success"]["differences"][0]
    rows.append(f"\\multicolumn{{6}}{{l}}{{{fmt.FAMILY_LABEL[d['a']]} $-$ "
                f"{fmt.FAMILY_LABEL[d['b']]}: {fmt.spct(d['difference'])} pp "
                f"{fmt.sci(*d['ci95'])}, {fmt.sep(d['separation'])}}} \\\\")
    return _tabular("lrrrll", ["Family", "Seed", "Step", "$k/n$",
                               "Wilson / bootstrap 95\\,\\%", "First success / per seed"], rows)


def level2_table(src: Sources) -> str:
    """Level 2: placement (H5) and held-out results."""
    r = src.json("l2_results")
    pb = r["placement"]["bootstrap"]
    rows: list[list[str] | str] = ["\\multicolumn{4}{l}{\\emph{Placement (validation regime; "
                                   "primary H5 on the cube shift)}} \\\\"]
    for arm in ("pc_diffusion", "rgb_diffusion"):
        for st in ("robustness_base", "ood_cube_shift"):
            v = pb["families"][f"{arm}:{st}"]
            rows.append([fmt.FAMILY_LABEL[arm], fmt.SET_LABEL[st], _fam_cell(v),
                         _seeds(v, whole=False)])
    rows.append("\\midrule")
    for d in pb["differences"]:
        fa, sa = d["a"].split(":")
        fb, sb = d["b"].split(":")
        if fa == fb:
            name = f"drop of {fmt.FAMILY_LABEL[fa]} (base $-$ cube shift)"
        else:
            name = f"point cloud $-$ RGB on {fmt.SET_LABEL[sa]}"
            if d["kind"].startswith("PRIMARY"):
                name = "\\textbf{H5:} " + name
        rows.append([name, "", f"{fmt.spct(d['difference'])} {fmt.sci(*d['ci95'])}",
                     fmt.sep(d["separation"])])
    c = pb["contrasts"][0]
    rows.append(["drop point cloud $-$ drop RGB", "",
                 f"{fmt.spct(c['estimate'])} {fmt.sci(*c['ci95'])}", fmt.sep(c["separation"])])
    rows.append("\\midrule")
    rows.append("\\multicolumn{4}{l}{\\emph{Held-out (secondary; RGB records reused, never "
                "rerun)}} \\\\")
    hs = r["heldout"]["success"]
    for arm, v in hs["families"].items():
        rows.append([fmt.FAMILY_LABEL[arm], "held-out", _fam_cell(v), _seeds(v, whole=False)])
    d = hs["differences"][0]
    rows.append(["point cloud $-$ RGB, success", "",
                 f"{fmt.spct(d['difference'])} {fmt.sci(*d['ci95'])}", fmt.sep(d["separation"])])
    dl = r["heldout"]["censored_episode_length"]["differences"][0]
    rows.append(["point cloud $-$ RGB, censored length (steps)", "",
                 f"{fmt.snum1(dl['difference'])} {fmt.sci1(*dl['ci95'])}",
                 fmt.sep(dl["separation"])])
    return _tabular("llll", ["Arm or contrast", "Set", "Success \\% or difference [95\\,\\% CI]",
                             "Per seed or outcome"], rows)


def sources_table(src: Sources) -> str:
    """Reproducibility appendix: every frozen source with its pinned SHA-256."""
    rows: list[list[str] | str] = []
    for s in SOURCES.values():
        if s.key.endswith("_tables"):
            continue
        src.raw(s.key)  # verified, and recorded as this table's source
        rows.append([fmt.path(s.path), f"\\texttt{{{s.sha256[:16]}}}", fmt.tex_escape(s.what)])
    # Renders copied by SHA-256 (located by hash, never by name) and the hash-chained summaries.
    for directory, sha, what in figures.RENDERS.values():
        path = figures.find_render(src.root, directory, sha)
        rows.append([fmt.path(path), f"\\texttt{{{sha[:16]}}}", fmt.tex_escape(what)])
    cited = [pc for t in TASKS for pc in src.json("l1_results")["tasks"][t]["per_checkpoint"]]
    rows.append([fmt.path("reports/session7/heldout/") + f" ({len(cited)} files)",
                 "per file", "Level 1 held-out evaluation summaries, each read only after its "
                 "SHA-256 equals the one recorded in the Level 1 results file (D-096)"])
    return _tabular("p{0.42\\linewidth}lp{0.36\\linewidth}",
                    ["Source", "SHA-256 (prefix)", "Content"], rows)


def sensitivity_main(src: Sources) -> str:
    """D-095 sensitivity: Level 1 held-out, RGB and Level 2 (not preregistered)."""
    rs = [r for r in sensitivity.rows(src) if not r.analysis.startswith(("H4", "A1", "A2", "A3"))]
    return sensitivity.table(rs)


def sensitivity_robustness(src: Sources) -> str:
    """D-095 sensitivity: H4 and A1-A3 (validation regime, not preregistered)."""
    return sensitivity.table(sensitivity.robustness_rows(src))


TABLES: dict[str, Callable[[Sources], str]] = {
    "heldout_success": heldout_success,
    "heldout_differences": heldout_differences,
    "heldout_checkpoints": heldout_checkpoints,
    "robustness_success": robustness_success,
    "robustness_contrasts": robustness_contrasts,
    "ablations": ablations,
    "selection_optimism": selection_optimism,
    "families": families,
    "cost_latency": cost_latency,
    "spend": spend,
    "taxonomy": taxonomy,
    "taxonomy_robustness": taxonomy_robustness,
    "rgb": rgb_table,
    "level2": level2_table,
    "sources": sources_table,
    "sensitivity_main": sensitivity_main,
    "sensitivity_robustness": sensitivity_robustness,
    "secondary": secondary.table,
}
