"""Formatting of frozen values for LaTeX, at the frozen tables' rounding (D-094).

The frozen analysis scripts print fractions as ``f"{100 * x:.1f}"`` and differences with an
explicit sign (``:+.1f``). These helpers use the same Python formatting, so every generated
value matches the frozen ``tables.md`` digit for digit; only the minus sign is typeset as a
proper minus.
"""

from __future__ import annotations

FAMILY_LABEL = {
    "bc_mlp": "BC-MLP",
    "bc_rnn": "BC-RNN",
    "bc_mlp_chunk": "BC-MLP-chunk",
    "diffusion_unet": "Diffusion U-Net",
    "random_uniform": "Random (uniform)",
    "rgb_bc_chunk": "Visual BC-chunk",
    "rgb_diffusion": "Visual diffusion",
    "pc_diffusion": "Point-cloud diffusion",
}

FAMILY_KEY = {
    "bc_mlp": "bcmlp",
    "bc_rnn": "bcrnn",
    "bc_mlp_chunk": "chunk",
    "diffusion_unet": "diff",
    "random_uniform": "random",
    "rgb_bc_chunk": "rgbbc",
    "rgb_diffusion": "rgbdiff",
    "pc_diffusion": "pcdiff",
}

SET_LABEL = {
    "robustness_base": "base",
    "ood_cube_shift": "cube shift",
    "ood_arm_offset": "arm offset",
}

SET_KEY = {"robustness_base": "base", "ood_cube_shift": "cube", "ood_arm_offset": "arm"}

TAXONOMY_ORDER = [
    "success", "engineering_error", "gripper_never_closed", "no_grasp_drift", "no_grasp_near",
    "timeout_holding", "in_goal_bin_not_success", "dropped",
]

TAXONOMY_LABEL = {
    "success": "success",
    "engineering_error": "engineering error",
    "gripper_never_closed": "gripper never closed",
    "no_grasp_drift": "no grasp, drifted",
    "no_grasp_near": "no grasp, near",
    "timeout_holding": "timeout holding",
    "in_goal_bin_not_success": "in goal bin, not success",
    "dropped": "dropped",
}


def minus(s: str) -> str:
    """Typeset a leading ASCII minus of a formatted number as a math minus."""
    return s.replace("-", "\\ensuremath{-}", 1) if s.startswith("-") else s


def signed(s: str) -> str:
    """Typeset an explicitly signed number (``+x`` or ``-x``) with math signs."""
    if s.startswith("+"):
        return "\\ensuremath{+}" + s[1:]
    return minus(s)


def pct(x: float) -> str:
    """A fraction as a percentage with one decimal, like the frozen tables."""
    return minus(f"{100 * x:.1f}")


def spct(x: float) -> str:
    """A signed difference of fractions in percentage points, like the frozen tables."""
    return signed(f"{100 * x:+.1f}")


def pct0(x: float) -> str:
    """A per-seed fraction as a whole percentage, like the frozen Level 1 tables."""
    return f"{100 * x:.0f}"


def num1(x: float) -> str:
    """A plain number with one decimal."""
    return minus(f"{x:.1f}")


def snum1(x: float) -> str:
    """A signed plain number with one decimal."""
    return signed(f"{x:+.1f}")


def num2(x: float) -> str:
    """A plain number with two decimals."""
    return minus(f"{x:.2f}")


def thousands(n: int) -> str:
    """An integer with thousands separators (``1,077,255``)."""
    return f"{n:,}"


def ci(lo: float, hi: float) -> str:
    """A percentage interval, ``[83.3, 96.7]``."""
    return f"[{pct(lo)}, {pct(hi)}]"


def sci(lo: float, hi: float) -> str:
    """A signed percentage-point interval, ``[−0.7, +14.7]``."""
    return f"[{spct(lo)}, {spct(hi)}]"


def ci1(lo: float, hi: float) -> str:
    """A plain one-decimal interval (steps)."""
    return f"[{num1(lo)}, {num1(hi)}]"


def sci1(lo: float, hi: float) -> str:
    """A signed plain one-decimal interval (steps)."""
    return f"[{snum1(lo)}, {snum1(hi)}]"


def sep(label: str) -> str:
    """The frozen separation label with its sign typeset (``separated (+)``)."""
    return label.replace("(-)", "(\\ensuremath{-})").replace("(+)", "(\\ensuremath{+})")


def tex_escape(s: str) -> str:
    """Escape LaTeX special characters in plain text (identifiers, paths)."""
    out = []
    for ch in s:
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        elif ch == "\\":
            out.append("\\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)


def path(s: str) -> str:
    """A repository path or identifier in typewriter type, breakable at slashes and hyphens."""
    t = tex_escape(s).replace("/", "/\\allowbreak{}").replace("-", "-\\allowbreak{}")
    return "\\texttt{" + t + "}"
