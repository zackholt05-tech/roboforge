"""Tests of the Session 12 release tooling (DECISIONS D-102 to D-104)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from roboforge.release import pages, preview, scan
from roboforge.release.macros import Macros, load_macros, parse_macros, to_text
from roboforge.release.readme import BEGIN, END, findings_block, replace_block

LINE = "\\expandafter\\def\\csname rf@{k}\\endcsname{{{v}}}"


def test_parse_and_convert_macros() -> None:
    text = "\n".join(
        [
            "% comment",
            LINE.format(k="a.d", v="\\ensuremath{-}41.3"),
            LINE.format(k="a.sep", v="separated (\\ensuremath{+})"),
            LINE.format(k="n", v="1,077,255"),
        ]
    )
    raw = parse_macros(text)
    assert to_text(raw["a.d"]) == "−41.3"
    assert to_text(raw["a.sep"]) == "separated (+)"
    assert to_text(raw["n"]) == "1,077,255"


def test_duplicate_macro_and_unknown_latex_are_errors() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_macros("\n".join([LINE.format(k="x", v="1"), LINE.format(k="x", v="2")]))
    with pytest.raises(ValueError, match="unconverted"):
        to_text("\\ensuremath{3\\times 10^{-4}}")


def test_unknown_key_raises() -> None:
    with pytest.raises(KeyError, match="undefined result key"):
        Macros({})["l1.lift.diff.succ"]


def test_findings_block_uses_the_frozen_values() -> None:
    values = load_macros()
    block = findings_block(Macros(values))
    assert block.startswith(BEGIN) and block.endswith(END)
    for key in ("l1.can.diff.succ", "l1.can.bcmlp.succ", "l1.can.diff-bcmlp.d", "l2.h5.d",
                "rgb.diff-bc.d", "spend.total", "sens.l1.can.diff-bcmlp.lo"):
        assert values[key] in block, key


def test_findings_block_refuses_a_verdict_the_prose_contradicts() -> None:
    values = dict(load_macros())
    values["l1.can.diff-chunk.sep"] = "separated (+)"
    with pytest.raises(ValueError, match="assumes"):
        findings_block(Macros(values))


def test_replace_block_needs_exactly_one_pair() -> None:
    assert replace_block(f"a\n{BEGIN}\nold\n{END}\nb", "NEW") == "a\nNEW\nb"
    with pytest.raises(ValueError):
        replace_block("no markers", "NEW")
    with pytest.raises(ValueError):
        replace_block(f"{BEGIN}{END}{BEGIN}{END}", "NEW")


def test_message_problems() -> None:
    assert scan.message_problems("c", "Session 12: release checks\n\n- docs") == []
    assert scan.message_problems("c", "fix\n\nCo-authored-by: Someone <a@b.c>")
    assert scan.message_problems("c", "fix\n\n🤖 Generated with something")
    assert scan.message_problems("c", "x <noreply@anthropic.com>")


def test_secret_patterns_flag_constructed_values_only() -> None:
    fake_aws = "AKIA" + "Q" * 16
    fake_gh = "gh" + "p_" + "a1" * 20
    hits = scan.scan_text([f"key = {fake_aws}", "nothing here", f"t={fake_gh}"], "f")
    assert hits == ["f:1: possible aws_access_key_id", "f:3: possible github_token"]
    assert scan.scan_text(["AKIA[0-9A-Z]{16}", "sk-short"], "f") == []


def test_private_path_rules() -> None:
    assert scan.path_disclosed("logs/session3/x.log")
    assert not scan.path_disclosed("logs/session12/x.log")
    assert not scan.path_disclosed("README.md")
    assert scan.path_disclosed("docs/KNOWN_ISSUES.md")


def test_placeholders_and_relative_links() -> None:
    assert scan.placeholder_problems("r", "git clone <repo> x\nfine\nTBD") == [
        "r:1: placeholder '<repo>'",
        "r:3: placeholder 'TBD'",
    ]
    md = "[a](docs/X.md#s) [b](https://x.org) [c](#top) ![d](media/p.gif) [e](mailto:a@b)"
    assert scan.relative_links(md) == ["docs/X.md", "media/p.gif"]


def _git(repo: Path, *args: str, name: str = "Zackary Holt",
         email: str = "zackholt05@gmail.com") -> None:
    env = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name,
           "GIT_COMMITTER_EMAIL": email, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
           "HOME": str(repo)}
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)


def test_check_history_on_a_scratch_repository(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a").write_text("1")
    _git(tmp_path, "add", "a")
    _git(tmp_path, "commit", "-q", "-m", "first")
    assert scan.check_history(tmp_path) == []
    (tmp_path / "a").write_text("2")
    _git(tmp_path, "commit", "-q", "-am", "second\n\nCo-authored-by: Bot <b@x>",
         name="Someone", email="s@x")
    problems = scan.check_history(tmp_path)
    assert any("author Someone" in p for p in problems)
    assert any("co-author trailer" in p for p in problems)
    _git(tmp_path, "tag", "light")
    assert any("lightweight" in p for p in scan.check_history(tmp_path))


def test_runtime_npm_packages_skip_dev(tmp_path: Path) -> None:
    lock = {"packages": {"": {}, "node_modules/a": {"version": "1.0.0"},
                         "node_modules/b": {"version": "2.0.0", "dev": True}}}
    (tmp_path / "package-lock.json").write_text(json.dumps(lock))
    assert [(n, v) for n, v, _ in pages.runtime_npm_packages(tmp_path)] == [("a", "1.0.0")]


def test_preview_pair_is_the_rule_3a_can_pair() -> None:
    clips = preview.select_pair()
    assert [c["family"] for c in clips] == ["diffusion_unet", "bc_mlp"]
    assert clips[0]["state_id"] == clips[1]["state_id"]
    assert all(c["task"] == "can" and c["seed"] == 0 and c["outcome"]["success"] for c in clips)
