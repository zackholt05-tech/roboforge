"""Release checks over the repository and its full history (DECISIONS D-104).

Every check returns a list of problems (empty means pass), so `scripts/release_check.py` can
print one report and CI can fail on any problem. The checks read git and the working tree
only; none writes a file, reads `runs/` or steps a simulator.

- `check_history`: author, committer and tagger identity on every ref; no co-author trailer
  or AI byline in any commit or tag message (MASTER_SPEC §0.2).
- `check_secrets`: a regular-expression scan for common credential formats over
  `git log -p --all` and the tracked files. It is a pattern scan, not a full secret scanner.
- `check_private_paths`: absolute home paths only in the files disclosed by D-100.
- `check_tracked_content`: no dataset, checkpoint or mesh file, nothing over 50 MB, and the
  credential patterns of `.gitignore` really ignored.
- `check_docs`: no placeholder in the public documents, every relative Markdown link resolves,
  and `CITATION.cff` has the release fields with Zackary Holt as the only author.
- `external_urls` / `check_urls`: the network check of external links (manual, `--network`).
"""

from __future__ import annotations

import re
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml

from roboforge.paths import REPO_ROOT

AUTHOR_NAME = "Zackary Holt"
AUTHOR_EMAIL = "zackholt05@gmail.com"

# A co-author trailer or an AI byline in a commit or tag message.
_TRAILER = re.compile(r"(?im)^\s*co-authored-by\s*:")
_BYLINE = re.compile(
    r"(?i)(generated (with|by) .{0,20}(claude|chatgpt|copilot|gpt|ai\b)|"
    r"\bclaude code\b|noreply@anthropic\.com|🤖)"
)

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key_id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "github_fine_grained_pat": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b"),
    "slack_token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai_style_key": re.compile(r"\bsk-(proj-)?[A-Za-z0-9]{32,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{34,}\b"),
    "modal_token_value": re.compile(r"\b(ak|as)-[A-Za-z0-9]{22,}\b"),
    "modal_token_assignment": re.compile(
        r"MODAL_TOKEN_(ID|SECRET)\s*[=:]\s*['\"]?[A-Za-z0-9-]{12,}"
    ),
    "private_key_block": re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY"),
}

# An absolute path into a user's home directory (the username must start alphanumerically, so
# documentation forms such as /Users/<user>/ or /Users/.../ do not match).
HOME_PATH = r"/(Users|home)/[A-Za-z0-9][A-Za-z0-9._-]*/"
_HOME_PATH = re.compile(HOME_PATH)
# D-100: evidence files whose local paths are disclosed and kept byte for byte, plus two test
# fixtures that construct fake home paths on purpose.
DISCLOSED_PATH_PREFIXES: tuple[str, ...] = tuple(f"logs/session{n}/" for n in range(10))
DISCLOSED_PATH_FILES: frozenset[str] = frozenset(
    {
        "docs/KNOWN_ISSUES.md",
        "reports/feasibility/data_report_can.json",
        "reports/feasibility/data_report_lift.json",
        "reports/session5/modal_sim/report.json",
        "reports/session5/modal_sim/report_symlinked_mount.json",
        "tests/unit/test_snapshot.py",
    }
)

FORBIDDEN_SUFFIXES = (".hdf5", ".h5", ".pt", ".pth", ".ckpt", ".safetensors", ".stl", ".obj")
MAX_TRACKED_BYTES = 50 * 1024 * 1024
MUST_BE_IGNORED = (".env", ".env.local", "id.pem", "deploy.key", ".modal.toml", "secrets/x")

PUBLIC_DOCS = (
    "README.md",
    "CITATION.cff",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "docs/REPRODUCIBILITY.md",
    "docs/FINAL_RELEASE_AUDIT.md",
)
_PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b|<repo>|<your[-_ ]|(?i:lorem ipsum)")
_MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_URL = re.compile(r"https?://[^\s<>\"'`)\]}]+")


def git(*args: str, root: Path = REPO_ROOT) -> str:
    """Run git in the repository and return stdout (raises on failure)."""
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def tracked_files(root: Path = REPO_ROOT) -> list[str]:
    return [p for p in git("ls-files", "-z", root=root).split("\0") if p]


# ----------------------------------------------------------------------------- history


def message_problems(where: str, message: str) -> list[str]:
    """Trailer and byline problems of one commit or tag message."""
    out = []
    if _TRAILER.search(message):
        out.append(f"{where}: co-author trailer in the message")
    m = _BYLINE.search(message)
    if m:
        out.append(f"{where}: AI byline in the message ({m.group(0)!r})")
    return out


def check_history(root: Path = REPO_ROOT) -> list[str]:
    """Identity of every commit and annotated tag on every ref; trailers and bylines."""
    problems: list[str] = []
    fmt = "%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e"
    log = git("log", "--all", f"--format={fmt}", root=root)
    n = 0
    for rec in log.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, an, ae, cn, ce, body = rec.split("\x1f", 5)
        n += 1
        if (an, ae) != (AUTHOR_NAME, AUTHOR_EMAIL):
            problems.append(f"{sha[:10]}: author {an} <{ae}>")
        if (cn, ce) != (AUTHOR_NAME, AUTHOR_EMAIL):
            problems.append(f"{sha[:10]}: committer {cn} <{ce}>")
        problems += message_problems(sha[:10], body)
    if n == 0:
        problems.append("no commit found")
    tags = git(
        "for-each-ref",
        "refs/tags",
        "--format=%(objecttype)%1f%(refname:short)%1f%(taggername)%1f%(taggeremail)",
        root=root,
    )
    for line in tags.splitlines():
        kind, name, tn, te = line.split("\x1f")
        if kind != "tag":
            problems.append(f"tag {name}: lightweight tag (release tags are annotated)")
            continue
        if (tn, te.strip("<>")) != (AUTHOR_NAME, AUTHOR_EMAIL):
            problems.append(f"tag {name}: tagger {tn} {te}")
        problems += message_problems(f"tag {name}", git("tag", "-l", "--format=%(contents)", name,
                                                        root=root))
    return problems


def history_summary(root: Path = REPO_ROOT) -> dict[str, object]:
    """Counts for the release audit: commits on all refs, distinct identities, tags."""
    ids = git("log", "--all", "--format=%an <%ae> | %cn <%ce>", root=root).splitlines()
    return {
        "commits_all_refs": len(ids),
        "identities": sorted(set(ids)),
        "refs": git("for-each-ref", "--format=%(refname)", root=root).split(),
    }


# ----------------------------------------------------------------------------- secrets


def scan_text(lines: Iterable[str], where: str) -> list[str]:
    """Credential-pattern hits in some lines (the matched value is never printed)."""
    hits = []
    for i, line in enumerate(lines, 1):
        for name, pat in SECRET_PATTERNS.items():
            if pat.search(line):
                hits.append(f"{where}:{i}: possible {name}")
    return hits


def _history_lines(root: Path) -> Iterator[str]:
    proc = subprocess.Popen(
        ["git", "log", "-p", "--all", "--no-color", "--no-ext-diff", "--format=commit %H"],
        cwd=root,
        stdout=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    assert proc.stdout is not None
    yield from proc.stdout
    if proc.wait() != 0:
        raise RuntimeError("git log -p --all failed")


def check_secrets(root: Path = REPO_ROOT) -> list[str]:
    """Pattern hits in the full history (by commit) and in the tracked working tree."""
    problems: list[str] = []
    commit = "?"
    for line in _history_lines(root):
        if line.startswith("commit ") and len(line.strip()) == 47:
            commit = line.split()[1][:10]
            continue
        if line.startswith("+") or line.startswith("-"):
            for name, pat in SECRET_PATTERNS.items():
                if pat.search(line):
                    problems.append(f"history {commit}: possible {name}")
    for rel in tracked_files(root):
        p = root / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        problems += scan_text(text.splitlines(), rel)
    return sorted(set(problems))


# ----------------------------------------------------------------------------- paths


def path_disclosed(rel: str) -> bool:
    return rel in DISCLOSED_PATH_FILES or rel.startswith(DISCLOSED_PATH_PREFIXES)


def files_with_home_paths(root: Path = REPO_ROOT) -> list[str]:
    out = subprocess.run(
        ["git", "grep", "-I", "-l", "-E", HOME_PATH],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if out.returncode not in (0, 1):
        raise RuntimeError(out.stderr)
    return sorted(out.stdout.split())


def check_private_paths(root: Path = REPO_ROOT) -> list[str]:
    return [
        f"{rel}: absolute home path outside the D-100 disclosed set"
        for rel in files_with_home_paths(root)
        if not path_disclosed(rel)
    ]


# ----------------------------------------------------------------------------- content


def check_tracked_content(root: Path = REPO_ROOT) -> list[str]:
    problems = []
    for rel in tracked_files(root):
        if rel.lower().endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"{rel}: dataset, checkpoint or mesh file is tracked")
        p = root / rel
        if p.is_file() and p.stat().st_size > MAX_TRACKED_BYTES:
            problems.append(f"{rel}: tracked file over 50 MB")
    for probe in MUST_BE_IGNORED:
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", probe], cwd=root)
        if r.returncode != 0:
            problems.append(f"{probe}: not ignored by .gitignore")
    return problems


# ----------------------------------------------------------------------------- docs


def placeholder_problems(rel: str, text: str) -> list[str]:
    return [
        f"{rel}:{i}: placeholder {m.group(0)!r}"
        for i, line in enumerate(text.splitlines(), 1)
        for m in [_PLACEHOLDER.search(line)]
        if m
    ]


def relative_links(md_text: str) -> list[str]:
    """Relative link targets of a Markdown text (no URL, mail or in-page anchor)."""
    out = []
    for target in _MD_LINK.findall(md_text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue
        out.append(target.split("#", 1)[0])
    return [t for t in out if t]


def check_links(root: Path = REPO_ROOT) -> list[str]:
    problems = []
    for rel in tracked_files(root):
        if not rel.endswith(".md"):
            continue
        p = root / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        # Fenced code blocks are examples, not links.
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for target in relative_links(text):
            if not (p.parent / target).exists():
                problems.append(f"{rel}: broken relative link {target!r}")
    return problems


def check_citation(root: Path = REPO_ROOT) -> list[str]:
    cff = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    problems = []
    for key in ("cff-version", "title", "version", "date-released", "license", "authors"):
        if not cff.get(key):
            problems.append(f"CITATION.cff: missing {key}")
    authors = cff.get("authors") or []
    names = [(a.get("given-names"), a.get("family-names")) for a in authors]
    if names != [("Zackary", "Holt")]:
        problems.append(f"CITATION.cff: authors are {names}, expected only Zackary Holt")
    if cff.get("license") != "Apache-2.0":
        problems.append("CITATION.cff: license is not Apache-2.0 (D-013)")
    if "no results yet" in str(cff.get("abstract", "")).lower():
        problems.append("CITATION.cff: abstract still says there are no results")
    return problems


def check_docs(root: Path = REPO_ROOT) -> list[str]:
    problems = []
    for rel in PUBLIC_DOCS:
        p = root / rel
        if not p.exists():
            problems.append(f"{rel}: missing")
            continue
        problems += placeholder_problems(rel, p.read_text(encoding="utf-8"))
    for p in sorted((root / "paper" / "sections").glob("*.tex")):
        problems += placeholder_problems(p.relative_to(root).as_posix(), p.read_text())
    return problems + check_links(root) + check_citation(root)


# ----------------------------------------------------------------------------- network

URL_DOCS = PUBLIC_DOCS + (
    "docs/DATA_PROVENANCE.md",
    "docs/SIMULATOR_CONTRACT.md",
    "MASTER_SPEC.md",
)


def external_urls(root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """``{url: [documents citing it]}`` for the public documents."""
    out: dict[str, list[str]] = {}
    for rel in URL_DOCS:
        p = root / rel
        if not p.exists():
            continue
        for url in _URL.findall(p.read_text(encoding="utf-8")):
            url = url.rstrip(".,;:")
            out.setdefault(url, []).append(rel)
    return dict(sorted(out.items()))


def fetch_status(url: str, timeout: float = 20.0) -> int | str:
    """HTTP status of a HEAD request, falling back to GET; a string on a network error."""
    headers = {"User-Agent": "roboforge-release-check/1.0"}
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return int(r.status)
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in (403, 404, 405, 429, 501):
                continue
            return int(exc.code)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if method == "HEAD":
                continue
            return f"error: {exc}"
    return "error: no response"
