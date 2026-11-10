"""Check that every bibliography entry's DOI, arXiv identifier or URL resolves (D-094).

    uv run --frozen python scripts/verify_bibliography.py   # needs the network

For each entry of ``paper/references.bib``:
- a DOI is resolved through https://doi.org with CSL-JSON content negotiation;
- an arXiv identifier through the arXiv export API;
- a software entry without either through its URL (HTTP 200 after redirects).

The resolved title (and first author, when the record has authors) must match the entry.
Results go to ``paper/bibliography_check.json``; the script exits 1 if any entry fails. It
uses the standard library only, retries transient network errors (KNOWN_ISSUES KI-047), and
is never part of ``make test``.

XML: the arXiv response is parsed with ``xml.etree.ElementTree``, not ``defusedxml``, because
D-093 adds no dependency. The input comes only from export.arxiv.org over HTTPS for the
identifiers in the bibliography, Python's bundled expat (2.6 or later) refuses external entity
expansion and limits entity amplification, and the script runs by hand on a developer host.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIB = REPO / "paper" / "references.bib"
OUT = REPO / "paper" / "bibliography_check.json"
UA = "RoboForge-bibcheck/1.0 (python-urllib)"


def parse_bib(text: str) -> list[dict[str, str]]:
    """Entries of a BibTeX file as dicts with ``_key`` and ``_type`` (brace-aware)."""
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),", text):
        i, f = m.end(), {"_type": m.group(1).lower(), "_key": m.group(2).strip()}
        while True:
            fm = re.compile(r"\s*([A-Za-z]+)\s*=\s*\{").match(text, i)
            if not fm:
                break
            j, depth = fm.end() - 1, 0
            while True:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            f[fm.group(1).lower()] = " ".join(text[fm.end():j].split())
            i = j + 1
            cm = re.compile(r"\s*,").match(text, i)
            if cm:
                i = cm.end()
        entries.append(f)
    return entries


def norm(s: str) -> str:
    """Lower-case ASCII letters and digits only (for title and name comparison)."""
    s = unicodedata.normalize("NFKD", s.replace("{", "").replace("}", ""))
    return re.sub(r"[^a-z0-9]", "", s.encode("ascii", "ignore").decode().lower())


def first_family(author_field: str) -> str:
    """Family name of the first author of a BibTeX author field ("Family, Given and ...")."""
    first = author_field.split(" and ")[0]
    return first.split(",")[0] if "," in first else first.split(" ")[-1]


def fetch(url: str, accept: str) -> tuple[int, str, bytes]:
    """GET with retries on transient errors; returns (status, final URL, body)."""
    last: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.geturl(), resp.read()
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return e.code, url, b""
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def check_doi(doi: str) -> dict[str, object]:
    status, final, body = fetch("https://doi.org/" + urllib.parse.quote(doi),
                                "application/vnd.citationstyles.csl+json")
    rec: dict[str, object] = {"identifier": f"doi:{doi}", "http_status": status, "resolved": final}
    if status == 200:
        d = json.loads(body)
        title = d.get("title")
        rec["title"] = title[0] if isinstance(title, list) else title
        rec["authors"] = [a.get("family", "") for a in d.get("author", [])]
    return rec


def check_arxiv(eid: str) -> dict[str, object]:
    status, final, body = fetch("https://export.arxiv.org/api/query?id_list=" + eid,
                                "application/atom+xml")
    rec: dict[str, object] = {"identifier": f"arXiv:{eid}", "http_status": status,
                              "resolved": f"https://arxiv.org/abs/{eid}"}
    if status == 200:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = ET.fromstring(body).find("a:entry", ns)
        if entry is not None and entry.find("a:title", ns) is not None:
            rec["title"] = " ".join(entry.find("a:title", ns).text.split())
            rec["authors"] = [n.text.split(" ")[-1] for n in entry.findall("a:author/a:name", ns)]
    return rec


def check_url(url: str) -> dict[str, object]:
    status, final, _ = fetch(url, "text/html")
    return {"identifier": f"url:{url}", "http_status": status, "resolved": final}


def verify(entry: dict[str, str]) -> dict[str, object]:
    """Check one entry: every identifier it carries must resolve and match."""
    checks = []
    if "doi" in entry:
        checks.append(check_doi(entry["doi"]))
    if "eprint" in entry:
        checks.append(check_arxiv(entry["eprint"]))
    if not checks and "url" in entry:
        checks.append(check_url(entry["url"]))
    title, fam = norm(entry.get("title", "")), norm(first_family(entry.get("author", "")))
    ok = bool(checks)
    for c in checks:
        c["ok_status"] = c["http_status"] == 200
        if "title" in c:
            c["title_match"] = norm(str(c["title"])) == title
            c["first_author_match"] = fam in {norm(a) for a in c.get("authors", [])}
        ok = ok and c["ok_status"] and c.get("title_match", True)
    # The first author must match at least one authoritative record (Crossref misorders a few
    # names, e.g. PointNet's "Charles, R. Qi"; the arXiv record carries the authors' spelling).
    author_checks = [c["first_author_match"] for c in checks if "first_author_match" in c]
    if author_checks:
        ok = ok and any(author_checks)
    return {"key": entry["_key"], "ok": ok, "checks": checks}


def main() -> int:
    """Verify every entry and write the report."""
    entries = parse_bib(BIB.read_text())
    results = [verify(e) for e in entries]
    report = {
        "schema": "roboforge.bibliography_check/1",
        "bib": "paper/references.bib",
        "checked_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_entries": len(results),
        "all_ok": all(r["ok"] for r in results),
        "entries": results,
    }
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for r in results:
        print(("ok  " if r["ok"] else "FAIL"), r["key"])
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
