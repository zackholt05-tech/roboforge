"""Release checks (DECISIONS D-104): history identity, secrets, private paths, tracked content,
public documents, the generated README block and the README preview.

    uv run --frozen python scripts/release_check.py                 # offline (make release-check)
    uv run --frozen python scripts/release_check.py --network --out reports/session12/x.json

`--network` adds an HTTP check of every external URL in the public documents. It is never
part of `make test` or CI. Exit status 1 when any offline check finds a problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from roboforge.paths import REPO_ROOT
from roboforge.provenance import environment_snapshot
from roboforge.release import preview, scan
from roboforge.release.readme import check_readme


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--network", action="store_true", help="also check external URLs")
    p.add_argument("--out", type=Path, help="write the JSON report here")
    a = p.parse_args(argv)

    checks = {
        "history": scan.check_history,
        "secrets": scan.check_secrets,
        "private_paths": scan.check_private_paths,
        "tracked_content": scan.check_tracked_content,
        "docs": scan.check_docs,
        "readme_findings": check_readme,
        "readme_preview": preview.check,
    }
    report: dict[str, object] = {"schema": "roboforge.release_check/1", "decision": "D-104"}
    results: dict[str, list[str]] = {}
    for name, fn in checks.items():
        results[name] = fn()
        status = "PASS" if not results[name] else f"FAIL ({len(results[name])})"
        print(f"{name:16s} {status}")
        for line in results[name][:50]:
            print(f"  - {line}")
    report["checks"] = results
    report["history"] = scan.history_summary()
    report["disclosed_path_files"] = [
        f for f in scan.files_with_home_paths() if scan.path_disclosed(f)
    ]
    print(f"disclosed local-path files (D-100): {len(report['disclosed_path_files'])}")

    if a.network:
        urls = scan.external_urls()
        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = dict(zip(urls, pool.map(scan.fetch_status, urls), strict=True))
        bad = {u: s for u, s in statuses.items() if not (isinstance(s, int) and s < 400)}
        report["urls"] = {u: {"status": statuses[u], "cited_in": urls[u]} for u in urls}
        print(f"external URLs    {len(urls) - len(bad)}/{len(urls)} answered with HTTP < 400")
        for u, s in bad.items():
            print(f"  - {s}  {u}  ({', '.join(sorted(set(urls[u])))})")

    report["environment"] = environment_snapshot()
    if a.out:
        out = a.out if a.out.is_absolute() else REPO_ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
        print(f"wrote {out}")
    failed = any(results.values())
    print("release-check: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
