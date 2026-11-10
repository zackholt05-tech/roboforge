"""Session 12 release tooling: README content from frozen files, release scans, Pages bundle.

Nothing here reads `runs/`, steps a simulator or changes a frozen file (DECISIONS D-097 to
D-104). Every number this package writes comes from `paper/generated/numbers.tex`, which
`make paper-check` proves equal to a fresh build from the hash-pinned frozen sources (D-094).
"""
