"""The v1.1 addendum: X1, X2 and X3 from their hash-pinned results only (DECISIONS D-133).

``sources`` pins the three extension ``results.json`` files; ``numbers`` turns them into
``\\ax{key}`` macros; ``tables`` and ``figures`` render them; ``build`` writes or checks
``paper/addendum/generated/``; ``readme`` writes the README's ``extensions`` block from the
generated macros. Nothing here reads ``runs/`` or computes a statistic.
"""
