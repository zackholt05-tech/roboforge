"""Diagnose a failed bitwise robustness check: regenerate one task's robustness suites in
memory (canonical container) and print every manifest field that differs from the committed
manifest. Read-only; writes nothing. Usage: python <this> <task>"""
import sys
from roboforge.sim import eval_states as es, robustness as rb

def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from walk(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, f"{path}[{i}]")
    elif a != b:
        yield path, a, b

task = sys.argv[1]
arts = rb.generate_task(task)
for name in rb.sets_for(task):
    diffs = list(walk(es.read_manifest(task, name), arts[name].manifest))
    print(f"{task}/{name}: {len(diffs)} differing fields")
    for p, a, b in diffs[:40]:
        print(f"  {p}: committed={a!r} regenerated={b!r}")
