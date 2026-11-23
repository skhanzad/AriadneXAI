"""
Drop NL RC/CC records whose stored corrupted prefix differs from what the current
symexec.corrupt_nl produces (used once, after fixing RC/CC to replace every mention
of the corrupted quantity in the step). Re-running run_audit.py afterwards
regenerates exactly the dropped calls.

Usage: python refresh_cc.py --agent gpt-4o [--dry-run]
"""
import argparse
import json
from pathlib import Path

import symexec as sx

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    P = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "data" / "symarith.jsonl")}
    f = HERE / "results" / f"raw_{args.agent}_nl.jsonl"
    recs = [json.loads(l) for l in f.open()]
    orig = {r["pid"]: r for r in recs if r["kind"] == "orig"}
    keep, dropped = [], 0
    for r in recs:
        if r["kind"] == "int" and r["iv"] in ("CC", "RC"):
            p = P[r["pid"]]
            steps, comps, node_of = sx.align_nl(p, orig[r["pid"]]["response"])
            k = r["step_index"]
            c = sx.corrupt_nl(p, steps[k], comps[k], node_of[k], r["iv"])
            new_prefix = None if c is None else "\n".join([s["line"] for s in steps[:k]] + [c[0]])
            if new_prefix != r["prefix"]:
                dropped += 1
                continue
        keep.append(r)
    print(f"{args.agent}: {dropped} RC/CC records to regenerate (of {sum(1 for r in recs if r.get('iv') in ('CC', 'RC'))})")
    if not args.dry_run:
        with f.open("w") as out:
            for r in keep:
                out.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
