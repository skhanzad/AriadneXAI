"""
Repair style of the visible-thinking agent (run_thinking.py): for each reverted run,
is the repair acknowledged in the continuation's THINKING, in its ANSWER, both, or neither?
Also reports the regex false-positive rate on paired baseline runs.
Writes results/thinking_summary.md.
"""
import json
from collections import defaultdict
from pathlib import Path

import symexec as sx
from run_audit import ERROR_FLAG_RE

RES = Path(__file__).parent / "results"


def main(agent="qwen3:8b"):
    P = {json.loads(l)["id"]: json.loads(l) for l in open(Path(__file__).parent / "data" / "symarith.jsonl")}
    recs = [json.loads(l) for l in (RES / f"raw_{agent.replace(':', '-')}_nl.jsonl").open()]
    by = defaultdict(dict)
    for r in recs:
        if r["kind"] != "orig":
            by[(r["pos"], r["prot"], r["pid"])][r["iv"]] = r
    rows = defaultdict(list)
    for (pos, prot, pid), d in by.items():
        if "base" not in d:
            continue
        for iv in ("RC", "CC"):
            if iv not in d:
                continue
            ri, rb, a = d[iv], d["base"], P[pid]["answer"]
            ans = sx.parse_answer(ri["response"])
            rows[(prot, pos, iv)].append({
                "follow": sx.same(ans, ri["implied"]), "revert": sx.same(ans, a),
                "think_flag": bool(ERROR_FLAG_RE.search(ri.get("reasoning") or "")),
                "ans_flag": bool(ERROR_FLAG_RE.search(ri["response"] or "")),
                "base_think_flag": bool(ERROR_FLAG_RE.search(rb.get("reasoning") or "")),
                "think_words": len((ri.get("reasoning") or "").split())})
    out = ["# Visible-thinking repair style: " + agent, "",
           "| prot | pos | iv | n | F% | R% | revert: flagged in thinking% | revert: flagged in answer% | "
           "revert: silent in answer but flagged in thinking% | baseline thinking flag% | mean thinking words |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    pct = lambda x, n: f"{100 * x / n:.1f}" if n else "--"
    for key in sorted(rows):
        rs = rows[key]
        n = len(rs)
        rev = [x for x in rs if x["revert"]]
        nr = len(rev)
        out.append(f"| {' | '.join(key)} | {n} | {pct(sum(x['follow'] for x in rs), n)} | {pct(nr, n)} | "
                   f"{pct(sum(x['think_flag'] for x in rev), nr)} | {pct(sum(x['ans_flag'] for x in rev), nr)} | "
                   f"{pct(sum(x['think_flag'] and not x['ans_flag'] for x in rev), nr)} | "
                   f"{pct(sum(x['base_think_flag'] for x in rs), n)} | {sum(x['think_words'] for x in rs) / n:.0f} |")
    (RES / "thinking_summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
