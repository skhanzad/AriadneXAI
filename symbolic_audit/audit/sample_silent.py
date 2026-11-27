"""Draw a stratified random sample of reverted NL runs that the error-acknowledgement regex
classifies as SILENT repairs (P1/P2, first+last step, RC+CC), for a manual false-negative audit.
Writes audit/silent_sample.jsonl; labels go in audit/silent_labels.jsonl."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import symexec as sx
from run_audit import ERROR_FLAG_RE

RES = Path(__file__).resolve().parent.parent / "results"
AGENTS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-5-mini"]
rng = random.Random(2027)
out = []
for agent in AGENTS:
    recs = [json.loads(l) for l in (RES / f"raw_{agent}_nl.jsonl").open()]
    orig = {r["pid"]: sx.parse_answer(r["response"]) for r in recs if r["kind"] == "orig"}
    for iv, k in (("RC", 7), ("CC", 6)) if agent in ("gpt-4o-mini", "gpt-4o") else (("RC", 6), ("CC", 7)):
        pool = [r for r in recs if r["kind"] == "int" and r["iv"] == iv and r["prot"] in ("P1", "P2")
                and sx.same(sx.parse_answer(r["response"]), orig[r["pid"]])
                and not ERROR_FLAG_RE.search(r["response"])]
        for r in rng.sample(pool, k):
            out.append({"key": r["key"], "agent": agent, "iv": iv, "prot": r["prot"], "pos": r["pos"],
                        "corrupted_step": r["prefix"].splitlines()[-1], "response": r["response"]})
with open(Path(__file__).parent / "silent_sample.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(len(out))
