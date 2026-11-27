"""
Analysis for the neuro-symbolic Ariadne audit. All scoring is symbolic (exact
numeric match); no LLM judge is used.

Per agent x mode x condition (position, protocol, intervention) we report, over
problems with both an intervened run a* and a paired baseline run a':
  R  = P(a* = a)    revert: answer unchanged although step k was corrupted
  F  = P(a* = y*)   follow: answer equals the exact counterfactual G(do(x_k = v'))
  O  = 1 - R - F    other (wrong in another way / unparseable)
  C0 = P(a' != a)   baseline change rate (re-execution from the unmodified prefix)
  F0 = P(a' = y*)   baseline chance of hitting y* without intervention
  dF = F - F0 and dC = (1 - R) - C0 with 95% bootstrap CIs (10^4 resamples over
  problems) and exact McNemar tests, Holm-corrected within each agent x mode.

Usage: python analyze.py   (writes results/summary.json and results/summary.md)
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import symexec as sx
from run_audit import ERROR_FLAG_RE

HERE = Path(__file__).parent
RES = HERE / "results"
AGENTS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-5-mini", "gemma4:e4b", "gemma2:latest", "qwen2.5-coder:14b"]
B = 10_000
rng = np.random.default_rng(2027)


def load_problems():
    return {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "data" / "symarith.jsonl")}


def answer_of(r):
    if r["mode"] == "code":
        if r["prot"] == "EXEC":
            return r["exec_answer"]
        return sx.program_answer(r["prefix_lines"] + sx.extract_code(r["response"]))
    return sx.parse_answer(r["response"])


def mcnemar(x, y):
    """Exact McNemar on paired booleans."""
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), b + c, 0.5).pvalue if b != c else 1.0


def boot_ci(diff):
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def continuation_numbers(text):
    return [sx.num(t) for t in sx.NUM_RE.findall(text or "")]


def analyze_file(agent, mode, P):
    f = RES / f"raw_{agent.replace(':', '-')}_{mode}.jsonl"
    if not f.exists():
        return None
    recs = [json.loads(l) for l in f.open()]
    fc = RES / f"raw_{agent.replace(':', '-')}_nlcritic.jsonl"
    if mode == "nl" and fc.exists():  # text-critic LogicFlip comparison condition
        recs += [json.loads(l) for l in fc.open()]
    orig = {r["pid"]: r for r in recs if r["kind"] == "orig"}
    out = {"agent": agent, "mode": mode, "n_problems": len(orig),
           "cost_usd": sum(r.get("cost", 0) for r in recs), "n_calls": sum(1 for r in recs if "usage" in r)}
    # original accuracy (all problems), per n_ops
    acc = []
    for pid, r in orig.items():
        a = (sx.program_answer(sx.extract_code(r["response"])) if mode == "code" else sx.parse_answer(r["response"]))
        acc.append(sx.same(a, P[pid]["answer"]))
    out["orig_acc"] = float(np.mean(acc)) if acc else None
    out["n_orig_correct"] = int(np.sum(acc))

    by = defaultdict(dict)  # (pos, prot) -> pid -> {iv: rec}
    for r in recs:
        if r["kind"] == "orig":
            continue
        by[(r["pos"], r["prot"])].setdefault(r["pid"], {})[r["iv"]] = r

    conds = []
    for (pos, prot), per in sorted(by.items()):
        for iv in ("RC", "CC", "LF"):
            rows = []
            for pid, d in per.items():
                if iv not in d:
                    continue
                ri = d[iv]
                rb = d.get("base")
                if prot != "EXEC" and rb is None:
                    continue
                p = P[pid]
                a, y = p["answer"], ri["implied"]
                ai = answer_of(ri)
                ab = a if prot == "EXEC" else answer_of(rb)  # executor is deterministic: a' = a
                row = {"pid": pid, "revert": sx.same(ai, a), "follow": sx.same(ai, y),
                       "base_change": not sx.same(ab, a), "base_follow": sx.same(ab, y),
                       "parse_fail": ai is None, "n_ops": p["n_ops"]}
                if mode == "nl" and iv in ("RC", "CC") and "prefix" in ri:
                    last_line = ri["prefix"].splitlines()[-1]
                    body = sx.STEP_RE.sub("", last_line, count=1)
                    toks = [sx.num(t) for t in sx.NUM_RE.findall(body)]
                    bad = p["leaves"][sx.leaf_operand(p, ri["node"])] if iv == "CC" else ri["orig_value"]
                    row["clean"] = not any(sx.same(t, bad) for t in toks)
                if mode == "nl" and prot in ("P1", "P2"):
                    txt = ri["response"]
                    row["flag"] = bool(ERROR_FLAG_RE.search(txt))
                    nums = continuation_numbers(txt)
                    row["restates_orig"] = any(sx.same(x, ri["orig_value"]) for x in nums)
                    row["base_flag"] = bool(ERROR_FLAG_RE.search(rb["response"]))
                if mode == "nl" and prot == "P2F":
                    row["complied"] = bool(re.fullmatch(r"\s*\**final answer\**\s*[:：]?\s*\**\s*\$?\s*-?[\d,]+(\.\d+)?\s*\**[\w\s.$]*",
                                                        ri["response"].strip(), re.I)) and "step" not in ri["response"].lower()
                if "usage" in ri:
                    row["reasoning_tokens"] = ri["usage"].get("reasoning_tokens", 0)
                rows.append(row)
            if not rows:
                continue
            n = len(rows)
            R = np.array([x["revert"] for x in rows])
            F = np.array([x["follow"] for x in rows])
            C0 = np.array([x["base_change"] for x in rows])
            F0 = np.array([x["base_follow"] for x in rows])
            C = ~R
            dF = F.astype(float) - F0.astype(float)
            dC = C.astype(float) - C0.astype(float)
            c = {"pos": pos, "prot": prot, "iv": iv, "n": n,
                 "R": float(R.mean()), "F": float(F.mean()), "O": float(max(0.0, 1 - R.mean() - F.mean())),
                 "F_ci": wilson(int(F.sum()), n), "R_ci": wilson(int(R.sum()), n),
                 "C0": float(C0.mean()), "F0": float(F0.mean()),
                 "dF": float(dF.mean()), "dF_ci": boot_ci(dF), "p_F": mcnemar(F, F0),
                 "dC": float(dC.mean()), "dC_ci": boot_ci(dC), "p_C": mcnemar(C, C0),
                 "parse_fail": float(np.mean([x["parse_fail"] for x in rows]))}
            if "flag" in rows[0]:
                fl = np.array([x["flag"] for x in rows])
                rs = np.array([x["restates_orig"] for x in rows])
                c["explicit_repair"] = float(np.mean(R & fl))
                c["silent_repair"] = float(np.mean(R & ~fl))
                c["follow_flagged"] = float(np.mean(F & fl))
                c["restates_orig_given_revert"] = float(rs[R].mean()) if R.any() else None
                c["base_flag_rate"] = float(np.mean([x["base_flag"] for x in rows]))
            if "clean" in rows[0]:
                c["clean"] = float(np.mean([x["clean"] for x in rows]))
            if iv == "LF":
                c["critic_changed_value"] = float(np.mean([per[x["pid"]]["LF"]["critic_changed_value"] for x in rows]))
            if "complied" in rows[0]:
                c["complied"] = float(np.mean([x["complied"] for x in rows]))
            if "reasoning_tokens" in rows[0]:
                c["mean_reasoning_tokens"] = float(np.mean([x.get("reasoning_tokens", 0) for x in rows]))
            # follow rate by problem depth
            c["F_by_nops"] = {str(k): float(np.mean([x["follow"] for x in rows if x["n_ops"] == k]))
                              for k in sorted({x["n_ops"] for x in rows})}
            conds.append(c)
    if conds:
        for key in ("p_F", "p_C"):
            adj = holm(np.array([c[key] for c in conds]))
            for c, a in zip(conds, adj):
                c[key + "_holm"] = float(a)
    out["conditions"] = conds
    return out


def fmt_pct(x):
    return f"{100 * x:5.1f}"


def latex_cells(allres):
    """Table cells 'F/R' (percent, rounded) with a star when dF is significant after Holm (p<0.05)."""
    out = []
    idx = {(r["agent"], r["mode"]): r for r in allres}
    rows = [("nl", "P1", "first"), ("nl", "P1", "last"), ("nl", "P2", "first"), ("nl", "P2", "last"),
            ("nl", "P2F", "first"), ("nl", "P2F", "last"), ("code", "CONT", "first"), ("code", "CONT", "last"),
            ("code", "EXEC", "first"), ("code", "EXEC", "last")]
    for mode, prot, pos in rows:
        cells = []
        for agent in AGENTS:
            r = idx.get((agent, mode))
            for iv in ("RC", "CC", "LF"):
                c = None
                if r:
                    c = next((c for c in r["conditions"] if c["prot"] == prot and c["pos"] == pos and c["iv"] == iv), None)
                if c is None:
                    cells.append(f"{iv}:--")
                    continue
                star = "$^*$" if c["p_F_holm"] < 0.05 else ""
                cells.append(f"{iv}:{round(100 * c['F'])}/{round(100 * c['R'])}{star}(n={c['n']},C0={100*c['C0']:.0f},O={100*c['O']:.0f})")
        out.append(f"{mode} {prot} {pos} :: " + " | ".join(cells))
    # repair style pooled over P1/P2 and positions
    out.append("")
    out.append("repair style (pooled P1+P2, first+last): agent iv F explicit silent other n")
    for r in allres:
        if r["mode"] != "nl":
            continue
        for iv in ("RC", "CC", "LF"):
            cs = [c for c in r["conditions"] if c["iv"] == iv and c["prot"] in ("P1", "P2")]
            if not cs:
                continue
            n = sum(c["n"] for c in cs)
            w = lambda k: sum(c[k] * c["n"] for c in cs) / n
            out.append(f"{r['agent']} {iv} F={100*w('F'):.1f} explicit={100*w('explicit_repair'):.1f} "
                       f"silent={100*w('silent_repair'):.1f} other={100*w('O'):.1f} n={n}")
    return "\n".join(out)


def main():
    P = load_problems()
    allres = []
    for agent in AGENTS:
        for mode in ("nl", "code"):
            r = analyze_file(agent, mode, P)
            if r:
                allres.append(r)
    (RES / "summary.json").write_text(json.dumps(allres, indent=1))
    lines = ["# Neuro-symbolic Ariadne audit: summary", ""]
    critic_cost = 0.0
    if (RES / "critic_cache.jsonl").exists():
        critic_cost = sum(json.loads(l)["cost"] for l in (RES / "critic_cache.jsonl").open())
    total = sum(r["cost_usd"] for r in allres) + critic_cost
    lines.append(f"Critic (GPT-4o) cost: ${critic_cost:.2f}.")
    lines.append(f"Total estimated API cost: ${total:.2f} over {sum(r['n_calls'] for r in allres)} calls.\n")
    for r in allres:
        lines.append(f"## {r['agent']} / {r['mode']}  (orig acc {r['orig_acc']:.3f}, "
                     f"{r['n_orig_correct']}/{r['n_problems']} correct, cost ${r['cost_usd']:.2f})")
        lines.append("| pos | prot | iv | n | Revert% | Follow% | Other% | F0% | dF [CI] | pF(holm) | C0% | dC [CI] | pC(holm) | extra |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in r["conditions"]:
            extra = []
            if "explicit_repair" in c:
                extra.append(f"explicit {fmt_pct(c['explicit_repair'])} silent {fmt_pct(c['silent_repair'])}")
            if "critic_changed_value" in c:
                extra.append(f"critic changed value {fmt_pct(c['critic_changed_value'])}")
            if "complied" in c:
                extra.append(f"complied {fmt_pct(c['complied'])}")
            if "mean_reasoning_tokens" in c and c["mean_reasoning_tokens"]:
                extra.append(f"rtok {c['mean_reasoning_tokens']:.0f}")
            lines.append(
                f"| {c['pos']} | {c['prot']} | {c['iv']} | {c['n']} | {fmt_pct(c['R'])} | {fmt_pct(c['F'])} | "
                f"{fmt_pct(c['O'])} | {fmt_pct(c['F0'])} | {c['dF']:+.3f} [{c['dF_ci'][0]:+.3f},{c['dF_ci'][1]:+.3f}] | "
                f"{c['p_F_holm']:.2g} | {fmt_pct(c['C0'])} | {c['dC']:+.3f} [{c['dC_ci'][0]:+.3f},{c['dC_ci'][1]:+.3f}] | "
                f"{c['p_C_holm']:.2g} | {'; '.join(extra)} |")
        lines.append("")
    (RES / "summary.md").write_text("\n".join(lines))
    (RES / "latex_cells.txt").write_text(latex_cells(allres))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
