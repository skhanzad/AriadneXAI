"""
Comparison condition: the original Ariadne text-critic intervention (LogicFlip,
produced by a GPT-4o critic at temperature 0.8) applied to the SAME target steps
as the symbolic interventions, under the same protocols. Baselines are shared
with run_audit.py (same keys), so this script must run after the NL audit.

Writes results/raw_<agent>_nlcritic.jsonl (agent calls) and
results/critic_cache.jsonl (critic calls; costed at GPT-4o prices).

Usage: python run_critic.py --agent gpt-4o-mini
"""
import argparse
import asyncio
import json
import time

from openai import AsyncOpenAI

import symexec as sx
from run_audit import (NL_PROMPT, NL_SYSTEM, P1_SUFFIX, P2_CONTINUE, P2F_FORCE, PRICES, RESULTS, Runner,
                       HERE)

CRITIC_MODEL = "gpt-4o"
CRITIC_SYSTEM = ("You are an expert at creating counterfactual reasoning steps. "
                 "Provide only the modified reasoning step, no explanations.")
# LogicFlip prompt of the original Ariadne critic (intervention_engine.py), plus "keep the label".
CRITIC_PROMPT = """You are a causal reasoning critic. Your task is to create a contradictory version of the following reasoning step.

Original Reasoning Step:
"{step}"

Intervention Type: logic_flip

Create a version that flips the logic:
- If it says something is True, make it False
- If it says Increase, make it Decrease
- If it says positive, make it negative
- Reverse the logical direction while keeping the same structure

Keep the "Step {n}:" label at the start. Provide ONLY the modified reasoning step, maintaining similar length and style. Do not include explanations or meta-commentary."""


class CriticRunner(Runner):
    def __init__(self, agent, concurrency, cap_run):
        super().__init__(agent, "nlcritic", concurrency, cap_run)
        self.critic = AsyncOpenAI(max_retries=4, timeout=120)
        self.cache_f = RESULTS / "critic_cache.jsonl"
        self.cache = {}
        if self.cache_f.exists():
            for l in self.cache_f.open():
                r = json.loads(l)
                self.cache[r["key"]] = r
        # original + baseline records come from the NL audit file
        self.nl = {}
        for l in (RESULTS / f"raw_{agent}_nl.jsonl").open():
            r = json.loads(l)
            self.nl[r["key"]] = r

    async def critic_step(self, key, line, n):
        if key in self.cache:
            return self.cache[key]["step"]
        async with self.sem:
            resp = await self.critic.chat.completions.create(
                model=CRITIC_MODEL, temperature=0.8, max_tokens=300,
                messages=[{"role": "system", "content": CRITIC_SYSTEM},
                          {"role": "user", "content": CRITIC_PROMPT.format(step=line.strip(), n=n)}])
        step = (resp.choices[0].message.content or "").strip().strip('"').strip()
        step = " ".join(step.splitlines())  # one line, as in the trace
        pin, pout = PRICES[CRITIC_MODEL]
        cost = (resp.usage.prompt_tokens * pin + resp.usage.completion_tokens * pout) / 1e6
        rec = {"key": key, "step": step, "orig_line": line, "cost": cost, "ts": time.time()}
        async with self.lock:
            with self.cache_f.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            self.cache[key] = rec
        return step

    async def problem(self, p):
        pid = p["id"]
        orig = self.nl.get(f"{pid}|orig")
        if orig is None:
            return
        steps, comps, node_of = sx.align_nl(p, orig["response"])
        if not sx.same(sx.parse_answer(orig["response"]), p["answer"]):
            return
        user = NL_PROMPT.format(q=p["question"])
        sys_m = {"role": "system", "content": NL_SYSTEM}
        jobs = []
        for pos, k in sx.pick_targets(p, node_of).items():
            if k is None:
                continue
            node = node_of[k]
            new_line = await self.critic_step(f"{self.agent}|{pid}|{pos}", steps[k]["line"], steps[k]["n"])
            if not sx.STEP_RE.match(new_line):
                new_line = f"Step {steps[k]['n']}: " + new_line
            # symbolic validity check of the critic's step: does it state a different value?
            st = sx.parse_steps(new_line)
            comp = sx.step_computation(st[0]) if st else None
            v2 = comp["value"] if comp else None
            changed_value = v2 is not None and not sx.same(v2, p["values"][node])
            implied = sx.implied_answer(p, node, v2) if changed_value else None
            prefix = "\n".join([s["line"] for s in steps[:k]] + [new_line])
            for prot in ("P1", "P2", "P2F"):
                if prot == "P1":
                    msgs = [sys_m, {"role": "user", "content": user + P1_SUFFIX.format(prefix=prefix)}]
                elif prot == "P2":
                    msgs = [sys_m, {"role": "user", "content": user}, {"role": "assistant", "content": prefix},
                            {"role": "user", "content": P2_CONTINUE}]
                else:
                    msgs = [sys_m, {"role": "user", "content": user}, {"role": "assistant", "content": prefix},
                            {"role": "user", "content": P2F_FORCE}]
                meta = {"pid": pid, "kind": "int", "iv": "LF", "pos": pos, "prot": prot, "node": node,
                        "step_index": k, "n_steps": len(steps), "orig_value": p["values"][node],
                        "v2": v2 if changed_value else None, "implied": implied,
                        "critic_changed_value": changed_value, "critic_step": new_line, "prefix": prefix}
                jobs.append(self.call(f"{pid}|{pos}|{prot}|LF", msgs, 300 if prot == "P2F" else 700, meta))
        await asyncio.gather(*jobs)

    async def run(self, problems):
        self.sem = asyncio.Semaphore(self.concurrency)
        self.lock = asyncio.Lock()
        await asyncio.gather(*(self.problem(p) for p in problems))
        print(f"done: {self.n_calls} new agent calls, agent cost ${self.ledger.run_cost:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=sorted(PRICES))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=float, default=2.0)
    args = ap.parse_args()
    problems = [json.loads(l) for l in open(HERE / "data" / "symarith.jsonl")]
    r = CriticRunner(args.agent, args.concurrency, args.budget)
    asyncio.run(r.run(problems))


if __name__ == "__main__":
    main()
