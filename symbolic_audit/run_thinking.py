"""
Ariadne audit of a reasoning model with VISIBLE thinking (Qwen3 via Ollama).

Two kinds of intervention, both scored exactly against the generating program:

  P2 / P2F : as in run_audit.py, on the "Step i:" lines of the visible answer
             (the model thinks again, visibly, before continuing).
  TH       : on the thinking itself. The first computation in the original
             thinking that evaluates to the target node (e.g. "38 + 14 equals 52")
             receives the same RC/CC corruption as the answer steps; the thinking
             is cut right after it and the model continues thinking from there
             (raw completion with the chat template), then answers. A paired
             baseline continues from the unedited cut.

Every record stores the continuation's thinking ("reasoning"), so repairs can be
classified as silent or explicit separately in the thinking and in the answer.

Writes results/raw_<model>_nl.jsonl (same format as run_audit.py; TH uses prot="TH").

Usage:
  python run_thinking.py --agent qwen3:8b --limit 5
"""
import argparse
import asyncio
import json
import re
import time

import httpx

import run_audit as ra
import symexec as sx

THINK_MODELS = ["qwen3:8b"]
ra.PRICES.update({m: (0.0, 0.0) for m in THINK_MODELS})
ra.OLLAMA_MODELS.extend(m for m in THINK_MODELS if m not in ra.OLLAMA_MODELS)
OLLAMA_NATIVE = ra.OLLAMA_URL.rsplit("/v1", 1)[0]
MAX_TOKENS = 4096  # thinking + answer
TEMPERATURE = 0.7

_N = r"(\d[\d,]*(?:\.\d+)?)"
_OPS = {"+": "+", "plus": "+", "-": "-", "−": "-", "minus": "-", "*": "*", "×": "*", "x": "*",
        "times": "*", "multiplied by": "*", "\\times": "*"}
THINK_CALC_RE = re.compile(
    _N + r"\s*(\+|-|−|\*|×|x|\\times|plus|minus|times|multiplied by)\s*\$?" + _N
    + r"\s*\$?\s*(?:=|equals|is|gives|makes|would be|comes to)\s*\$?" + _N)
STANDALONE = r"(?<![\d.,]){}(?![\d]|[.,]\d)"


def qwen3_prompt(system, user, thinking_prefix):
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{thinking_prefix}")


def _replace_all(text, old, new):
    """Replace every standalone mention of number `old` (also with thousands separators)."""
    variants = {str(int(old))} | ({f"{int(old):,}"} if old >= 1000 else set())
    for v in variants:
        text = re.sub(STANDALONE.format(re.escape(v)), str(int(new)), text)
    return text


def think_calcs(problem, thinking):
    """All binary calculations in the thinking whose arithmetic is right and whose result is a graph node."""
    val2node = {v: n for n, v in problem["values"].items()}
    out = []
    for m in THINK_CALC_RE.finditer(thinking):
        a, op, b, v = m.group(1), _OPS[m.group(2)], m.group(3), m.group(4)
        try:
            ev = sx.eval_expr(f"{sx.num(a)} {op} {sx.num(b)}")
        except Exception:
            continue
        if not sx.same(ev, sx.num(v)):
            continue
        node = next((n for vv, n in val2node.items() if sx.same(vv, sx.num(v))), None)
        if node is not None:
            out.append({"m": m, "node": node, "a": sx.num(a), "b": sx.num(b), "op": op})
    return out


def think_targets(problem, calcs):
    fin = sx.final_node(problem)
    first = next((c for c in calcs if c["node"] != fin), None)
    last = next((c for c in calcs if c["node"] == fin), None)
    if first is not None and last is not None and first["m"].start() > last["m"].start():
        first = None
    return {"first": first, "last": last}


def corrupt_think(problem, thinking, calc, kind):
    """Return (prefix, v2) with the thinking cut right after the corrupted calculation, or None."""
    m, node = calc["m"], calc["node"]
    pre, cut = thinking[:m.start()], thinking[m.start():m.end()]
    if kind == "base":
        return thinking[:m.end()], None
    if kind == "RC":
        v2 = sx.rc_params(problem, node)
        if v2 is None:
            return None
        new_cut = cut[:m.start(4) - m.start()] + str(int(v2))
        return _replace_all(pre, problem["values"][node], v2) + new_cut, v2
    if kind == "CC":
        p = sx.cc_params(problem, node)
        if p is None:
            return None
        a, a2 = p["a"], p["a2"]
        if sx.same(calc["a"], a) == sx.same(calc["b"], a):  # the misread quantity must be exactly one operand
            return None
        x, y = (a2, calc["b"]) if sx.same(calc["a"], a) else (calc["a"], a2)
        v2 = sx.eval_expr(f"{x} {calc['op']} {y}")
        if not sx.same(v2, p["v"]):
            return None
        v2 = int(round(v2))
        g = [m.start(i) - m.start() for i in (1, 3, 4)] + [m.end(i) - m.start() for i in (1, 3, 4)]
        s1, s3, s4, e1, e3, e4 = g
        new_cut = (cut[:s1] + str(int(x)) + cut[e1:s3] + str(int(y)) + cut[e3:s4] + str(v2))
        return _replace_all(_replace_all(pre, problem["values"][node], v2), a, a2) + new_cut, v2
    raise ValueError(kind)


class ThinkRunner(ra.Runner):
    PROTOCOLS = ("P2", "P2F")

    def __init__(self, agent, concurrency):
        super().__init__(agent, "nl", concurrency, cap_run=1.0)
        self.http = httpx.AsyncClient(timeout=1200)

    async def _post(self, path, payload):
        for attempt in range(6):
            try:
                async with self.sem:
                    r = await self.http.post(f"{OLLAMA_NATIVE}{path}", json=payload)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                print(f"[retry {attempt}] {type(e).__name__}: {str(e)[:120]}", flush=True)
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("Ollama call failed")

    async def chat_think(self, messages):
        j = await self._post("/api/chat", {"model": self.agent, "messages": messages, "think": True, "stream": False,
                                           "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}})
        msg = j["message"]
        usage = {"prompt_tokens": j.get("prompt_eval_count", 0), "completion_tokens": j.get("eval_count", 0)}
        return msg.get("content", ""), msg.get("thinking", ""), usage, j.get("done_reason")

    async def raw_think(self, prompt):
        j = await self._post("/api/generate", {"model": self.agent, "prompt": prompt, "raw": True, "stream": False,
                                               "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}})
        text = j.get("response", "")
        thinking, _, content = text.partition("</think>")
        if not _:  # never closed its thinking: no answer
            thinking, content = text, ""
        usage = {"prompt_tokens": j.get("prompt_eval_count", 0), "completion_tokens": j.get("eval_count", 0)}
        return content.strip(), thinking, usage, j.get("done_reason")

    async def _log(self, key, meta, messages, fn):
        if key in self.done:
            return self.done[key]
        text, thinking, usage, finish = await fn()
        rec = {"key": key, "agent": self.agent, "mode": "nl", **meta, "messages": messages, "response": text,
               "reasoning": thinking, "finish_reason": finish, "usage": usage, "cost": 0.0, "ts": time.time()}
        async with self.lock:
            with self.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            self.done[key] = rec
            self.n_calls += 1
            if self.n_calls % 100 == 0:
                print(f"  {self.n_calls} calls", flush=True)
        return rec

    async def call(self, key, messages, max_tokens, meta):  # used by Runner.nl_problem
        return await self._log(key, meta, messages, lambda: self.chat_think(messages))

    async def nl_problem(self, p):
        await super().nl_problem(p)  # orig + P2/P2F on answer steps
        pid = p["id"]
        orig = self.done.get(f"{pid}|orig")
        if orig is None or not sx.same(sx.parse_answer(orig["response"]), p["answer"]):
            return
        thinking = orig.get("reasoning") or ""
        user = ra.NL_PROMPT.format(q=p["question"])
        targets = think_targets(p, think_calcs(p, thinking))
        jobs = []
        for pos, calc in targets.items():
            if calc is None:
                continue
            for cond in ("base", "RC", "CC"):
                c = corrupt_think(p, thinking, calc, cond)
                if c is None:
                    continue
                prefix, v2 = c
                prompt = qwen3_prompt(ra.NL_SYSTEM, user, prefix)
                meta = {"pid": pid, "kind": "base" if cond == "base" else "int", "iv": cond, "pos": pos,
                        "prot": "TH", "node": calc["node"], "orig_value": p["values"][calc["node"]], "v2": v2,
                        "implied": None if v2 is None else sx.implied_answer(p, calc["node"], v2),
                        "think_prefix": prefix, "think_cut": prefix[-200:]}
                jobs.append(self._log(f"{pid}|{pos}|TH|{cond}", meta, [{"raw": prompt}],
                                      lambda prompt=prompt: self.raw_think(prompt)))
        await asyncio.gather(*jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="qwen3:8b", choices=THINK_MODELS)
    ap.add_argument("--data", default=str(ra.HERE / "data" / "symarith.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    ra.RESULTS.mkdir(parents=True, exist_ok=True)
    problems = [json.loads(l) for l in open(args.data)]
    if args.limit:
        problems = problems[:args.limit]
    r = ThinkRunner(args.agent, args.concurrency)
    asyncio.run(r.run(problems))


if __name__ == "__main__":
    main()
