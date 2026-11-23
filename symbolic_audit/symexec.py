"""
Symbolic machinery for the neuro-symbolic Ariadne audit:
  * exact evaluation of a problem's computation graph, with do()-overrides;
  * parsing of natural-language CoT steps ("Step i: q = expr = value") and of
    straight-line Python programs, with a safe arithmetic evaluator;
  * alignment of trace steps to graph nodes, and exact symbolic interventions:
      RC (result corruption): the step's stated value v becomes v + delta;
      CC (consistent corruption): one given quantity a in the step's expression
          becomes a', and the stated value is recomputed, so the step is
          arithmetically valid but misreads the problem.
"""
import ast
import math
import operator
import random
import re

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


# ---------------------------------------------------------------- graph -----

def evaluate_program(ops, leaves, override=None):
    """Evaluate a chain program. `override` = {node: value} implements do(node = value)."""
    override = override or {}
    env = dict(leaves)
    for node, op, l, r in ops:
        env[node] = override[node] if node in override else OPS[op](env[l], env[r])
    return env


def implied_answer(problem, node, value):
    """Counterfactual answer y* = G(do(node = value))."""
    env = evaluate_program(problem["ops"], problem["leaves"], {node: value})
    return env[problem["ops"][-1][0]]


def final_node(problem):
    return problem["ops"][-1][0]


def leaf_operand(problem, node):
    """The given quantity that node consumes (the right operand for x1 = a op b)."""
    for n, op, l, r in problem["ops"]:
        if n == node:
            if r in problem["leaves"]:
                return r
            if l in problem["leaves"]:
                return l
    return None


def _all_numbers(problem):
    return set(problem["leaves"].values()) | set(problem["values"].values())


def rc_params(problem, node):
    """Deterministic result corruption for `node`: returns v' (same for every agent)."""
    rng = random.Random(f"{problem['id']}|{node}|rc")
    taken = _all_numbers(problem)
    x = problem["values"][node]
    for _ in range(1000):
        delta = rng.choice([-1, 1]) * rng.randint(3, 12)
        v = x + delta
        if v <= 0 or v in taken:
            continue
        y = implied_answer(problem, node, v)
        if y <= 0 or y == problem["answer"]:
            continue
        return v
    return None


def cc_params(problem, node):
    """Deterministic consistent corruption: replace leaf a by a' in the step computing `node`."""
    leaf = leaf_operand(problem, node)
    if leaf is None:
        return None
    rng = random.Random(f"{problem['id']}|{node}|cc")
    a = problem["leaves"][leaf]
    taken = _all_numbers(problem)
    leaves2 = dict(problem["leaves"])
    for _ in range(1000):
        delta = rng.choice([-1, 1]) * (rng.randint(1, 3) if a <= 12 else rng.randint(2, 9))
        a2 = a + delta
        if a2 <= 0 or a2 in taken:
            continue
        leaves2[leaf] = a2
        # value of `node` with the misread operand, everything upstream unchanged
        env = evaluate_program(problem["ops"], problem["leaves"])
        for n, op, l, r in problem["ops"]:
            if n == node:
                lv = a2 if l == leaf else env[l]
                rv = a2 if r == leaf else env[r]
                v = OPS[op](lv, rv)
        if v <= 0 or v in taken:
            continue
        y = implied_answer(problem, node, v)
        if y <= 0 or y == problem["answer"]:
            continue
        return {"leaf": leaf, "a": a, "a2": a2, "v": v}
    return None


# ------------------------------------------------------------ safe eval -----

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}
_FUN = {"round": round, "int": int, "abs": abs, "min": min, "max": max, "float": float}


def safe_eval(node, env=None):
    env = env or {}
    if isinstance(node, ast.Expression):
        return safe_eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"undefined name {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        l, r = safe_eval(node.left, env), safe_eval(node.right, env)
        if isinstance(node.op, ast.Pow) and abs(r) > 8:
            raise ValueError("power too large")
        return _BIN[type(node.op)](l, r)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = safe_eval(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUN and not node.keywords:
        return _FUN[node.func.id](*[safe_eval(a, env) for a in node.args])
    raise ValueError(f"disallowed expression: {ast.dump(node)[:80]}")


def eval_expr(src, env=None):
    return safe_eval(ast.parse(src, mode="eval"), env)


def same(a, b):
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)


# --------------------------------------------------- NL step parsing --------

NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
STEP_RE = re.compile(r"^\s*[#>*_\-\s]*step\s*(\d+)\s*[*_]*\s*[:.)]\s*[*_]*\s*", re.I)
FINAL_RE = re.compile(r"final\s+answer\s*[*_]*\s*[:：]?\s*[*_]*\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.I)


def num(tok):
    return float(tok.replace(",", ""))


def fmt_like(value, original_token):
    v = int(value) if float(value).is_integer() else value
    if "," in original_token and isinstance(v, int):
        return f"{v:,}"
    return str(v)


def parse_answer(text):
    if text is None:
        return None
    m = FINAL_RE.findall(text)
    if m:
        try:
            return num(m[-1])
        except ValueError:
            pass
    boxed = re.findall(r"\\boxed\{\s*\$?(-?\d[\d,]*(?:\.\d+)?)", text)
    if boxed:
        return num(boxed[-1])
    toks = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if toks:
        try:
            return num(toks[-1])
        except ValueError:
            return None
    return None


def parse_steps(text):
    """Return list of {'n', 'line', 'body_start'} for lines that start with 'Step i:'."""
    steps = []
    for line in text.splitlines():
        m = STEP_RE.match(line)
        if m:
            steps.append({"n": int(m.group(1)), "line": line, "body_start": m.end()})
    return steps


def _expr_tokens(seg, offset):
    """Tokenize an arithmetic expression inside free text: numbers (with spans) and operators."""
    s = seg
    toks = []
    i = 0
    while i < len(s):
        m = NUM_RE.match(s, i)
        if m and not (i > 0 and (s[i - 1].isalpha())):
            toks.append(("num", m.group(0), offset + m.start(), offset + m.end()))
            i = m.end()
            continue
        c = s[i]
        prev_c = s[i - 1] if i > 0 else ""
        next_c = s[i + 1] if i + 1 < len(s) else ""
        if (c == "/" and (prev_c.isalpha() or next_c.isalpha())) or (c == "-" and prev_c.isalpha() and next_c.isalpha()):
            pass  # unit notation (seats/row) or hyphenated word, not an operator
        elif c in "+-*/()":
            toks.append(("op", c, None, None))
        elif c in "\u2212\u2013":
            toks.append(("op", "-", None, None))
        elif c in "\u00d7\u00b7\u2217":
            toks.append(("op", "*", None, None))
        elif c == "\u00f7":
            toks.append(("op", "/", None, None))
        elif c in "xX" and toks and toks[-1][0] == "num":
            # 'x' used as multiplication sign between two numbers
            rest = s[i + 1:].lstrip(" $")
            if rest[:1].isdigit():
                toks.append(("op", "*", None, None))
        i += 1
    return toks


def step_computation(step):
    """Parse 'Step i: ... = expr = value'. Returns dict or None."""
    line, b0 = step["line"], step["body_start"]
    body = line[b0:]
    eqs = [i for i, c in enumerate(body) if c == "="]
    if not eqs:
        return None
    last = eqs[-1]
    prev = eqs[-2] if len(eqs) >= 2 else -1
    vseg = body[last + 1:]
    vm = NUM_RE.search(vseg)
    if not vm:
        return None
    neg = vseg[:vm.start()].strip().endswith("-") or vseg[:vm.start()].strip().endswith("\u2212")
    value = num(vm.group(0)) * (-1 if neg else 1)
    vspan = (b0 + last + 1 + vm.start(), b0 + last + 1 + vm.end(), vm.group(0))
    eseg = body[prev + 1:last]
    toks = _expr_tokens(eseg, b0 + prev + 1)
    # drop leading/trailing operators and unbalanced junk
    while toks and toks[0][0] == "op" and toks[0][1] not in "(-":
        toks.pop(0)
    while toks and toks[-1][0] == "op" and toks[-1][1] != ")":
        toks.pop()
    src = " ".join(t[1].replace(",", "") for t in toks)
    has_op = any(t[0] == "op" and t[1] in "+-*/" for t in toks)
    try:
        ev = eval_expr(src) if src else None
    except Exception:
        ev = None
    return {"value": value, "value_span": vspan, "expr_src": src, "expr_value": ev,
            "num_tokens": [t for t in toks if t[0] == "num"], "has_op": has_op,
            "valid": has_op and ev is not None and same(ev, value)}


def align_nl(problem, text):
    """Map NL trace steps to graph nodes by exact value. Returns (steps, comps, node_of_step)."""
    steps = parse_steps(text)
    comps = [step_computation(s) for s in steps]
    val2node = {v: n for n, v in problem["values"].items()}
    node_of = []
    for c in comps:
        n = None
        if c and c["valid"]:
            for v, nd in val2node.items():
                if same(c["value"], v):
                    n = nd
        node_of.append(n)
    return steps, comps, node_of


def pick_targets(problem, node_of):
    """first: earliest step aligned to a non-final node; last: first step aligned to the final node."""
    fin = final_node(problem)
    first = next((i for i, n in enumerate(node_of) if n is not None and n != fin), None)
    last = next((i for i, n in enumerate(node_of) if n == fin), None)
    if first is not None and last is not None and first > last:
        first = None
    return {"first": first, "last": last}


def corrupt_nl(problem, step, comp, node, kind):
    """Return (new_line, v') or None. kind in {'RC','CC'}."""
    line = step["line"]
    vs, ve, vtok = comp["value_span"]
    if kind == "RC":
        v2 = rc_params(problem, node)
        if v2 is None:
            return None
        # replace the stated value and every other mention of it in the step
        # (e.g. "450 widgets = 45 x 10 = 450 widgets"), so the original result is not left visible
        b0 = step["body_start"]
        spans = [(m.start() + b0, m.end() + b0, m.group(0)) for m in NUM_RE.finditer(line[b0:vs])
                 if not (m.start() + b0 > 0 and line[m.start() + b0 - 1].isalpha()) and same(num(m.group(0)), comp["value"])]
        new = line[:vs] + fmt_like(v2, vtok) + line[ve:]
        for s2, e2, t2 in sorted(spans, reverse=True):
            new = new[:s2] + fmt_like(v2, t2) + new[e2:]
        return new, v2
    if kind == "CC":
        p = cc_params(problem, node)
        if p is None:
            return None
        hits = [t for t in comp["num_tokens"] if same(num(t[1]), p["a"])]
        if len(hits) != 1:
            return None
        _, tok, s, e = hits[0]
        new_src = " ".join((str(p["a2"]) if t is hits[0] else t[1].replace(",", ""))
                           if t[0] == "num" else t[1]
                           for t in _expr_tokens_from_comp(comp))
        try:
            v2 = eval_expr(new_src)
        except Exception:
            return None
        if not same(v2, p["v"]):
            return None
        v2 = int(round(v2)) if float(v2).is_integer() else v2
        # replace the value and EVERY mention of a before it (e.g. "8 trays x 22 per tray = 8 x 22"),
        # so the corrupted step is internally consistent; splice right-to-left to keep offsets valid
        b0 = step["body_start"]
        spans = [(m.start() + b0, m.end() + b0, m.group(0)) for m in NUM_RE.finditer(line[b0:vs])
                 if not (m.start() + b0 > 0 and line[m.start() + b0 - 1].isalpha()) and same(num(m.group(0)), p["a"])]
        new = line[:vs] + fmt_like(v2, vtok) + line[ve:]
        for s2, e2, t2 in sorted(spans, reverse=True):
            new = new[:s2] + fmt_like(p["a2"], t2) + new[e2:]
        return new, v2
    raise ValueError(kind)


def _expr_tokens_from_comp(comp):
    # rebuild full token list (numbers + operators) from expr_src, mapping numbers back to spans
    out, k = [], 0
    nums = comp["num_tokens"]
    for piece in re.findall(r"\d+(?:\.\d+)?|[+\-*/()]", comp["expr_src"]):
        if piece[0].isdigit():
            out.append(nums[k])
            k += 1
        else:
            out.append(("op", piece, None, None))
    return out


# ------------------------------------------------------- program traces -----

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    m = FENCE_RE.findall(text or "")
    code = m[0] if m else (text or "")
    keep = []
    for l in code.splitlines():
        l = l.split("#")[0].rstrip()
        if not l.strip() or l.strip().startswith("```"):
            continue
        try:  # keep only single-statement assignment / expression lines; drop prose
            body = ast.parse(l.strip()).body
        except SyntaxError:
            continue
        if len(body) == 1 and isinstance(body[0], (ast.Assign, ast.Expr)):
            keep.append(l.strip())
    return keep


def run_program(lines):
    """Execute straight-line assignments. Returns (env, per-line records) or raises."""
    env, recs = {}, []
    for i, l in enumerate(lines):
        tree = ast.parse(l)
        if len(tree.body) != 1:
            raise ValueError("one statement per line required")
        st = tree.body[0]
        if isinstance(st, ast.Expr):  # e.g. print(answer)
            recs.append(None)
            continue
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
            raise ValueError("only simple assignments allowed")
        v = safe_eval(st.value, env)
        env[st.targets[0].id] = v
        has_op = any(isinstance(n, ast.BinOp) for n in ast.walk(st.value))
        recs.append({"target": st.targets[0].id, "value": v, "has_op": has_op})
    return env, recs


def program_answer(lines):
    try:
        env, recs = run_program(lines)
    except Exception:
        return None
    if "answer" in env:
        return env["answer"]
    vals = [r["value"] for r in recs if r]
    return vals[-1] if vals else None


def align_code(problem, lines):
    try:
        env, recs = run_program(lines)
    except Exception:
        return None, None
    val2node = {v: n for n, v in problem["values"].items()}
    node_of = []
    for r in recs:
        n = None
        if r and r["has_op"]:
            for v, nd in val2node.items():
                if same(r["value"], v):
                    n = nd
        node_of.append(n)
    return recs, node_of


def corrupt_code(problem, lines, idx, node):
    """CC on program line idx: replace the literal for the node's given quantity a by a'."""
    p = cc_params(problem, node)
    if p is None:
        return None
    lit = r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])"

    def replace_literal(line):
        eq = line.index("=")
        rhs = line[eq + 1:]
        hits = [m for m in re.finditer(lit, rhs) if same(float(m.group(0)), p["a"])]
        if len(hits) != 1:
            return None
        m = hits[0]
        return line[:eq + 1] + rhs[:m.start()] + str(p["a2"]) + rhs[m.end():]

    new = replace_literal(lines[idx])
    if new is not None:
        new_lines = lines[:idx] + [new] + lines[idx + 1:]
    else:
        # the agent stored the given quantity in its own variable (e.g. `price = 15`):
        # corrupt that definition line, which must precede the target line
        try:
            tree = ast.parse(lines[idx]).body[0]
        except Exception:
            return None
        names = {n.id for n in ast.walk(tree.value) if isinstance(n, ast.Name)}
        cand = None
        for j in range(idx - 1, -1, -1):
            st = ast.parse(lines[j]).body[0]
            if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name) and st.targets[0].id in names:
                names.discard(st.targets[0].id)
                if isinstance(st.value, ast.Constant) and same(st.value.value, p["a"]):
                    cand = j
                    break
        if cand is None:
            return None
        new = replace_literal(lines[cand])
        if new is None:
            return None
        new_lines = lines[:cand] + [new] + lines[cand + 1:]
    try:
        _, recs = run_program(new_lines[:idx + 1])
    except Exception:
        return None
    v2 = recs[idx]["value"]
    if not same(v2, p["v"]):
        return None
    return new_lines, v2
