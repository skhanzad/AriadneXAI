# Neuro-symbolic Ariadne audit (NeSy-FM @ AAAI-27 experiments)

Every number in `paper/main.tex` is produced by `analyze.py` from the raw logs in `results/`.

| File | Role |
|---|---|
| `generate_symarith.py` | Builds SymArith: 210 problems (14 templates x 15 seeds, seed 2027), each with its generating program. Output: `data/symarith.jsonl`. |
| `symexec.py` | Symbolic core: program evaluation with `do()` overrides, NL step and Python-program parsing, alignment of steps to program nodes, and the exact RC/CC interventions. |
| `run_audit.py` | Agent runs: original trace, RC/CC at the first and last aligned step, protocols P1/P2/P2F (NL) and CONT/EXEC (programs), and paired baselines. Resumable; logs every call with its cost. |
| `run_critic.py` | Comparison condition: the original LogicFlip text critic (GPT-4o, T=0.8) applied to the same steps. Run it after the NL audit. |
| `refresh_cc.py` | One-off repair: drops RC/CC records whose corrupted step left an earlier mention of the original value, so `run_audit.py` regenerates them. |
| `analyze.py` | Exact scoring (follow / revert / other), baselines, bootstrap CIs, McNemar + Holm. Writes `results/summary.{json,md}` and `results/latex_cells.txt`. |

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install openai numpy scipy
export OPENAI_API_KEY=...            # or: set -a; source .env; set +a
cd symbolic_audit
../.venv/bin/python generate_symarith.py
for a in gpt-4o-mini gpt-4.1-mini gpt-4o gpt-5-mini; do
  ../.venv/bin/python run_audit.py --agent $a --mode nl
  ../.venv/bin/python run_audit.py --agent $a --mode code
  ../.venv/bin/python run_critic.py --agent $a
done
# open-weight agents, served locally by Ollama (no API key; endpoint via OLLAMA_URL)
for a in gemma2:latest gemma4:e4b qwen2.5-coder:14b; do
  ollama pull $a
  ../.venv/bin/python run_audit.py --agent $a --mode nl --concurrency 4
  ../.venv/bin/python run_audit.py --agent $a --mode code --concurrency 4
done
../.venv/bin/python analyze.py
```

The retained runs comprise 26,211 API calls costing US$12.29 at list prices (cached-input discounts ignored), plus 11,252 local Ollama calls. `run_audit.py` enforces a per-run budget (`--budget`) and a global cap of $34.
