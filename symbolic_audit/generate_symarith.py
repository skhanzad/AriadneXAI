"""
Generate SymArith: multi-step arithmetic word problems with a known symbolic
computation graph (a chain of +, -, * operations over given quantities).

Every problem stores its program, so an intervention do(x_j = v) on any
intermediate quantity has an exactly computable counterfactual answer.

Usage: python symbolic_audit/generate_symarith.py [--per-template 15] [--seed 2027]
"""
import argparse
import json
import random
from pathlib import Path

from symexec import evaluate_program

# Each template: name, text with {leaf} slots, ops [(node, op, left, right)], leaf ranges.
# Operands are leaf names (single letters) or earlier node names (x1, x2, ...).
# Every leaf is used exactly once, so do(leaf) and do(node) coincide for the first use.
TEMPLATES = [
    ("bakery",
     "A bakery bakes {a} trays of muffins with {b} muffins on each tray. It sells {c} muffins in the "
     "morning and then bakes {d} more muffins in the afternoon. How many muffins does the bakery have "
     "at the end of the day?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "+", "x2", "d")],
     {"a": (4, 12), "b": (6, 24), "c": (10, 60), "d": (10, 60)}),
    ("pencils",
     "A school has {a} classrooms with {b} students in each classroom. On Monday, {c} students are "
     "absent. Each student who is present needs {d} pencils, and the school already has {e} pencils. "
     "How many more pencils must the school buy?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "*", "x2", "d"), ("x4", "-", "x3", "e")],
     {"a": (4, 12), "b": (18, 32), "c": (5, 30), "d": (2, 6), "e": (20, 90)}),
    ("orchard",
     "A farmer has {a} rows of apple trees with {b} trees in each row. A storm destroys {c} trees. "
     "Each remaining tree produces {d} apples. How many apples does the orchard produce?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "*", "x2", "d")],
     {"a": (5, 15), "b": (8, 20), "c": (4, 30), "d": (20, 60)}),
    ("warehouse",
     "A warehouse receives {a} pallets with {b} boxes on each pallet. It ships out {c} boxes and then "
     "receives {d} more boxes. Each box holds {e} items. How many items are in the warehouse?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "+", "x2", "d"), ("x4", "*", "x3", "e")],
     {"a": (3, 9), "b": (12, 30), "c": (10, 50), "d": (10, 50), "e": (3, 12)}),
    ("savings",
     "Maya saves {a} dollars every week for {b} weeks. She spends {c} dollars on a bike and then earns "
     "{d} dollars at a garage sale. How many dollars does Maya have now?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "+", "x2", "d")],
     {"a": (12, 45), "b": (6, 20), "c": (40, 150), "d": (15, 90)}),
    ("train",
     "A train has {a} cars with {b} passengers in each car. At the first stop, {c} passengers get off "
     "and {d} passengers get on. Every passenger then pays a fare of {e} dollars. How many dollars in "
     "fares do the passengers on the train pay in total?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "+", "x2", "d"), ("x4", "*", "x3", "e")],
     {"a": (4, 10), "b": (15, 40), "c": (10, 50), "d": (5, 40), "e": (3, 9)}),
    ("garden",
     "A gardener plants {a} beds with {b} tulip bulbs in each bed. {c} bulbs fail to sprout. Each "
     "sprouted bulb produces {d} flowers. The gardener picks {e} flowers for a bouquet and then buys "
     "{f} potted flowers. How many flowers does the gardener have?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "*", "x2", "d"), ("x4", "-", "x3", "e"),
      ("x5", "+", "x4", "f")],
     {"a": (3, 9), "b": (8, 20), "c": (3, 20), "d": (2, 5), "e": (10, 60), "f": (5, 40)}),
    ("concert",
     "A concert hall has {a} rows with {b} seats in each row. {c} seats are reserved for staff, and all "
     "other seats are sold for {d} dollars each. What is the ticket revenue in dollars?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "*", "x2", "d")],
     {"a": (10, 30), "b": (12, 30), "c": (8, 40), "d": (12, 45)}),
    ("factory",
     "A factory makes {a} widgets per hour for {b} hours. Quality control rejects {c} of these widgets. "
     "The accepted widgets are added to a stock of {d} widgets, and then {e} widgets are shipped. How "
     "many widgets remain in stock?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "+", "x2", "d"), ("x4", "-", "x3", "e")],
     {"a": (20, 60), "b": (5, 12), "c": (10, 40), "d": (50, 200), "e": (30, 150)}),
    ("raffle",
     "Each of {a} volunteers sells {b} raffle tickets. {c} tickets are refunded. Each remaining ticket "
     "earns the club {d} dollars, and the club spends {e} dollars on prizes. What is the club's profit "
     "in dollars?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "*", "x2", "d"), ("x4", "-", "x3", "e")],
     {"a": (6, 20), "b": (8, 25), "c": (5, 30), "d": (2, 9), "e": (40, 200)}),
    ("aquarium",
     "An aquarium has {a} tanks with {b} fish in each tank. It sells {c} fish and then buys {d} more "
     "fish. Each fish eats {e} grams of food per day, and the aquarium's turtles eat {f} grams per day. "
     "How many grams of food does the aquarium use per day?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "+", "x2", "d"), ("x4", "*", "x3", "e"),
      ("x5", "+", "x4", "f")],
     {"a": (4, 12), "b": (6, 18), "c": (5, 25), "d": (5, 30), "e": (2, 7), "f": (20, 90)}),
    ("printer",
     "A printer prints {a} pages per minute for {b} minutes. {c} pages jam and are thrown away. The "
     "good pages are bound into a report together with {d} cover pages. How many pages does the report "
     "have?",
     [("x1", "*", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "+", "x2", "d")],
     {"a": (12, 40), "b": (6, 20), "c": (5, 40), "d": (2, 9)}),
    ("museum",
     "A school bus leaves with {a} students. At the first stop {b} students get on, and at the second "
     "stop {c} students get off. Each student on the bus then pays {d} dollars for a museum ticket, and "
     "the teacher pays {e} dollars for a teacher ticket. How many dollars are paid for tickets in total?",
     [("x1", "+", "a", "b"), ("x2", "-", "x1", "c"), ("x3", "*", "x2", "d"), ("x4", "+", "x3", "e")],
     {"a": (15, 40), "b": (5, 20), "c": (3, 15), "d": (6, 18), "e": (10, 30)}),
    ("tanks",
     "A tank holds {a} liters of water. {b} liters leak out, and then {c} liters are added. A farm has "
     "{d} tanks in exactly this state. How many liters of water are in the farm's tanks?",
     [("x1", "-", "a", "b"), ("x2", "+", "x1", "c"), ("x3", "*", "x2", "d")],
     {"a": (120, 400), "b": (15, 80), "c": (10, 60), "d": (3, 12)}),
]

MAX_ANSWER = 100_000


def sample_problem(rng, template):
    name, text, ops, ranges = template
    for _ in range(10_000):
        leaves = {k: rng.randint(lo, hi) for k, (lo, hi) in ranges.items()}
        values = evaluate_program(ops, leaves)
        inter = [values[n] for n, *_ in ops]
        numbers = list(leaves.values()) + inter
        if any(v <= 0 for v in inter):
            continue
        if len(set(numbers)) != len(numbers):  # all quantities distinct -> unambiguous alignment
            continue
        if inter[-1] > MAX_ANSWER:
            continue
        # keep subtractions meaningful (not removing almost everything)
        ok = True
        for (n, op, l, r) in ops:
            if op == "-" and values[n] < 0.2 * (values.get(l, leaves.get(l))):
                ok = False
        if not ok:
            continue
        return {
            "template": name,
            "question": text.format(**leaves),
            "leaves": leaves,
            "ops": [list(o) for o in ops],
            "values": {n: values[n] for n, *_ in ops},
            "answer": inter[-1],
            "n_ops": len(ops),
        }
    raise RuntimeError(f"could not sample template {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "symarith.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    for t in TEMPLATES:
        k = 0
        while k < args.per_template:
            p = sample_problem(rng, t)
            if p["question"] in seen:
                continue
            seen.add(p["question"])
            k += 1
            rows.append(p)
    rng.shuffle(rows)
    with out.open("w") as f:
        for i, p in enumerate(rows):
            p = {"id": f"A{i:03d}", **p}
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(rows)} problems to {out}")


if __name__ == "__main__":
    main()
