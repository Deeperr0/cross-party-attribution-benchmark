"""Check whether self-consistency predicts accuracy against the gold labels.

Multi-party answers are split by whether they agree with the model's own implied
ranking, and gold accuracy is compared between the two groups.
"""

import json
from collections import defaultdict

from scipy.stats import fisher_exact

from score_selfconsistency import compute as sc_compute


def gold_map():
    bench = json.load(open("data/benchmark.json"))
    g = {}
    for c in ["japan", "south_korea"]:
        for i in bench[c]["issues"]:
            g[(c, i["issue_code"])] = (set(i["highest_party_ties"]),
                                       set(i["lowest_party_ties"]))
    return g


GOLD = gold_map()


def analyse(suffix, label, models=None):
    comparable, *_ = sc_compute(f"scored/outputs_scored_singleparty{suffix}.json",
                                f"scored/outputs_scored_multiparty{suffix}.json")
    if models:
        comparable = [c for c in comparable if c["model"] in models]

    cells = {"agree": [0, 0], "disagree": [0, 0]}
    for c in comparable:
        if c["highest_agree"] is None or c["stated_highest"] is None:
            continue
        gh, _ = GOLD[(c["country"], c["issue_code"])]
        correct = c["stated_highest"] in gh
        side = "agree" if c["highest_agree"] else "disagree"
        cells[side][0] += correct
        cells[side][1] += 1

    a, na = cells["agree"]
    d, nd = cells["disagree"]
    if not na or not nd:
        print(f"  {label}: insufficient data")
        return
    _, p = fisher_exact([[a, na - a], [d, nd - d]])
    print(f"  {label}")
    print(f"    self-CONSISTENT answers   -> gold-correct {a}/{na} = {100*a/na:.1f}%")
    print(f"    self-CONTRADICTORY answers-> gold-correct {d}/{nd} = {100*d/nd:.1f}%")
    print(f"    difference {100*a/na - 100*d/nd:+.1f} points, Fisher p={p:.4f}")
    print(f"    -> a contradiction flag raises the odds of a wrong answer "
          f"{(1-d/nd)/(1-a/na):.2f}x" if a / na < 1 else "")


def models_in(suffix):
    return {r["model"] for r in json.load(open(f"scored/outputs_scored_singleparty{suffix}.json"))}


common = models_in("") & models_in("_structured")
print(f"models present in both runs: {len(common)}\n")

print("DOES SELF-CONSISTENCY PREDICT ACCURACY? (highest-emphasis judgment)")
analyse("", "free-text run", common)
print()
analyse("_structured", "semi-structured run", common)
print()
analyse("_grounded_structured", "semi-structured, grounded", common)
