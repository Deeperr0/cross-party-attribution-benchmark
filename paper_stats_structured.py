"""Statistics reported in the paper, computed on the tagged run and printed next
to the free-text run. Run after score_structured.py (with and without --grounded).
"""

import json
from collections import Counter, defaultdict

from scipy.stats import chi2_contingency, fisher_exact, binomtest

from score_selfconsistency import compute as sc_compute

SHORT = {"cohere-command-a": "Cohere-A", "groq-gpt-oss-120b": "gpt-oss-120b",
         "groq-gpt-oss-20b": "gpt-oss-20b", "mistral-large": "Mistral-L",
         "zhipu-glm5": "GLM-5", "anthropic-sonnet": "Claude-Sonnet",
         "deepseek-v3": "DeepSeek-V3", "gemini-2.5-flash": "Gemini-Flash"}
VERD = ["correct", "partial", "misattributed", "hallucinated", "no_knowledge"]
DROP = ("TRUNCATED_EXCLUDE", "AMBIGUOUS_NEEDS_MANUAL_READ")


def load_single(suffix):
    return [r for r in json.load(open(f"scored/outputs_scored_singleparty{suffix}.json"))
            if r["issue_code"] != "per408" and r["verdict"] not in DROP]


def load_multi(suffix):
    return [r for r in json.load(open(f"scored/outputs_scored_multiparty{suffix}.json"))
            if r["issue_code"] != "per408"]


def rule(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def faithfulness(sp, label):
    n = len(sp)
    c = sum(1 for r in sp if r["verdict"] == "correct")
    gold = Counter(r["gold_bucket"] for r in sp)
    maj = 100 * max(gold.values()) / n
    print(f"  {label:14} exact-match {c}/{n} = {100*c/n:5.1f}%   "
          f"(uniform chance 25.0%, majority-class baseline {maj:.1f}%)")
    return c, n


def inflation(sp, label):
    n = len(sp)
    pred = sum(1 for r in sp if r["extracted_bucket"] in
               ("moderate emphasis", "major emphasis / core priority"))
    gold = sum(1 for r in sp if r["gold_bucket"] in
               ("moderate emphasis", "major emphasis / core priority"))
    print(f"  {label:14} says moderate-or-core {100*pred/n:5.1f}%   truth {100*gold/n:5.1f}%")


def ranking(mp, label):
    out = []
    for lab, key in [("highest", "highest_correct"), ("lowest", "lowest_correct")]:
        v = [r[key] for r in mp if r[key] is not None]
        c = sum(v)
        p = binomtest(c, len(v), 0.25, alternative="greater").pvalue
        out.append((c, len(v)))
        print(f"  {label:14} {lab:8} {c}/{len(v)} = {100*c/len(v):5.1f}%   binomial p={p:.2e}")
    coll = sum(1 for r in mp if r["extracted_highest"]
               and r["extracted_highest"] == r["extracted_lowest"])
    print(f"  {label:14} MOST==LEAST collapses: {coll}/{len(mp)}")
    return out


def selfcons(suffix, label):
    comparable, n, n_hi, n_lo, hi, lo = sc_compute(
        f"scored/outputs_scored_singleparty{suffix}.json",
        f"scored/outputs_scored_multiparty{suffix}.json")
    print(f"  {label:14} highest {hi:.1%} (n={n_hi})   lowest {lo:.1%} (n={n_lo})   "
          f"comparable cells={n}")
    return (round(hi * n_hi), n_hi), (round(lo * n_lo), n_lo), comparable


def compare(a, b, lab):
    (ac, an), (bc, bn) = a, b
    _, p = fisher_exact([[bc, bn - bc], [ac, an - ac]])
    print(f"  {lab}: free-text {100*ac/an:.1f}%  ->  structured {100*bc/bn:.1f}%   Fisher p={p:.3f}")


rule("Single-party exact-match accuracy")
sp_free, sp_str = load_single(""), load_single("_structured")
f_free = faithfulness(sp_free, "free-text")
f_str = faithfulness(sp_str, "structured")
print()
inflation(sp_free, "free-text")
inflation(sp_str, "structured")

rule("Verdict distribution, tagged run (cols: Corr Part Misat Hall NoKn)")
by = defaultdict(Counter)
for r in sp_str:
    by[r["model"]][r["verdict"]] += 1
tot = Counter()
for m in sorted(by, key=lambda m: -by[m]["correct"] / sum(by[m].values())):
    n = sum(by[m].values())
    tot += by[m]
    print(f"  {SHORT.get(m, m):14} " + " ".join(f"{100*by[m][v]/n:5.1f}" for v in VERD) + f"   n={n}")
N = sum(tot.values())
print(f"  {'OVERALL':14} " + " ".join(f"{100*tot[v]/N:5.1f}" for v in VERD) + f"   n={N}")
rows = [[by[m]["no_knowledge"], sum(by[m].values()) - by[m]["no_knowledge"]] for m in by]
chi2, p, _, _ = chi2_contingency(rows)
print(f"\n  calibration (no_knowledge across models): chi2={chi2:.2f}, p={p:.2e}")

rule("Multi-party ranking accuracy")
mp_free, mp_str = load_multi(""), load_multi("_structured")
r_free = ranking(mp_free, "free-text")
r_str = ranking(mp_str, "structured")

rule("Self-consistency")
hi_free, lo_free, _ = selfcons("", "free-text")
hi_str, lo_str, _ = selfcons("_structured", "structured")
print()
compare(hi_free, hi_str, "highest")
compare(lo_free, lo_str, "lowest ")

rule("Self-consistency by model, tagged run")
sc = json.load(open("scored/outputs_selfconsistency_structured.json")) \
    if __import__("os").path.exists("scored/outputs_selfconsistency_structured.json") else None
if sc is None:
    _, _, _, _, _, _ = sc_compute("scored/outputs_scored_singleparty_structured.json",
                                  "scored/outputs_scored_multiparty_structured.json")
    comparable = sc_compute("scored/outputs_scored_singleparty_structured.json",
                            "scored/outputs_scored_multiparty_structured.json")[0]
    json.dump(comparable, open("scored/outputs_selfconsistency_structured.json", "w"), indent=2)
    sc = comparable
agg = defaultdict(lambda: {"hi": [0, 0], "lo": [0, 0]})
for r in sc:
    for k, key in [("hi", "highest_agree"), ("lo", "lowest_agree")]:
        if r[key] is not None:
            agg[r["model"]][k][0] += r[key]
            agg[r["model"]][k][1] += 1
for m in sorted(agg):
    h, l = agg[m]["hi"], agg[m]["lo"]
    hs = f"{100*h[0]/h[1]:.1f} (n={h[1]})" if h[1] else "n/a"
    ls = f"{100*l[0]/l[1]:.1f} (n={l[1]})" if l[1] else "n/a"
    print(f"  {SHORT.get(m, m):14} highest={hs:16} lowest={ls}")

rule("Grounding, tagged run")
try:
    spg = load_single("_grounded_structured")
    gmodels = {r["model"] for r in spg}
    cu = sum(1 for r in sp_str if r["model"] in gmodels and r["verdict"] == "correct")
    nu = sum(1 for r in sp_str if r["model"] in gmodels)
    cg = sum(1 for r in spg if r["verdict"] == "correct")
    print(f"  single-party accuracy: ungrounded {100*cu/nu:.1f}%  grounded {100*cg/len(spg):.1f}%"
          f"  (n_ung={nu}, n_gr={len(spg)})")
    ghi, glo, gcomp = selfcons("_grounded_structured", "grounded")
    json.dump(gcomp, open("scored/outputs_selfconsistency_grounded_structured.json", "w"), indent=2)
    ucomp = sc_compute("scored/outputs_scored_singleparty_structured.json",
                       "scored/outputs_scored_multiparty_structured.json")[0]
    u = [r for r in ucomp if r["model"] in gmodels]

    def rate(recs, key):
        v = [r[key] for r in recs if r[key] is not None]
        return sum(v), len(v)
    for lab, key in [("highest", "highest_agree"), ("lowest", "lowest_agree")]:
        ua, un = rate(u, key)
        ga, gn = rate(gcomp, key)
        _, pf = fisher_exact([[ga, gn - ga], [ua, un - ua]])
        print(f"  {lab}: ungrounded {ua}/{un}={100*ua/un:.1f}%  grounded {ga}/{gn}={100*ga/gn:.1f}%"
              f"  Fisher p={pf:.3f}")
except FileNotFoundError:
    print("  grounded structured run not scored yet -- run score_structured.py --grounded")
