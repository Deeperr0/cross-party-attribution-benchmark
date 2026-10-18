import json
import sys

BUCKET_ORDER = ["not a stated priority", "minor emphasis", "moderate emphasis", "major emphasis / core priority"]
BUCKET_RANK = {b: i for i, b in enumerate(BUCKET_ORDER)}


CANONICAL_ORDER = {
    "japan": ["Japanese Communist Party", "Constitutional Democratic Party of Japan", "Komeito", "Liberal Democratic Party"],
    "south_korea": ["Justice Party", "Democratic Party of Korea", "People Party", "United Future Party"],
}


def implied_extreme_group(party_buckets, want_max):
    """Return the parties tied at the max/min bucket among scoreable answers.

    Returns None if fewer than two parties are scoreable or all share one bucket."""
    ranked = {p: BUCKET_RANK[b] for p, b in party_buckets.items() if b in BUCKET_RANK}
    if len(ranked) < 2 or len(set(ranked.values())) < 2:
        return None
    target = max(ranked.values()) if want_max else min(ranked.values())
    return {p for p, r in ranked.items() if r == target}


def compute(singleparty_path, multiparty_path):
    # per408 is excluded from all headline results (see paper, Section 3).
    single = [r for r in json.load(open(singleparty_path)) if r["issue_code"] != "per408"]
    multi = [r for r in json.load(open(multiparty_path)) if r["issue_code"] != "per408"]

    by_key = {}
    for r in single:
        key = (r["model"], r["country"], r["issue_code"])
        by_key.setdefault(key, {})[r["party"]] = r["extracted_bucket"]

    multi_by_key = {(r["model"], r["country"], r["issue_code"]): r for r in multi}

    comparable = []
    for key, party_buckets in by_key.items():
        if key not in multi_by_key:
            continue
        highest_group = implied_extreme_group(party_buckets, want_max=True)
        lowest_group = implied_extreme_group(party_buckets, want_max=False)
        if highest_group is None or lowest_group is None:
            continue
        m = multi_by_key[key]
        # No single party named (e.g. an explicit tie): missing, not a disagreement.
        comparable.append({
            "model": key[0], "country": key[1], "issue_code": key[2],
            "implied_highest_group": sorted(highest_group), "implied_lowest_group": sorted(lowest_group),
            "stated_highest": m["extracted_highest"], "stated_lowest": m["extracted_lowest"],
            "highest_agree": (m["extracted_highest"] in highest_group) if m["extracted_highest"] is not None else None,
            "lowest_agree": (m["extracted_lowest"] in lowest_group) if m["extracted_lowest"] is not None else None,
        })

    n = len(comparable)
    hi_scoreable = [c["highest_agree"] for c in comparable if c["highest_agree"] is not None]
    lo_scoreable = [c["lowest_agree"] for c in comparable if c["lowest_agree"] is not None]
    n_hi, n_lo = len(hi_scoreable), len(lo_scoreable)
    hi = sum(hi_scoreable) / n_hi if n_hi else float("nan")
    lo = sum(lo_scoreable) / n_lo if n_lo else float("nan")
    return comparable, n, n_hi, n_lo, hi, lo


if __name__ == "__main__":
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    single_path = f"scored/outputs_scored_singleparty{suffix}.json"
    multi_path = f"scored/outputs_scored_multiparty{suffix}.json"
    comparable, n, n_hi, n_lo, hi, lo = compute(single_path, multi_path)
    json.dump(comparable, open(f"scored/outputs_selfconsistency{suffix}.json", "w"), indent=2)
    no_stated_hi = sum(1 for c in comparable if c["highest_agree"] is None)
    no_stated_lo = sum(1 for c in comparable if c["lowest_agree"] is None)
    print(f"n={n} comparable model-issue combinations (single-party answers differ across parties)")
    print(f"highest-party self-consistency: {hi:.1%} (n={n_hi}; {no_stated_hi} excluded -- multi-party gave no single named winner)")
    print(f"lowest-party self-consistency: {lo:.1%} (n={n_lo}; {no_stated_lo} excluded -- multi-party gave no single named loser)")

    from collections import defaultdict
    per_model = defaultdict(lambda: {"hi": 0, "lo": 0, "n_hi": 0, "n_lo": 0})
    for c in comparable:
        pm = per_model[c["model"]]
        if c["highest_agree"] is not None:
            pm["n_hi"] += 1
            pm["hi"] += c["highest_agree"]
        if c["lowest_agree"] is not None:
            pm["n_lo"] += 1
            pm["lo"] += c["lowest_agree"]
    for model, pm in sorted(per_model.items()):
        hi_str = f"{pm['hi']/pm['n_hi']:.1%} (n={pm['n_hi']})" if pm["n_hi"] else "n/a"
        lo_str = f"{pm['lo']/pm['n_lo']:.1%} (n={pm['n_lo']})" if pm["n_lo"] else "n/a"
        print(f"  {model}: highest={hi_str} lowest={lo_str}")
