import glob
import json

from score_singleparty import classify
from score_multiparty import extract_ranking


def load(pattern):
    records = []
    for path in sorted(glob.glob(pattern)):
        model = path.split("/")[-1].replace(".jsonl", "")
        with open(path) as f:
            for l in f:
                d = json.loads(l)
                d["model"] = model
                records.append(d)
    return records


if __name__ == "__main__":
    bench = json.load(open("data/benchmark.json"))
    benchmark_lookup = {}
    tie_lookup = {}
    for country in ["japan", "south_korea"]:
        for issue in bench[country]["issues"]:
            benchmark_lookup[(country, issue["issue_code"])] = issue["cells"]
            tie_lookup[(country, issue["issue_code"])] = {
                "highest_ties": issue["highest_party_ties"],
                "lowest_ties": issue["lowest_party_ties"],
            }

    records = load("outputs/grounded/*.jsonl")

    # single-party
    single = [r for r in records if r["type"] == "single_party" and r["error"] is None]
    single_results = []
    ambiguous = []
    for r in single:
        c = classify(r, benchmark_lookup)
        result = {
            "model": r["model"], "country": r["country"], "issue_code": r["issue_code"],
            "party": r["party"], "gold_bucket": r["gold_bucket"],
            "extracted_bucket": c["extracted_bucket"], "verdict": c["verdict"],
        }
        single_results.append(result)
        if c["verdict"] == "AMBIGUOUS_NEEDS_MANUAL_READ":
            ambiguous.append(r)
    json.dump(single_results, open("scored/outputs_scored_singleparty_grounded.json", "w"), indent=2)
    from collections import Counter
    print(f"Grounded single-party: scored {len(single_results)}, {len(ambiguous)} need manual review")
    print(Counter(r["verdict"] for r in single_results))

    # multi-party
    multi = [r for r in records if r["type"] == "multi_party" and r["error"] is None]
    multi_results = []
    still_ambiguous = []
    for r in multi:
        highest, lowest = extract_ranking(r["response"], r["country"])
        ties = tie_lookup[(r["country"], r["issue_code"])]
        highest_correct = highest in ties["highest_ties"] if highest else None
        lowest_correct = lowest in ties["lowest_ties"] if lowest else None
        multi_results.append({
            "model": r["model"], "country": r["country"], "issue_code": r["issue_code"],
            "gold_highest_ties": ties["highest_ties"], "gold_lowest_ties": ties["lowest_ties"],
            "extracted_highest": highest, "extracted_lowest": lowest,
            "highest_correct": highest_correct, "lowest_correct": lowest_correct,
            "manual_note": None,
        })
        if highest is None or lowest is None:
            still_ambiguous.append(r)
    json.dump(multi_results, open("scored/outputs_scored_multiparty_grounded.json", "w"), indent=2)
    print(f"Grounded multi-party: scored {len(multi_results)}, {len(still_ambiguous)} still unresolved")
