import json
import glob

MOST_KEYWORDS = ["most emphasis", "most attention", "greatest emphasis", "highest priority",
                 "highest emphasis", "strongest emphasis", "top priority", "places the most",
                 "emphasizes .* the most", "most:"]
LEAST_KEYWORDS = ["least emphasis", "least attention", "lowest priority", "least priority",
                  "smallest emphasis", "weakest emphasis", "places the least", "not really addressed",
                  "not a priority", "not really a priority", "least:"]

ALL_PARTIES = {
    "japan": ["Japanese Communist Party", "Constitutional Democratic Party of Japan", "Komeito", "Liberal Democratic Party"],
    "south_korea": ["Justice Party", "Democratic Party of Korea", "People Party", "United Future Party"],
}
ABBREV = {
    "JCP": "Japanese Communist Party", "CDP": "Constitutional Democratic Party of Japan",
    "LDP": "Liberal Democratic Party", "New Komeito": "Komeito",
    "DPK": "Democratic Party of Korea", "Democratic Party": "Democratic Party of Korea",
    "UFP": "United Future Party", "PPP": "United Future Party", "JP": "Justice Party",
}


def find_party_near(text, keyword, parties):
    idx = text.lower().find(keyword)
    if idx == -1:
        return None
    window = text[max(0, idx - 300):idx + 100]
    best_party, best_dist = None, None
    for p in parties:
        names_to_check = [p] + [k for k, v in ABBREV.items() if v == p]
        for name in names_to_check:
            pos = window.lower().rfind(name.lower())
            if pos != -1:
                dist = len(window) - pos
                if best_dist is None or dist < best_dist:
                    best_dist, best_party = dist, p
    return best_party


def extract_ranking(response, country):
    parties = ALL_PARTIES[country]
    highest = None
    for kw in MOST_KEYWORDS:
        highest = find_party_near(response, kw, parties)
        if highest:
            break
    lowest = None
    for kw in LEAST_KEYWORDS:
        lowest = find_party_near(response, kw, parties)
        if lowest:
            break
    return highest, lowest


# Hand-coded verdicts for responses the keyword extractor could not resolve.
MANUAL_OVERRIDES = {
    ("groq-gpt-oss-120b", "japan", "per501"): {
        "extracted_highest": "Japanese Communist Party",  # correct
        "extracted_lowest": "Liberal Democratic Party",   # incorrect: lowest is CDP (1.53 vs 4.51)
        "note": "Model's stated 'least emphasis' (LDP) is a real ranking error, not a tie or extraction failure -- true lowest is CDP.",
    },
    ("groq-gpt-oss-120b", "japan", "per408"): {
        "extracted_highest": "Liberal Democratic Party",  # incorrect: highest is CDP (0.51)
        "extracted_lowest": "Komeito",                     # correct: tied with LDP at 0.0
        "note": "Highest is a genuine miss (LDP claimed, real highest is CDP). Lowest (Komeito) ties with gold LDP at value 0.0 -- tie-aware scoring credits this as correct.",
    },
    ("mistral-large", "south_korea", "per411"): {
        "extracted_highest": "Justice Party",    # stated ranking: Justice > DPK > People > UFP
        "extracted_lowest": "United Future Party",
        "note": "Both a genuine miss -- real highest is DPK (7.68), real lowest is People Party (1.04), no ties involved here.",
    },
    ("mistral-large", "south_korea", "per408"): {
        "extracted_highest": "Justice Party",     # stated ranking: Justice > DPK > People > UFP
        "extracted_lowest": "United Future Party",
        "note": "Notable inversion: model's claimed highest (Justice, real value 0.32) is close to the real LOWEST tier, and claimed lowest (UFP, real value 0.81) is actually the REAL HIGHEST. A clean illustrative misattribution example for the paper.",
    },
}


if __name__ == "__main__":
    bench = json.load(open("data/benchmark.json"))
    tie_lookup = {}
    for country in ["japan", "south_korea"]:
        for issue in bench[country]["issues"]:
            tie_lookup[(country, issue["issue_code"])] = {
                "highest_ties": issue["highest_party_ties"],
                "lowest_ties": issue["lowest_party_ties"],
            }

    records = []
    for path in sorted(glob.glob("outputs/*.jsonl")):
        model = path.split("/")[-1].replace(".jsonl", "")
        with open(path) as f:
            for l in f:
                d = json.loads(l)
                d["model"] = model
                records.append(d)

    multi = [r for r in records if r["type"] == "multi_party" and r["error"] is None]
    results = []
    still_ambiguous = []
    for r in multi:
        key = (r["model"], r["country"], r["issue_code"])
        if key in MANUAL_OVERRIDES:
            ov = MANUAL_OVERRIDES[key]
            highest, lowest = ov["extracted_highest"], ov["extracted_lowest"]
            note = ov["note"]
        else:
            highest, lowest = extract_ranking(r["response"], r["country"])
            note = None

        ties = tie_lookup[(r["country"], r["issue_code"])]
        highest_correct = highest in ties["highest_ties"] if highest else None
        lowest_correct = lowest in ties["lowest_ties"] if lowest else None

        results.append({
            "model": r["model"], "country": r["country"], "issue_code": r["issue_code"],
            "gold_highest_ties": ties["highest_ties"], "gold_lowest_ties": ties["lowest_ties"],
            "extracted_highest": highest, "extracted_lowest": lowest,
            "highest_correct": highest_correct, "lowest_correct": lowest_correct,
            "manual_note": note,
        })
        if highest is None or lowest is None:
            still_ambiguous.append(r)

    json.dump(results, open("scored/outputs_scored_multiparty.json", "w"), indent=2)
    print(f"Scored {len(results)} multi-party items ({len(MANUAL_OVERRIDES)} via manual read, rest automated).")
    print(f"{len(still_ambiguous)} still unresolved.")

    n = len(results)
    hi_acc = sum(1 for r in results if r["highest_correct"]) / n
    lo_acc = sum(1 for r in results if r["lowest_correct"]) / n
    print(f"Overall ranking accuracy (tie-aware): highest={hi_acc:.1%}, lowest={lo_acc:.1%}")
