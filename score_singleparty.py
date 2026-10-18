import json
import re
import glob

BUCKETS = ["not a stated priority", "minor emphasis", "moderate emphasis", "major emphasis / core priority"]

NO_KNOWLEDGE_PATTERNS = [
    r"not confident", r"not familiar", r"don'?t know", r"do not know",
    r"not sure which", r"cannot confirm", r"i'?m not aware", r"unable to find",
    r"not clear which party", r"no official platform", r"i don'?t have (specific|detailed|enough)",
    r"i do not have (specific|detailed|enough)", r"which .* are you referring to",
    r"could you clarify", r"i'?m not certain",
    # Includes hedged phrasings such as "I don't have reliable, specific information".
    r"i don'?t have (reliable|confident)", r"i do not have (reliable|confident)",
]

BUCKET_PATTERNS = {
    # Generic phrases like "central to their platform" are not used: they can refer
    # to a different topic than the one being asked about.
    "major emphasis / core priority": [r"core priority", r"top priority", r"central pillar",
        r"flagship issue", r"major emphasis", r"major priority", r"high priority",
        r"core component", r"\bcore\b.{0,15}priority",
        r"defining issue"],
    # Compounds such as "moderate to core" or "moderate-to-high" are read as moderate.
    "moderate emphasis": [r"moderate focus", r"moderate emphasis", r"moderate priority",
        r"moderate.{0,8}high", r"moderate.{0,8}low", r"secondary priority",
        r"moderate to (core|major|top|high)", r"\*\*moderate\*\*", r"\bmoderately\b"],
    "minor emphasis": [r"minor mention", r"minor emphasis", r"limited emphasis", r"minor focus",
        r"minor.{0,20}mention", r"brief mention", r"passing mention"],
    "not a stated priority": [r"not really addressed", r"not a\b.{0,15}priority", r"not addressed",
        r"no explicit position", r"doesn'?t address", r"does not address",
        r"not (really )?mentioned", r"not a core (issue|focus)"],
}

# Normalize Unicode hyphens and dashes to ASCII before matching.
_DASH_CHARS = "‐‑‒–—―"


def normalize_dashes(text):
    for ch in _DASH_CHARS:
        text = text.replace(ch, "-")
    return text

# A match preceded by one of these within ~35 characters is treated as negated,
# e.g. "not a core priority" or "rather than a core priority".
NEGATION_WINDOW = 35
NEGATION_WORDS = re.compile(
    r"\b(not|n't|isn't|doesn't|don't|won't|never|hardly|rarely|no longer|"
    r"not necessarily|far from|less of a|rather than)\b", re.IGNORECASE)


def find_valid_matches(text, patterns):
    """Return positions of all matches NOT preceded by a negation word."""
    positions = []
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            window_start = max(0, m.start() - NEGATION_WINDOW)
            preceding = text[window_start:m.start()]
            if not NEGATION_WORDS.search(preceding):
                positions.append(m.start())
    return positions


SPECIFIC_LEVEL_BUCKETS = ["major emphasis / core priority", "moderate emphasis", "minor emphasis"]


def extract_bucket(response):
    response = normalize_dashes(response)
    if re.search("|".join(NO_KNOWLEDGE_PATTERNS), response, re.IGNORECASE):
        return "no_knowledge"

    # Last non-negated match position for each bucket.
    last_match = {}
    for bucket, patterns in BUCKET_PATTERNS.items():
        positions = find_valid_matches(response, patterns)
        if positions:
            last_match[bucket] = max(positions)

    if not last_match:
        return None

    # A specific level (major/moderate/minor) takes precedence over the generic
    # "not a stated priority", which is used only when no specific level matched.
    specific = {b: p for b, p in last_match.items() if b in SPECIFIC_LEVEL_BUCKETS}
    if specific:
        return max(specific, key=specific.get)
    return "not a stated priority"


def bucket_distance(b1, b2):
    return abs(BUCKETS.index(b1) - BUCKETS.index(b2))


SENTENCE_END = re.compile(r"[.!?]['\")\]]*\s*$|```\s*$")


def classify(record, benchmark_lookup):
    extracted = extract_bucket(record["response"])
    gold = record["gold_bucket"]
    if extracted is None:
        response = record["response"].strip()
        if response and not SENTENCE_END.search(response):
            # Truncated before any emphasis statement; excluded.
            return {"extracted_bucket": None, "verdict": "TRUNCATED_EXCLUDE"}
        return {"extracted_bucket": None, "verdict": "AMBIGUOUS_NEEDS_MANUAL_READ"}
    if extracted == "no_knowledge":
        return {"extracted_bucket": "no_knowledge", "verdict": "no_knowledge"}
    if extracted == gold:
        return {"extracted_bucket": extracted, "verdict": "correct"}
    dist = bucket_distance(extracted, gold)
    if dist == 1:
        return {"extracted_bucket": extracted, "verdict": "partial"}
    siblings = benchmark_lookup.get((record["country"], record["issue_code"]), [])
    sibling_buckets = {s["party"]: s["bucket"] for s in siblings if s["party"] != record["party"]}
    if extracted in sibling_buckets.values():
        return {"extracted_bucket": extracted, "verdict": "misattributed"}
    return {"extracted_bucket": extracted, "verdict": "hallucinated"}


if __name__ == "__main__":
    bench = json.load(open("data/benchmark.json"))
    benchmark_lookup = {}
    for country in ["japan", "south_korea"]:
        for issue in bench[country]["issues"]:
            benchmark_lookup[(country, issue["issue_code"])] = issue["cells"]

    records = []
    for path in sorted(glob.glob("outputs/*.jsonl")):
        model = path.split("/")[-1].replace(".jsonl", "")
        with open(path) as f:
            for l in f:
                d = json.loads(l)
                d["model"] = model
                records.append(d)

    single = [r for r in records if r["type"] == "single_party" and r["error"] is None]
    results = []
    ambiguous = []
    for r in single:
        c = classify(r, benchmark_lookup)
        result = {
            "model": r["model"], "country": r["country"], "issue_code": r["issue_code"],
            "party": r["party"], "gold_bucket": r["gold_bucket"],
            "extracted_bucket": c["extracted_bucket"], "verdict": c["verdict"],
        }
        results.append(result)
        if c["verdict"] == "AMBIGUOUS_NEEDS_MANUAL_READ":
            ambiguous.append(r)

    json.dump(results, open("scored/outputs_scored_singleparty.json", "w"), indent=2)
    print(f"Scored {len(results)} single-party items.")
    print(f"{len(ambiguous)} need manual review (no pattern matched).")
    from collections import Counter
    print(Counter(r["verdict"] for r in results))
