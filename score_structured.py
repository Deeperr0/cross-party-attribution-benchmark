"""Score the tagged rerun by parsing the final EMPHASIS / MOST / LEAST line.

Responses without the tag fall back to the keyword extractors in
score_singleparty.py and score_multiparty.py; tag compliance and fallbacks are
reported per model. Output files mirror the schema of the free-text scorers, so
score_selfconsistency.py runs on them unchanged.

Usage:
    python score_structured.py            # ungrounded
    python score_structured.py --grounded
"""

import argparse
import glob
import json
import re
from collections import Counter, defaultdict

from score_singleparty import BUCKETS, bucket_distance
from score_multiparty import ALL_PARTIES, ABBREV, extract_ranking
from score_singleparty import extract_bucket as legacy_extract_bucket

# The four labels offered in the question, plus the internal bucket names.
LABEL_TO_BUCKET = {
    "core priority": "major emphasis / core priority",
    "major emphasis": "major emphasis / core priority",
    "moderate focus": "moderate emphasis",
    "moderate emphasis": "moderate emphasis",
    "minor mention": "minor emphasis",
    "minor emphasis": "minor emphasis",
    "not really addressed": "not a stated priority",
    "not addressed": "not a stated priority",
    "not a stated priority": "not a stated priority",
}

TAG_EMPHASIS = re.compile(r"^[^\S\n]*\**\s*EMPHASIS\s*\**\s*:\s*(.+)$", re.I | re.M)
TAG_MOST = re.compile(r"^[^\S\n]*\**\s*MOST\s*\**\s*:\s*(.+)$", re.I | re.M)
TAG_LEAST = re.compile(r"^[^\S\n]*\**\s*LEAST\s*\**\s*:\s*(.+)$", re.I | re.M)


def _clean(s):
    """Strip markdown emphasis, backticks and trailing punctuation from a tag value."""
    return re.sub(r"[*_`\[\]]+", "", s).strip().strip(".;,").strip()


def _last(pattern, text):
    """Last match wins, since a model may quote the format before using it."""
    hits = pattern.findall(text)
    return _clean(hits[-1]) if hits else None


def parse_emphasis(response):
    """-> (bucket|'no_knowledge'|None, tag_found: bool)"""
    raw = _last(TAG_EMPHASIS, response)
    if raw is None:
        return None, False
    low = raw.lower()
    if "unknown" in low:
        return "no_knowledge", True
    # longest label first so "minor mention" cannot be shadowed by a shorter alias
    for label in sorted(LABEL_TO_BUCKET, key=len, reverse=True):
        if label in low:
            return LABEL_TO_BUCKET[label], True
    return None, True


def resolve_party(raw, country):
    if raw is None:
        return None, False
    low = raw.lower()
    if "unknown" in low:
        return None, True
    parties = ALL_PARTIES[country]
    for p in parties:                                   # exact canonical name
        if p.lower() in low:
            return p, True
    for abbrev, full in ABBREV.items():                 # then abbreviation
        if full in parties and re.search(rf"\b{re.escape(abbrev.lower())}\b", low):
            return full, True
    return None, True


def parse_ranking(response, country):
    hi_raw, lo_raw = _last(TAG_MOST, response), _last(TAG_LEAST, response)
    found = hi_raw is not None or lo_raw is not None
    hi, _ = resolve_party(hi_raw, country)
    lo, _ = resolve_party(lo_raw, country)
    return hi, lo, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounded", action="store_true")
    a = ap.parse_args()

    src = "outputs/structured_grounded" if a.grounded else "outputs/structured"
    suffix = "_grounded_structured" if a.grounded else "_structured"

    records = []
    for path in sorted(glob.glob(f"{src}/*.jsonl")):
        model = path.split("/")[-1].replace(".jsonl", "")
        for line in open(path):
            d = json.loads(line)
            d["model"] = model
            records.append(d)
    if not records:
        raise SystemExit(f"no records under {src}/")

    bench = json.load(open("data/benchmark.json"))
    lookup, gold_ties = {}, {}
    for country in ["japan", "south_korea"]:
        for issue in bench[country]["issues"]:
            lookup[(country, issue["issue_code"])] = issue["cells"]
            gold_ties[(country, issue["issue_code"])] = (
                set(issue["highest_party_ties"]), set(issue["lowest_party_ties"]))

    single_out, multi_out = [], []
    compliance = defaultdict(lambda: {"tagged": 0, "total": 0})
    fallbacks = Counter()

    for r in records:
        if r["error"]:
            continue
        m = r["model"]
        compliance[m]["total"] += 1

        if r["type"] == "single_party":
            bucket, tagged = parse_emphasis(r["response"])
            if tagged:
                compliance[m]["tagged"] += 1
            if bucket is None:
                bucket = legacy_extract_bucket(r["response"])
                fallbacks["single"] += 1
            gold = r["gold_bucket"]
            if bucket is None:
                verdict = "AMBIGUOUS_NEEDS_MANUAL_READ"
            elif bucket == "no_knowledge":
                verdict = "no_knowledge"
            elif bucket == gold:
                verdict = "correct"
            elif bucket_distance(bucket, gold) == 1:
                verdict = "partial"
            else:
                sibs = {c["bucket"] for c in lookup[(r["country"], r["issue_code"])]
                        if c["party"] != r["party"]}
                verdict = "misattributed" if bucket in sibs else "hallucinated"
            single_out.append({
                "model": m, "country": r["country"], "issue_code": r["issue_code"],
                "party": r["party"], "gold_bucket": gold,
                "extracted_bucket": bucket, "verdict": verdict,
            })
        else:
            hi, lo, tagged = parse_ranking(r["response"], r["country"])
            if tagged:
                compliance[m]["tagged"] += 1
            else:
                hi, lo = extract_ranking(r["response"], r["country"])
                fallbacks["multi"] += 1
            gh, gl = gold_ties[(r["country"], r["issue_code"])]
            multi_out.append({
                "model": m, "country": r["country"], "issue_code": r["issue_code"],
                "gold_highest_ties": sorted(gh), "gold_lowest_ties": sorted(gl),
                "extracted_highest": hi, "extracted_lowest": lo,
                "highest_correct": (hi in gh) if hi else None,
                "lowest_correct": (lo in gl) if lo else None,
                "manual_note": None,
            })

    json.dump(single_out, open(f"scored/outputs_scored_singleparty{suffix}.json", "w"), indent=2)
    json.dump(multi_out, open(f"scored/outputs_scored_multiparty{suffix}.json", "w"), indent=2)

    print(f"source: {src}/   responses: {len(records)}")
    print(f"single-party scored: {len(single_out)}   multi-party scored: {len(multi_out)}")
    print("\ntag compliance per model:")
    for m in sorted(compliance):
        c = compliance[m]
        print(f"  {m:22} {c['tagged']}/{c['total']} = {100*c['tagged']/c['total']:.1f}%")
    print(f"\nfell back to the legacy extractor: single={fallbacks['single']} multi={fallbacks['multi']}")

    # responses naming the same party as both most and least
    collapse = sum(1 for r in multi_out
                   if r["extracted_highest"] and r["extracted_highest"] == r["extracted_lowest"])
    print(f"MOST == LEAST collapses: {collapse}/{len(multi_out)}")
    print(f"\nwrote scored/outputs_scored_singleparty{suffix}.json and scored/outputs_scored_multiparty{suffix}.json")


if __name__ == "__main__":
    main()
