import json
import random
import sys
import time
from pathlib import Path

from run_eval import MODELS, DISPATCH, PROVIDER_SLEEP, call_with_retry, sanitize_error


def build_grounding_block(cells, country, issue_code):
    # Shuffle party order per issue so position carries no information.
    ordered = list(cells)
    random.Random(f"{country}-{issue_code}").shuffle(ordered)
    facts = "; ".join(f"{c['party']} -- {c['bucket']}" for c in ordered)
    return (
        "For reference, here is each party's documented emphasis on this issue, "
        f"from their official manifestos: {facts}.\n\n"
    )


def build_prompt_list():
    bench = json.load(open("data/benchmark.json"))
    prompts = []
    for country_key in ["japan", "south_korea"]:
        country = bench[country_key]
        for issue in country["issues"]:
            grounding = build_grounding_block(issue["cells"], country_key, issue["issue_code"])
            for i, cell in enumerate(issue["cells"]):
                prompts.append({
                    "country": country_key,
                    "issue_code": issue["issue_code"],
                    "type": "single_party",
                    "party": cell["party"],
                    "gold_bucket": cell["bucket"],
                    "gold_value": cell["value"],
                    "prompt": grounding + issue["single_party_prompts"][i],
                })
            prompts.append({
                "country": country_key,
                "issue_code": issue["issue_code"],
                "type": "multi_party",
                "gold_highest": issue["highest_party"],
                "gold_lowest": issue["lowest_party"],
                "prompt": grounding + issue["multi_party_prompt"],
            })
    return prompts


def run(model_key, limit=None, sleep=None):
    cfg = MODELS[model_key]
    fn = DISPATCH[cfg["provider"]]
    if sleep is None:
        sleep = PROVIDER_SLEEP.get(cfg["provider"], 2.0)
    prompts = build_prompt_list()
    if limit:
        prompts = prompts[:limit]
    Path("outputs/grounded").mkdir(parents=True, exist_ok=True)
    out_path = f"outputs/grounded/{model_key}.jsonl"
    with open(out_path, "a") as f:
        for i, item in enumerate(prompts):
            response, err = call_with_retry(fn, cfg["model"], item["prompt"])
            item["response"] = response
            item["error"] = sanitize_error(err) if err else None
            f.write(json.dumps(item) + "\n")
            f.flush()
            print(f"[{model_key}] {i+1}/{len(prompts)} {'OK' if not item['error'] else 'ERR (after retries): ' + item['error']}")
            time.sleep(sleep)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    model_key = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(model_key, limit=limit)
