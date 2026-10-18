"""Rerun of the evaluation with a tagged final line appended to each prompt.

The prose question is unchanged; the model is additionally asked to end its reply
with "EMPHASIS: <label>" (single-party) or "MOST:/LEAST: <party>" (multi-party),
using the labels already offered in the question. UNKNOWN is always allowed.

Usage:
    python run_eval_structured.py <model_key> [--grounded] [--limit N]
    python run_eval_structured.py --all [--grounded]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from run_eval import (MODELS, DISPATCH, PROVIDER_SLEEP, call_with_retry,
                      sanitize_error, build_prompt_list)
from run_eval_grounded import build_prompt_list as build_prompt_list_grounded

SINGLE_TAG = (
    "\n\nAfter your answer, end your reply with one final line in exactly this "
    "format and nothing else on that line:\n"
    "EMPHASIS: <label>\n"
    "where <label> is exactly one of: core priority | moderate focus | "
    "minor mention | not really addressed | UNKNOWN\n"
    "Use UNKNOWN if you are not confident you know this party's real platform."
)

MULTI_TAG = (
    "\n\nAfter your answer, end your reply with exactly these two final lines, "
    "in this format and nothing else on those lines:\n"
    "MOST: <party name>\n"
    "LEAST: <party name>\n"
    "Use each party's name exactly as written in the question. Use UNKNOWN for "
    "either line if you are not confident."
)


def add_tag(item):
    item = dict(item)
    item["prompt"] = item["prompt"] + (
        SINGLE_TAG if item["type"] == "single_party" else MULTI_TAG)
    return item


def run(model_key, grounded=False, limit=None, sleep=None):
    cfg = MODELS[model_key]
    fn = DISPATCH[cfg["provider"]]
    if sleep is None:
        sleep = PROVIDER_SLEEP.get(cfg["provider"], 2.0)

    builder = build_prompt_list_grounded if grounded else build_prompt_list
    prompts = [add_tag(p) for p in builder()]
    if limit:
        prompts = prompts[:limit]

    out_dir = "outputs/structured_grounded" if grounded else "outputs/structured"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/{model_key}.jsonl"

    # Resume a partially completed run.
    done = 0
    if Path(out_path).exists():
        done = sum(1 for _ in open(out_path))
        if done:
            print(f"[{model_key}] resuming, {done} already written")
    prompts = prompts[done:]

    with open(out_path, "a") as f:
        for i, item in enumerate(prompts):
            response, err = call_with_retry(fn, cfg["model"], item["prompt"])
            item["response"] = response
            item["error"] = sanitize_error(err) if err else None
            f.write(json.dumps(item) + "\n")
            f.flush()
            status = "OK" if not item["error"] else "ERR: " + item["error"][:60]
            print(f"[{model_key}] {done + i + 1}/{done + len(prompts)} {status}", flush=True)
            time.sleep(sleep)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--grounded", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    if a.all:
        for m in MODELS:
            try:
                run(m, grounded=a.grounded, limit=a.limit)
            except Exception as e:
                print(f"[{m}] FAILED: {sanitize_error(str(e))}", file=sys.stderr)
    elif a.model:
        run(a.model, grounded=a.grounded, limit=a.limit)
    else:
        ap.error("give a model key or --all")
