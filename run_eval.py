import json
import re
import sys
import time
import requests
from env_loader import load_env

env = load_env()

MAX_TOKENS = 1500
MAX_RETRIES = 5


def sanitize_error(text):
    # Keep API keys out of logged error messages.
    text = re.sub(r"(\?[^\s\"']*)", "?[REDACTED_QUERY]", text)
    text = re.sub(r"(key=|Bearer\s+)[A-Za-z0-9\-_.]{10,}", r"\1[REDACTED]", text)
    return text


MODELS = {
    "groq-gpt-oss-120b": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "groq-gpt-oss-20b": {"provider": "groq", "model": "openai/gpt-oss-20b"},
    "mistral-large": {"provider": "mistral", "model": "mistral-large-latest"},
    "zhipu-glm5": {"provider": "zhipu", "model": "glm-5"},
    "cohere-command-a": {"provider": "cohere", "model": "command-a-03-2025"},
    "anthropic-sonnet": {"provider": "anthropic", "model": "claude-sonnet-5"},
    "gemini-2.5-flash": {"provider": "gemini", "model": "models/gemini-2.5-flash"},
    "deepseek-v3": {"provider": "deepseek", "model": "deepseek-chat"},
}

SYSTEM_PROMPT = (
    "You are a knowledgeable political analyst. Answer concisely and directly. "
    "If you are not confident you know a party's real platform, say so explicitly "
    "rather than guessing."
)


class EmptyResponse(Exception):
    pass


def call_gemini(model, prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent",
        headers={"x-goog-api-key": env["GEMINI_API_KEY"]},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": MAX_TOKENS},
        },
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_groq(model, prompt):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {env['GROQ_API_KEY']}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        },
        timeout=90,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    if not content or not content.strip():
        # gpt-oss can spend the whole token budget on reasoning; retry.
        raise EmptyResponse("empty content despite 200 OK (reasoning-budget exhaustion)")
    return content


def call_mistral(model, prompt):
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {env['MISTRAL_API_KEY']}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_zhipu(model, prompt):
    key = env.get("ZHIPU_API_KEY") or env.get("ZHIPUAI_API_KEY")
    r = requests.post(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_cohere(model, prompt):
    r = requests.post(
        "https://api.cohere.com/v2/chat",
        headers={"Authorization": f"Bearer {env['COHERE_API_KEY']}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        },
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    return "".join(part.get("text", "") for part in data["message"]["content"])


def call_anthropic(model, prompt):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": env["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            # temperature is deprecated for this model and left at its default
        },
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


def call_deepseek(model, prompt):
    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


DISPATCH = {
    "gemini": call_gemini,
    "groq": call_groq,
    "mistral": call_mistral,
    "zhipu": call_zhipu,
    "cohere": call_cohere,
    "anthropic": call_anthropic,
    "deepseek": call_deepseek,
}


def call_with_retry(fn, model, prompt):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(model, prompt), None
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_err = f"{type(e).__name__}: {e}"
            if status == 429 or (status is not None and status >= 500):
                time.sleep(min(2 ** attempt, 30))
                continue
            break
        except (EmptyResponse, requests.Timeout, requests.ConnectionError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 30))
            continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break
    return None, last_err


def build_prompt_list():
    bench = json.load(open("data/benchmark.json"))
    prompts = []
    for country_key in ["japan", "south_korea"]:
        country = bench[country_key]
        for issue in country["issues"]:
            for i, cell in enumerate(issue["cells"]):
                prompts.append({
                    "country": country_key,
                    "issue_code": issue["issue_code"],
                    "type": "single_party",
                    "party": cell["party"],
                    "gold_bucket": cell["bucket"],
                    "gold_value": cell["value"],
                    "prompt": issue["single_party_prompts"][i],
                })
            prompts.append({
                "country": country_key,
                "issue_code": issue["issue_code"],
                "type": "multi_party",
                "gold_highest": issue["highest_party"],
                "gold_lowest": issue["lowest_party"],
                "prompt": issue["multi_party_prompt"],
            })
    return prompts


# Seconds between requests, set to stay under each provider's rate limit.
PROVIDER_SLEEP = {
    "groq": 0.6,
    "mistral": 16.0,
    "cohere": 3.5,
    "zhipu": 3.0,
    "gemini": 6.5,
    "anthropic": 1.0,
    "deepseek": 1.0,
}


def run(model_key, limit=None, sleep=None):
    cfg = MODELS[model_key]
    fn = DISPATCH[cfg["provider"]]
    if sleep is None:
        sleep = PROVIDER_SLEEP.get(cfg["provider"], 2.0)
    prompts = build_prompt_list()
    if limit:
        prompts = prompts[:limit]
    out_path = f"outputs/{model_key}.jsonl"
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
