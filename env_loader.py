"""Load API keys from the environment, falling back to a local .env file."""
import os
import re

_VALID = re.compile(r"^[A-Z0-9_]+$")


def load_env():
    env = {}
    for k, v in os.environ.items():
        if _VALID.match(k) and v:
            env[k] = v
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(env_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if _VALID.match(key) and val:
                    env.setdefault(key, val)
    except FileNotFoundError:
        pass
    return env


if __name__ == "__main__":
    print("Loaded keys (names only):", sorted(load_env().keys()))
