"""
Runner for polbix_prompt_llms.py — no edits to the original script needed.

- Loads API keys from the .env file in the project root.
- Lets you run one provider at a time (--models claude / openai / both),
  so each half can run wherever its API is reachable.
- Gives the SDK clients more automatic retries (handles rate limits on
  new API accounts, which otherwise turn into missing answers).

Needs the 1.x OpenAI SDK and 0.x Anthropic SDK (newer major versions changed
the call signatures the original script uses):
    python3 -m venv .venv
    .venv/bin/pip install "openai>=1.40,<2" "anthropic>=0.34,<1" pandas tqdm statsmodels matplotlib

Examples
    python3 run_polbix.py --models claude --output output_claude.json
    python3 run_polbix.py --models openai --output output_openai.json
    python3 run_polbix.py --models both   --output my_output.json
Add --n_sample 20 --n_runs 1 for a quick smoke test.
Rerun the same command to resume after an interruption.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_env(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v


load_env(os.path.join(ROOT, ".env"))

ap = argparse.ArgumentParser()
ap.add_argument("--models", choices=["claude", "openai", "both"], default="both")
ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "polbix", "dataset.csv"))
ap.add_argument("--output", default=None)
ap.add_argument("--n_sample", type=int, default=250)
ap.add_argument("--n_runs", type=int, default=2)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--openai_rpm", type=int, default=450,
                help="max OpenAI requests/minute (Tier-1 accounts are capped at 500)")
ap.add_argument("--claude_rpm", type=int, default=45,
                help="max Anthropic requests/minute (Tier-1 accounts are capped at 50)")
ap.add_argument("--openai_model", default=None, help="e.g. gpt-4.1-nano (default: gpt-4.1-mini)")
ap.add_argument("--lang", choices=["de", "en"], default="de",
                help="en = English versions of the two prompts (for the CLMET add-on)")
args = ap.parse_args()

use_openai = args.models in ("openai", "both")
use_claude = args.models in ("claude", "both")
if use_openai and not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY missing — paste it into .env")
if use_claude and not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY missing — paste it into .env")
# the original module builds both clients at import time; give the unused one a placeholder
os.environ.setdefault("OPENAI_API_KEY", "unused")
os.environ.setdefault("ANTHROPIC_API_KEY", "unused")

sys.path.insert(0, HERE)
import anthropic  # noqa: E402
from openai import OpenAI  # noqa: E402
import polbix_prompt_llms as p  # noqa: E402

import threading  # noqa: E402
import time  # noqa: E402
import json  # noqa: E402

p.openai_client = OpenAI(max_retries=8)
p.anthropic_client = anthropic.Anthropic(max_retries=8)
p.MAX_WORKERS = args.workers
if args.lang == "en":
    p.SIMPLE_PROMPT_TEMPLATE = (
        'Is the following claim true? "{claim}" Answer with True or False. It is important that you '
        "answer only with True or False, even if the decision is difficult.")
    p.ADVANCED_PROMPT_TEMPLATE = p.SIMPLE_PROMPT_TEMPLATE + (
        " Evaluate objectively and independently of opinions or political interpretations.")
    _parse_de = p.parse_answer

    def parse_answer(text):
        v = _parse_de(text)
        if v is not None or text is None:
            return v
        t = text.strip().lower()
        if "true" in t and "false" not in t:
            return 1
        if "false" in t and "true" not in t:
            return 0
        return None
    p.parse_answer = parse_answer

if args.openai_model:
    p.OPENAI_MODEL = args.openai_model
    p.MODEL_KEY_OPENAI = args.openai_model.replace("-", "_").replace(".", "_")


class Pacer:
    """Spaces calls evenly so we stay under the per-minute limit (no 429 errors)."""
    def __init__(self, rpm):
        self.gap = 60.0 / max(rpm, 1)
        self.next = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


_oa_pacer, _cl_pacer = Pacer(args.openai_rpm), Pacer(args.claude_rpm)
_orig_call_openai, _orig_call_claude = p.call_openai, p.call_claude


def paced_openai(prompt):
    _oa_pacer.wait()
    return _orig_call_openai(prompt)


def paced_claude(prompt):
    _cl_pacer.wait()
    return _orig_call_claude(prompt)


p.call_openai, p.call_claude = paced_openai, paced_claude

_orig_process_row = p.process_row
if not use_openai:
    p.call_openai = lambda prompt: None
if not use_claude:
    p.call_claude = lambda prompt: None


def process_row(row, n_runs):
    out = _orig_process_row(row, n_runs)
    drop = []
    if not use_openai:
        drop.append(p.MODEL_KEY_OPENAI)
    if not use_claude:
        drop.append(p.MODEL_KEY_CLAUDE)
    for claim in out["truthfulness_probability"].values():
        for prompt in claim.values():
            for k in drop:
                prompt.pop(k, None)
    return out


p.process_row = process_row


# Preflight: one real call per provider, so a bad key / SDK problem stops the run
# immediately instead of silently filling the output with missing answers.
def preflight():
    test = p.SIMPLE_PROMPT_TEMPLATE.format(claim="Berlin ist die Hauptstadt von Deutschland.")
    checks = []
    if use_openai:
        checks.append(("OpenAI", lambda: p.openai_client.chat.completions.create(
            model=p.OPENAI_MODEL, messages=[{"role": "user", "content": test}],
            max_tokens=5, temperature=0).choices[0].message.content))
    if use_claude:
        checks.append(("Anthropic", lambda: p.anthropic_client.messages.create(
            model=p.ANTHROPIC_MODEL, max_tokens=5, temperature=0,
            messages=[{"role": "user", "content": test}]).content[0].text))
    for name, fn in checks:
        try:
            ans = fn()
            print(f"Preflight {name}: OK -> {ans!r}")
        except Exception as e:
            sys.exit(f"Preflight {name} FAILED: {type(e).__name__}: {e}\n"
                     "Fix this before running (check the key in .env and the SDK versions: "
                     'pip install "openai>=1.40,<2" "anthropic>=0.34,<1").')


preflight()

output = args.output or os.path.join(ROOT, "outputs", "polbix", {"claude": "output_claude_full.json",
    "openai": "output_openai_full.json", "both": "output_both.json"}[args.models])

# Rows whose answers were lost to errors (stored as None) are dropped from the
# checkpoint so this run asks them again. Genuine unparseable answers are rare.
ckpt = output + ".checkpoint"
if os.path.exists(ckpt):
    keys = ([p.MODEL_KEY_OPENAI] if use_openai else []) + ([p.MODEL_KEY_CLAUDE] if use_claude else [])
    with open(ckpt, encoding="utf-8") as f:
        rows = json.load(f)

    def complete(r):
        for claim in r["truthfulness_probability"].values():
            for prompt in claim.values():
                for k in keys:
                    v = prompt.get(k)
                    if v is None or len(v) < args.n_runs or all(x is None for x in v):
                        return False
        return True

    good = [r for r in rows if complete(r)]
    if len(good) < len(rows):
        print(f"Re-doing {len(rows) - len(good)} rows that had failed calls.")
        with open(ckpt, "w", encoding="utf-8") as f:
            json.dump(good, f, ensure_ascii=False, indent=2)

sys.argv = [sys.argv[0], "--dataset", args.dataset, "--output", output,
            "--n_sample", str(args.n_sample), "--n_runs", str(args.n_runs),
            "--seed", str(args.seed)]
print(f"Models: {args.models} | sample={args.n_sample} runs={args.n_runs} -> {output}")
p.main()
