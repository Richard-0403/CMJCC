"""Test whether the endpoint's empty responses correlate with prompt size.

Why
---
The 12-scenario hybrid smokes fail in a pattern, not at random: final fallbacks land on
SC-D-11 and SC-D-12 -- the only two three-turn scenarios -- and rise monotonically with turn
index (1, 2, 3). Every failure is a transport error with a ZERO-length body, and the endpoint
accepts ``response_format`` on every call, so native structured output is not the problem.

That points at prompt size, but the bundles cannot confirm it: prompts are redacted from
``model_calls.jsonl`` because they contain candidate text, so the correlation is an inference.
This script measures it directly instead of leaving the decision resting on a guess.

Method
------
Send the REAL extraction prompt at increasing sizes, several times per size, straight at the
configured endpoint, and record the HTTP status, the body length and the latency of each
attempt. Sizes come from padding a realistic utterance, so the prompt shape is the one the
pipeline actually sends rather than an artificial blob. A size that fails repeatedly while
smaller ones succeed is a limit; scattered failures across all sizes are not.

Nothing is written to any experiment directory and no API key is printed.

Usage
-----
    python scripts/probe_endpoint_prompt_size.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jobrec.llm.remote_provider import (  # noqa: E402
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MODEL_ENV,
)
from jobrec.prompts import render_intent_extraction  # noqa: E402

#: Realistic filler: dialogue-shaped sentences, so growth changes SIZE and not KIND.
_FILLER = (
    "I previously mentioned that I am open to roles in data analytics and business "
    "intelligence, and I would prefer somewhere with structured mentoring. "
)

#: Approximate prompt sizes to probe, in characters.
_SIZES = (500, 1000, 2000, 4000, 8000, 16000, 32000)


def _utterance_of_size(target: int) -> str:
    base = "I want a data analyst role in Kuala Lumpur, onsite or hybrid, at least RM4000. "
    if len(base) >= target:
        return base[:target]
    pad = _FILLER * (1 + (target - len(base)) // len(_FILLER))
    return (base + pad)[:target]


def _attempt(client: httpx.Client, url: str, headers: dict, model: str,
             prompt: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    started = time.perf_counter()
    try:
        response = client.post(url, headers=headers, json=body)
    except httpx.HTTPError as exc:
        return {"ok": False, "status": None, "body_len": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": type(exc).__name__}
    latency = round((time.perf_counter() - started) * 1000, 1)
    text = response.text or ""
    content = ""
    if response.status_code == 200:
        try:
            data = response.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get(
                "content") or ""
        except Exception:  # noqa: BLE001 - a malformed 200 is itself the finding
            content = ""
    return {"ok": bool(content.strip()), "status": response.status_code,
            "body_len": len(text), "content_len": len(content),
            "latency_ms": latency, "error": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", default="artifacts/p0_2_smoke/endpoint_probe.json")
    args = parser.parse_args()

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"{API_KEY_ENV} is not set; nothing to probe.")
        return 1
    base_url = os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
    model = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Host only, never the key or the full URL with any query it may carry.
    from jobrec.evaluation.experiment_identity import endpoint_identity

    print(f"endpoint : {endpoint_identity(base_url)}")
    print(f"model    : {model}")
    print(f"repeats  : {args.repeats} per size, timeout {args.timeout}s\n")

    rows: list[dict] = []
    with httpx.Client(timeout=args.timeout) as client:
        for size in _SIZES:
            prompt = render_intent_extraction(_utterance_of_size(size))
            attempts = [_attempt(client, url, headers, model, prompt)
                        for _ in range(args.repeats)]
            ok = sum(1 for a in attempts if a["ok"])
            statuses = sorted({str(a["status"]) for a in attempts})
            errors = sorted({a["error"] for a in attempts if a["error"]})
            latencies = [a["latency_ms"] for a in attempts]
            rows.append({"target_size": size, "prompt_chars": len(prompt),
                         "ok": ok, "of": args.repeats, "statuses": statuses,
                         "errors": errors, "attempts": attempts})
            print(f"  prompt {len(prompt):>6} chars: {ok}/{args.repeats} usable"
                  f"  status={','.join(statuses)}"
                  f"  median {statistics.median(latencies):>8.0f} ms"
                  + (f"  errors={','.join(errors)}" if errors else ""))

    first_failure = next((r for r in rows if r["ok"] < r["of"]), None)
    print()
    if first_failure is None:
        print("No size failed. The smoke failures are NOT explained by prompt size; look at "
              "rate limiting or concurrency instead.")
    else:
        largest_clean = [r for r in rows if r["ok"] == r["of"]
                         and r["prompt_chars"] < first_failure["prompt_chars"]]
        ceiling = largest_clean[-1]["prompt_chars"] if largest_clean else 0
        print(f"First size to fail: {first_failure['prompt_chars']} chars "
              f"({first_failure['ok']}/{first_failure['of']} usable).")
        print(f"Largest size that never failed: {ceiling} chars.")
        print("A clean ceiling below the failing size is a limit, not bad luck.")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"endpoint": endpoint_identity(base_url), "model": model,
                               "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
