"""
Day 10: LLM client.
Uses Groq's OpenAI-compatible chat completions endpoint (Groq — the fast
inference company at groq.com — not to be confused with xAI's "Grok"
model; different companies, similar-sounding names). Groq hosts several
open-weight models (Llama, Mixtral, etc.) behind a fast, OpenAI-compatible
API, which is what this project actually uses.

Kept variable names generic (not "GROQ_"-specific) in a couple of spots
so this can be repointed to a different OpenAI-compatible provider later
without renaming things throughout the codebase.

IMPORTANT: LLM_MODEL below is a reasonable current default for Groq, but
model availability changes — check https://console.groq.com/docs/models
for the current list if you get a "model not found" / decommissioned error.

Day 14: Automatic retry on rate limits (429). We hit Groq's free-tier
tokens-per-minute limit repeatedly during development (every LLM call in
this pipeline — synthesis, citation verification's NLI calls are local so
unaffected, but subject extraction and answer synthesis both call this).
Retrying here, once, centrally, means every caller (ingestion, /answer,
/baseline/answer) gets this resilience automatically instead of each
needing its own retry logic — important since a 429 during a live demo
would otherwise surface as a raw error to the audience.
"""

import os
import re
import time
import requests

LLM_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
LLM_API_BASE = os.getenv("GROQ_API_BASE") or os.getenv("GROK_API_BASE", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("GROQ_MODEL") or os.getenv("GROK_MODEL", "llama-3.3-70b-versatile")

MAX_RETRIES = 3


class GrokAPIError(Exception):
    pass


def _parse_retry_after_seconds(error_text: str, default: float) -> float:
    """
    Groq's 429 error message includes a suggested wait time, e.g.
    "Please try again in 3.435s." — parse it out so we wait exactly as
    long as needed rather than guessing. Falls back to `default` if the
    message format doesn't match (e.g. a different error entirely).
    """
    match = re.search(r"try again in ([\d.]+)s", error_text)
    if match:
        try:
            return float(match.group(1)) + 0.5  # small safety margin
        except ValueError:
            pass
    return default


def call_grok(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1500,
    temperature: float = 0.2,
) -> str:
    """
    Sends a single system+user message pair to Grok, returns the raw text
    of the model's reply (no parsing — caller is responsible for that,
    since Day 10/11 expects structured JSON back).

    Automatically retries up to MAX_RETRIES times on 429 (rate limit)
    errors, waiting however long Groq's own error message says to wait.
    Other errors (bad key, bad model, etc.) fail immediately — retrying
    those would just waste time on an error that won't resolve itself.
    """
    if not LLM_API_KEY:
        raise GrokAPIError(
            "No LLM API key set. Add GROQ_API_KEY to your .env file — see .env.example."
        )

    url = f"{LLM_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_error_text = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            break  # success
        except requests.exceptions.HTTPError as e:
            last_error_text = response.text
            is_rate_limit = response.status_code == 429

            if is_rate_limit and attempt < MAX_RETRIES:
                wait_seconds = _parse_retry_after_seconds(last_error_text, default=5.0 * attempt)
                print(
                    f"[grok_client] Rate limited (attempt {attempt}/{MAX_RETRIES}), "
                    f"waiting {wait_seconds:.1f}s before retry..."
                )
                time.sleep(wait_seconds)
                continue

            raise GrokAPIError(
                f"LLM API returned an error ({response.status_code}): {last_error_text}. "
                f"If this mentions an invalid/decommissioned model, check the model list at "
                f"https://console.groq.com/docs/models and update GROQ_MODEL in .env."
            ) from e
        except requests.exceptions.RequestException as e:
            raise GrokAPIError(f"Failed to reach LLM API: {e}") from e
    else:
        # Loop exhausted all retries without breaking (all attempts were rate-limited)
        raise GrokAPIError(
            f"LLM API rate limit persisted after {MAX_RETRIES} attempts: {last_error_text}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise GrokAPIError(f"Unexpected Grok API response shape: {data}") from e
