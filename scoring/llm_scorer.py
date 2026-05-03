"""
Approach (c): Prompt-based LLM rubric scorer.
Unified interface for Claude, GPT-4o, Groq (llama-3.3-70b), and Ollama local models.

Required environment variables (only for the model(s) you call):
    ANTHROPIC_API_KEY   — claude
    OPENAI_API_KEY      — gpt4o
    GROQ_API_KEY        — llama-3.3-70b

Load them before use:  from dotenv import load_dotenv; load_dotenv()
"""
import json
import os
import re
import time
from typing import Any

from config import LLM_MODELS, RUBRIC

# SDK clients are initialised lazily so you only need the key for the model you call
_anthropic_client = None
_openai_client = None
_groq_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        import openai
        _groq_client = openai.OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_client


_ollama_client = None


def _get_ollama():
    global _ollama_client
    if _ollama_client is None:
        import openai
        _ollama_client = openai.OpenAI(
            api_key="ollama",  # required by the openai client but ignored by Ollama
            base_url="http://localhost:11434/v1",
        )
    return _ollama_client


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_scoring_prompt(reflection_text: str) -> str:
    """Build the rubric-grounded scoring prompt for any LLM."""
    rubric_lines = []
    for dim, info in RUBRIC.items():
        rubric_lines.append(f"  {dim.upper()} — {info['name']}")
        for score, desc in sorted(info["levels"].items()):
            rubric_lines.append(f"    Score {score}: {desc}")
    rubric_text = "\n".join(rubric_lines)

    return f"""You are an expert pharmacy education instructor evaluating a student reflection.

Score the reflection on three rubric dimensions: WHAT, WHY, and HOW.
Each dimension is scored 0, 1, or 2 according to the rubric below.
Apply each score strictly based on the evidence in the reflection — do not default to the same score across dimensions or across reflections.

RUBRIC:
{rubric_text}

REFLECTION:
\"\"\"
{reflection_text}
\"\"\"

Return ONLY a valid JSON object with exactly this structure (no markdown fences):
{{
  "what_score": <0|1|2>,
  "what_evidence": "<exact quoted text from the reflection that supports this score>",
  "what_explanation": "<1-2 sentence rationale referencing the rubric>",
  "why_score": <0|1|2>,
  "why_evidence": "<exact quoted text from the reflection>",
  "why_explanation": "<1-2 sentence rationale>",
  "how_score": <0|1|2>,
  "how_evidence": "<exact quoted text from the reflection>",
  "how_explanation": "<1-2 sentence rationale>"
}}

IMPORTANT: evidence values must be exact substrings copied from the reflection above."""


# ---------------------------------------------------------------------------
# Per-provider scoring functions
# ---------------------------------------------------------------------------

def score_with_claude(reflection_text: str) -> dict[str, Any]:
    client = _get_anthropic()
    response = client.messages.create(
        model=LLM_MODELS["claude"],
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": build_scoring_prompt(reflection_text)}],
    )
    result = _parse_response(response.content[0].text)
    result["model"] = LLM_MODELS["claude"]
    return result


def score_with_gpt4o(reflection_text: str) -> dict[str, Any]:
    client = _get_openai()
    response = client.chat.completions.create(
        model=LLM_MODELS["gpt4o"],
        messages=[{"role": "user", "content": build_scoring_prompt(reflection_text)}],
        max_tokens=512,
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = _parse_response(response.choices[0].message.content)
    result["model"] = LLM_MODELS["gpt4o"]
    return result


def _make_groq_scorer(model_key: str):
    """Factory: returns a scorer function for any Groq-hosted model."""
    def _score(reflection_text: str) -> dict[str, Any]:
        client = _get_groq()
        response = client.chat.completions.create(
            model=LLM_MODELS[model_key],
            messages=[{"role": "user", "content": build_scoring_prompt(reflection_text)}],
            max_tokens=512,
            temperature=0,
        )
        result = _parse_response(response.choices[0].message.content)
        result["model"] = LLM_MODELS[model_key]
        return result
    return _score


def _make_ollama_scorer(model_key: str):
    """Factory: returns a scorer function for any Ollama-hosted model."""
    def _score(reflection_text: str) -> dict[str, Any]:
        client = _get_ollama()
        response = client.chat.completions.create(
            model=LLM_MODELS[model_key],
            messages=[{"role": "user", "content": build_scoring_prompt(reflection_text)}],
            max_tokens=512,
            temperature=0,
        )
        result = _parse_response(response.choices[0].message.content)
        result["model"] = LLM_MODELS[model_key]
        return result
    return _score


# Models routed by provider
_GROQ_MODELS   = {"llama-3.3-70b", "llama-4-scout"}
_OLLAMA_MODELS = {"ollama-qwen2.5-72b", "ollama-gemma3-12b"}

_SCORER_MAP: dict[str, Any] = {
    "claude": score_with_claude,
    "gpt4o":  score_with_gpt4o,
    **{k: _make_groq_scorer(k)   for k in _GROQ_MODELS},
    **{k: _make_ollama_scorer(k) for k in _OLLAMA_MODELS},
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score_reflection(
    reflection_text: str,
    model: str = "llama-3.3-70b",
    retries: int = 2,
    retry_delay: float = 1.5,
    rate_limit_retries: int = 8,
) -> dict[str, Any]:
    """Score a single reflection with the given LLM model.

    Args:
        reflection_text: Full text of the student reflection.
        model: One of 'claude', 'gpt4o', 'llama-3.3-70b',
               'ollama-qwen2.5-72b', 'ollama-gemma3-12b'.
        retries: Extra attempts on JSON parse failure.
        retry_delay: Seconds between retries.
        rate_limit_retries: Max retries on HTTP 429 rate-limit errors; uses
            the wait time from the error message with exponential backoff fallback.

    Returns:
        Dict with what_score, why_score, how_score, evidence, explanations, model name.
        Evidence exact-match flags are added automatically.
    """
    if model not in _SCORER_MAP:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(_SCORER_MAP)}")

    last_error: Exception | None = None
    parse_attempts = 0
    rl_attempts = 0

    while parse_attempts <= retries:
        try:
            result = _SCORER_MAP[model](reflection_text)
            _add_exact_match_flags(result, reflection_text)
            return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            parse_attempts += 1
            if parse_attempts <= retries:
                time.sleep(retry_delay)
        except Exception as e:
            if getattr(e, "status_code", None) == 429 or "ratelimit" in type(e).__name__.lower():
                rl_attempts += 1
                if rl_attempts > rate_limit_retries:
                    raise
                wait = _parse_retry_after(e) or min(2 ** rl_attempts, 300)
                mins, secs = divmod(int(wait), 60)
                wait_str = f"{mins}m{secs}s" if mins else f"{secs}s"
                print(f"  [rate limit] waiting {wait_str} (attempt {rl_attempts}/{rate_limit_retries})...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"LLM scoring failed after {parse_attempts} attempt(s). Last error: {last_error}"
    )


def score_dataset(
    texts: list[str],
    model: str = "llama-3.3-70b",
    sleep_between: float = 2.0,
) -> list[dict[str, Any]]:
    """Score a list of reflections sequentially, with a small delay between API calls.

    Note: free-tier Groq limits (e.g. 6 000 TPM for llama-3.1-8b) may require a
    larger sleep_between (~10s) to avoid sustained rate-limit pressure.
    """
    results = []
    for i, text in enumerate(texts):
        result = score_reflection(text, model=model)
        result["reflection_idx"] = i
        results.append(result)
        if i < len(texts) - 1:
            time.sleep(sleep_between)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_retry_after(exc: Exception) -> float | None:
    """Extract wait time (seconds) from a Groq/OpenAI rate-limit error message.

    Handles formats like: '880ms', '1.2s', '6m33.12s', '1h2m3s'.
    Returns None if no match found; caller should fall back to exponential backoff.
    """
    msg = str(exc)
    m = re.search(r"try again in ([\w.]+)", msg)
    if not m:
        return None
    time_str = m.group(1)
    total = 0.0
    for part in re.finditer(r"(\d+(?:\.\d+)?)(ms|h|m|s)", time_str):
        value, unit = float(part.group(1)), part.group(2)
        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        elif unit == "s":
            total += value
        elif unit == "ms":
            total += value / 1000
    return total if total > 0 else None


def _parse_response(raw: str) -> dict[str, Any]:
    """Strip markdown fences if present, parse JSON, and validate required fields."""
    # First try to extract the outermost {...} block (handles preamble/postamble text
    # and unstripped fences that small models like llama-3.1-8b sometimes emit).
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # Fall back to fence-stripping and parse the full string
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            parsed = json.loads(cleaned)
    else:
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(cleaned)
    required = [
        "what_score", "what_evidence", "what_explanation",
        "why_score",  "why_evidence",  "why_explanation",
        "how_score",  "how_evidence",  "how_explanation",
    ]
    for field in required:
        if field not in parsed:
            raise ValueError(f"LLM response missing required field: '{field}'")
    for dim in ("what", "why", "how"):
        score = parsed[f"{dim}_score"]
        if score not in (0, 1, 2):
            raise ValueError(f"Invalid {dim}_score value: {score!r} (must be 0, 1, or 2)")
    return parsed


def _add_exact_match_flags(result: dict[str, Any], original_text: str) -> None:
    """Add *_evidence_exact_match booleans to verify auditability."""
    for dim in ("what", "why", "how"):
        evidence = result.get(f"{dim}_evidence", "")
        result[f"{dim}_evidence_exact_match"] = bool(evidence and evidence in original_text)
