"""
Approach (c): Prompt-based LLM rubric scorer.
Unified interface for Claude, GPT-4o, Groq, Gemini, and Ollama local models.

Required environment variables (only for the model(s) you call):
    ANTHROPIC_API_KEY   — claude-haiku, claude
    OPENAI_API_KEY      — gpt4o
    GROQ_API_KEY        — llama-3.3-70b, llama-4-scout
    GEMINI_API_KEY      — gemini-flash

Load them before use:  from dotenv import load_dotenv; load_dotenv()
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from config import LLM_MODELS, PROCESSED_DATA_DIR, RUBRIC

# ---------------------------------------------------------------------------
# Persistent score cache  (keyed by model + sha256 of reflection text)
# ---------------------------------------------------------------------------
_CACHE_PATH = PROCESSED_DATA_DIR / "experiments" / "llm_score_cache.json"
_score_cache: dict[str, Any] = {}


def set_cache_path(path: Path | str) -> None:
    """Redirect the score cache to a different file and reload it.

    Call this once at startup in any script that needs an isolated cache
    (e.g. main.py uses synthetic_run/, experiment scripts use experiments/).
    """
    global _CACHE_PATH, _score_cache
    _CACHE_PATH = Path(path)
    _load_cache()


def _load_cache() -> None:
    global _score_cache
    if _CACHE_PATH.exists():
        try:
            _score_cache = json.loads(_CACHE_PATH.read_text())
        except Exception:
            _score_cache = {}


def _save_cache() -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_score_cache, indent=2))


def _cache_key(model: str, text: str) -> str:
    return f"{model}::{hashlib.sha256(text.encode()).hexdigest()}"


_load_cache()

# SDK clients are initialised lazily so you only need the key for the model you call
_anthropic_client = None
_openai_client = None
_groq_client = None
_gemini_client = None


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


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


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

def _make_claude_scorer(model_key: str):
    """Factory: returns a scorer function for any Anthropic-hosted model."""
    def _score(reflection_text: str) -> dict[str, Any]:
        client = _get_anthropic()
        response = client.messages.create(
            model=LLM_MODELS[model_key],
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": build_scoring_prompt(reflection_text)}],
        )
        result = _parse_response(response.content[0].text)
        result["model"] = LLM_MODELS[model_key]
        return result
    return _score


def score_with_gpt4o(reflection_text: str) -> dict[str, Any]:
    client = _get_openai()
    response = client.chat.completions.create(
        model=LLM_MODELS["gpt4o"],
        messages=[{"role": "user", "content": build_scoring_prompt(reflection_text)}],
        max_tokens=1024,
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
            max_tokens=1024,
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            print(f"  [error] Groq {model_key} returned empty content. Full response: {response}")
            raise ValueError("Empty response from Groq API")
        result = _parse_response(content)
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
            max_tokens=1024,
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            print(f"  [error] Ollama {model_key} returned empty content. Full response: {response}")
            raise ValueError("Empty response from Ollama API")
        result = _parse_response(content)
        result["model"] = LLM_MODELS[model_key]
        return result
    return _score


def _make_gemini_scorer(model_key: str):
    """Factory: returns a scorer function for any Google Gemini model."""
    def _score(reflection_text: str) -> dict[str, Any]:
        from google.genai import types
        client = _get_gemini()
        response = client.models.generate_content(
            model=LLM_MODELS[model_key],
            contents=build_scoring_prompt(reflection_text),
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=4096),
        )
        content = response.text
        if not content or not content.strip():
            print(f"  [error] Gemini {model_key} returned empty content. Full response: {response}")
            raise ValueError("Empty response from Gemini API")
        result = _parse_response(content)
        result["model"] = LLM_MODELS[model_key]
        return result
    return _score


# Models routed by provider
_GROQ_MODELS   = {"llama-3.3-70b", "llama-4-scout"}
_GEMINI_MODELS = {"gemini-flash"}
# _OLLAMA_MODELS = {"ollama-qwen2.5-72b", "ollama-gemma3-12b", "ollama-llama-4-maverick"}
_CLAUDE_MODELS = {"claude-haiku", "claude"}

_SCORER_MAP: dict[str, Any] = {
    **{k: _make_claude_scorer(k)  for k in _CLAUDE_MODELS},
    "gpt4o":        score_with_gpt4o,
    **{k: _make_groq_scorer(k)    for k in _GROQ_MODELS},
    **{k: _make_gemini_scorer(k)  for k in _GEMINI_MODELS},
    # **{k: _make_ollama_scorer(k)  for k in _OLLAMA_MODELS},
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score_reflection(
    reflection_text: str,
    model: str = "llama-3.3-70b",
    retries: int = 5,
    retry_delay: float = 1.5,
    rate_limit_retries: int = 8,
) -> dict[str, Any]:
    """Score a single reflection with the given LLM model.

    Args:
        reflection_text: Full text of the student reflection.
        model: One of 'claude-haiku', 'claude', 'gpt4o',
               'llama-3.3-70b', 'llama-4-scout', 'gemini-flash', 'ollama-qwen2.5-72b', etc.
        retries: Extra attempts on JSON parse failure (default: 5).
        retry_delay: Seconds between retries.
        rate_limit_retries: Max retries on HTTP 429 rate-limit errors; uses
            the wait time from the error message with exponential backoff fallback.

    Returns:
        Dict with what_score, why_score, how_score, evidence, explanations, model name.
        Evidence exact-match flags are added automatically.
    """
    if model not in _SCORER_MAP:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(_SCORER_MAP)}")

    key = _cache_key(model, reflection_text)
    if key in _score_cache:
        return dict(_score_cache[key])

    last_error: Exception | None = None
    parse_attempts = 0
    rl_attempts = 0

    while parse_attempts <= retries:
        try:
            result = _SCORER_MAP[model](reflection_text)
            _add_exact_match_flags(result, reflection_text)
            _score_cache[key] = result
            _save_cache()
            return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            parse_attempts += 1
            if parse_attempts <= retries:
                print(f"  [parse attempt {parse_attempts}] {e}")
                time.sleep(retry_delay)
        except Exception as e:
            is_rate_limit = getattr(e, "status_code", None) == 429 or "ratelimit" in type(e).__name__.lower()
            is_unavailable = getattr(e, "status_code", None) == 503 or "unavailable" in str(e).lower()
            if is_rate_limit or is_unavailable:
                rl_attempts += 1
                if rl_attempts > rate_limit_retries:
                    raise
                wait = _parse_retry_after(e) or (120 if is_unavailable else min(2 ** rl_attempts, 300))
                mins, secs = divmod(int(wait), 60)
                wait_str = f"{mins}m{secs}s" if mins else f"{secs}s"
                label = "unavailable" if is_unavailable else "rate limit"
                print(f"  [{label}] waiting {wait_str} (attempt {rl_attempts}/{rate_limit_retries})...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"LLM scoring failed after {parse_attempts} attempt(s). Last error: {last_error}"
    )


def score_dataset(
    texts: list[str],
    model: str = "llama-3.3-70b",
    sleep_between: float | None = None,
) -> list[dict[str, Any]]:
    """Score a list of reflections sequentially, with a small delay between API calls.

    sleep_between defaults to 120s for Gemini models (high-demand 503s) and 2s for others.
    Note: free-tier Groq limits (e.g. 6 000 TPM for llama-3.1-8b) may require a
    larger sleep_between (~10s) to avoid sustained rate-limit pressure.
    """
    if sleep_between is None:
        sleep_between = 120.0 if model in _GEMINI_MODELS else 2.0
    results = []
    for i, text in enumerate(texts):
        result = score_reflection(text, model=model)
        result["reflection_idx"] = i
        results.append(result)
        if i < len(texts) - 1:
            print(f"  [gemini] waiting {int(sleep_between)}s before next call..." if model in _GEMINI_MODELS else "", end="", flush=True)
            time.sleep(sleep_between)
            if model in _GEMINI_MODELS:
                print(" done")
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


def _sanitize_json_multiline_strings(raw: str) -> str:
    """Escape literal newlines inside JSON string values so json.loads can parse them."""
    out: list[str] = []
    in_string = False
    escape = False
    for ch in raw:
        if ch == '"' and not escape:
            in_string = not in_string
            out.append(ch)
            continue
        if ch == '\\' and not escape:
            escape = True
            out.append(ch)
            continue
        if ch == "\n" and in_string and not escape:
            out.append("\\n")
            escape = False
            continue
        if ch == "\r" and in_string and not escape:
            out.append("\\r")
            escape = False
            continue
        out.append(ch)
        escape = False
    return "".join(out)


def _parse_response_from_text(raw: str) -> dict[str, Any]:
    """Fallback extraction of required fields from malformed LLM output."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    result: dict[str, Any] = {}

    for dim in ("what", "why", "how"):
        score_key = f"{dim}_score"
        # Match "score_key": <number> or score_key: <number>
        match = re.search(r'"?' + re.escape(score_key) + r'"?\s*[:=]\s*([0-2])', cleaned)
        if not match:
            raise ValueError(f"Unable to extract field '{score_key}' from LLM response.")
        result[score_key] = int(match.group(1))

        for text_key in ("evidence", "explanation"):
            key = f"{dim}_{text_key}"
            # Look for: "key": "value" - handle escaped quotes inside values
            pattern = r'"' + re.escape(key) + r'"\s*[:=]\s*"((?:[^"\\]|\\.)*)"'
            match = re.search(pattern, cleaned, re.S)
            if match:
                value = match.group(1)
            else:
                # Try: key: "value"
                pattern = re.escape(key) + r'\s*[:=]\s*"((?:[^"\\]|\\.)*)"'
                match = re.search(pattern, cleaned, re.S)
                if not match:
                    raise ValueError(f"Unable to extract field '{key}' from LLM response.")
                value = match.group(1)
            
            if not value:
                raise ValueError(f"Empty value extracted for field '{key}'")
            
            try:
                value = bytes(value, "utf-8").decode("unicode_escape")
            except Exception:
                pass
            result[key] = value

    return result


def _parse_response(raw: str) -> dict[str, Any]:
    """Strip markdown fences if present, parse JSON, and validate required fields."""
    if not raw or not raw.strip():
        print(f"  [error] Received empty response. Raw: {repr(raw)}")
        raise ValueError("Empty response from LLM API")
    
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    candidate = raw[start : end + 1] if start != -1 and end != -1 and end > start else None
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    parsed = None
    parse_error = None
    
    for text in (candidate, cleaned):
        if text is None:
            continue
        # Try parsing as-is
        try:
            parsed = json.loads(text)
            break
        except json.JSONDecodeError as e:
            parse_error = e
            # Try with multiline escaping
            try:
                parsed = json.loads(_sanitize_json_multiline_strings(text))
                break
            except json.JSONDecodeError:
                # Try fixing trailing commas (common issue)
                try:
                    fixed = re.sub(r',\s*([}\]])', r'\1', text)
                    parsed = json.loads(fixed)
                    break
                except json.JSONDecodeError:
                    continue

    if parsed is None:
        print(f"  [debug] JSON parsing failed: {parse_error}")
        print(f"  [debug] Raw response (first 500 chars): {raw[:500]}")
        parsed = _parse_response_from_text(raw)

    required = [
        "what_score", "what_evidence", "what_explanation",
        "why_score",  "why_evidence",  "why_explanation",
        "how_score",  "how_evidence",  "how_explanation",
    ]
    for field in required:
        if field not in parsed:
            print(f"  [debug] Missing field '{field}' in parsed result: {list(parsed.keys())}")
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
