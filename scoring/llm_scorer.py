"""
Approach (c): Prompt-based LLM rubric scorer.
Unified interface for Claude Sonnet, GPT-4o, and Llama (via HuggingFace Inference API).

Requires environment variables:  ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY
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


def _get_llama():
    global _groq_client
    if _groq_client is None:
        import openai
        _groq_client = openai.OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_client


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
# Per-model scoring functions
# ---------------------------------------------------------------------------

def score_with_claude(reflection_text: str) -> dict[str, Any]:
    client = _get_anthropic()
    response = client.messages.create(
        model=LLM_MODELS["claude"],
        max_tokens=1024,
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
        response_format={"type": "json_object"},
    )
    result = _parse_response(response.choices[0].message.content)
    result["model"] = LLM_MODELS["gpt4o"]
    return result


def score_with_llama(reflection_text: str) -> dict[str, Any]:
    client = _get_llama()
    response = client.chat.completions.create(
        model=LLM_MODELS["llama"],
        messages=[{"role": "user", "content": build_scoring_prompt(reflection_text)}],
        max_tokens=512,
    )
    result = _parse_response(response.choices[0].message.content)
    result["model"] = LLM_MODELS["llama"]
    return result


_SCORER_MAP = {
    "claude": score_with_claude,
    "gpt4o":  score_with_gpt4o,
    "llama":  score_with_llama,
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score_reflection(
    reflection_text: str,
    model: str = "claude",
    retries: int = 2,
    retry_delay: float = 1.5,
) -> dict[str, Any]:
    """Score a single reflection with the given LLM model.

    Args:
        reflection_text: Full text of the student reflection.
        model: One of 'claude', 'gpt4o', 'llama'.
        retries: Extra attempts on JSON parse failure.
        retry_delay: Seconds between retries.

    Returns:
        Dict with what_score, why_score, how_score, evidence, explanations, model name.
        Evidence exact-match flags are added automatically.
    """
    if model not in _SCORER_MAP:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(_SCORER_MAP)}")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = _SCORER_MAP[model](reflection_text)
            _add_exact_match_flags(result, reflection_text)
            return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            if attempt < retries:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"LLM scoring failed after {retries + 1} attempt(s). Last error: {last_error}"
    )


def score_dataset(
    texts: list[str],
    model: str = "claude",
    sleep_between: float = 0.5,
) -> list[dict[str, Any]]:
    """Score a list of reflections sequentially, with a small delay between API calls."""
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

def _parse_response(raw: str) -> dict[str, Any]:
    """Strip markdown fences if present, parse JSON, and validate required fields."""
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
    parsed = json.loads(raw)
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
