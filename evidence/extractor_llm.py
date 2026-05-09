"""
Agentic LLM-based evidence extractor.

Two-stage pipeline (inspired by the "Read the Paper, Write the Code" agentic approach):
  Stage 1 — Embedding retrieval (from evidence/extractor.py):
             chunk reflection → embed → cosine similarity → top-k candidate sentences
  Stage 2 — LLM extraction:
             LLM selects the single best evidence span per dimension, returns structured
             Pydantic output with an explanation and confidence score.

The resulting ReflectionEvidence object can be blinded (scores/text redacted) via
to_blinded() for downstream information isolation.

Requires environment variables: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY
(only the key for the chosen model is needed).
"""
import json
import re
import time
from typing import Any

from pydantic import BaseModel, field_validator

from config import DIMENSIONS, EVIDENCE_LLM_MODEL, EVIDENCE_TOP_K, LLM_MODELS, RUBRIC
from evidence.extractor import chunk_reflection, embed_chunks, retrieve_evidence
from scoring.llm_scorer import _get_anthropic, _get_gemini, _get_openai


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class DimensionEvidence(BaseModel):
    """Structured evidence for a single rubric dimension."""

    evidence: str       # exact quoted substring copied from the reflection
    explanation: str    # 1-2 sentence rationale referencing the rubric criteria
    confidence: float   # self-reported extraction confidence (0.0–1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


class ReflectionEvidence(BaseModel):
    """Structured evidence for all three rubric dimensions of one reflection."""

    what: DimensionEvidence
    why: DimensionEvidence
    how: DimensionEvidence

    def to_blinded(self) -> dict[str, dict]:
        """Return structure-only dict with evidence text redacted.

        Useful for passing to downstream agents without leaking the raw evidence
        (information isolation principle: extractor produces templates, not values).
        """
        return {
            dim: {
                "evidence": "[REDACTED]",
                "explanation": getattr(self, dim).explanation,
                "confidence": getattr(self, dim).confidence,
            }
            for dim in DIMENSIONS
        }

    def exact_match_flags(self, reflection_text: str) -> dict[str, bool]:
        """Check that each evidence span is an exact substring of the reflection."""
        return {
            dim: bool(getattr(self, dim).evidence and getattr(self, dim).evidence in reflection_text)
            for dim in DIMENSIONS
        }

    def to_dict(self) -> dict[str, Any]:
        """Flat dict representation compatible with the embedding extractor's output format."""
        out: dict[str, Any] = {}
        for dim in DIMENSIONS:
            de: DimensionEvidence = getattr(self, dim)
            out[f"{dim}_evidence"] = de.evidence
            out[f"{dim}_explanation"] = de.explanation
            out[f"{dim}_confidence"] = de.confidence
        return out


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_dimension_prompt(
    reflection_text: str,
    dimension: str,
    candidates: list[str],
    rubric_info: dict,
) -> str:
    """Build a per-dimension extraction prompt with candidate sentences as context."""
    level_lines = "\n".join(
        f"  Score {score}: {desc}"
        for score, desc in sorted(rubric_info["levels"].items())
    )
    candidate_lines = "\n".join(
        f'{i + 1}. "{sent}"' for i, sent in enumerate(candidates)
    )
    dim_upper = dimension.upper()
    return f"""You are a pharmacy education expert extracting evidence from a student reflection.

Your task: find the SINGLE BEST piece of evidence for the {dim_upper} rubric dimension.

RUBRIC — {dim_upper} ({rubric_info['name']}):
{rubric_info['description']}
{level_lines}

CANDIDATE SENTENCES (pre-selected by semantic similarity, ranked by relevance):
{candidate_lines}

FULL REFLECTION:
\"\"\"
{reflection_text}
\"\"\"

Return ONLY valid JSON (no markdown fences):
{{
  "evidence": "<exact substring copied verbatim from the full reflection above>",
  "explanation": "<1-2 sentence rationale referencing the rubric criteria>",
  "confidence": <float between 0.0 and 1.0>
}}

IMPORTANT:
- evidence MUST be an exact substring of the full reflection — do not paraphrase or truncate.
- If no suitable evidence exists, set evidence to the most relevant candidate sentence exactly as it appears in the reflection."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, model: str) -> str:
    """Dispatch to the appropriate LLM and return the raw text response."""
    if model == "claude":
        client = _get_anthropic()
        response = client.messages.create(
            model=LLM_MODELS["claude"],
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    if model == "gpt4o":
        client = _get_openai()
        response = client.chat.completions.create(
            model=LLM_MODELS["gpt4o"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    if model == "gemini":
        genai = _get_gemini()
        gemini_model = genai.GenerativeModel(LLM_MODELS["gemini"])
        response = gemini_model.generate_content(prompt)
        return response.text

    raise ValueError(f"Unknown model '{model}'. Choose from: claude, gpt4o, gemini")


# ---------------------------------------------------------------------------
# Response parsing and validation
# ---------------------------------------------------------------------------

def _parse_and_validate(raw: str, reflection_text: str) -> DimensionEvidence:
    """Strip markdown fences, parse JSON, validate via Pydantic."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    parsed = json.loads(cleaned)
    de = DimensionEvidence(**parsed)
    # Best-effort exact-match repair: if evidence is not a substring, try stripping quotes
    if de.evidence not in reflection_text:
        stripped = de.evidence.strip('"').strip("'").strip()
        if stripped in reflection_text:
            de = DimensionEvidence(
                evidence=stripped,
                explanation=de.explanation,
                confidence=de.confidence,
            )
    return de


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_evidence_llm(
    reflection_text: str,
    model: str = EVIDENCE_LLM_MODEL,
    top_k: int = EVIDENCE_TOP_K,
    retries: int = 2,
) -> ReflectionEvidence:
    """LLM-based evidence extraction with Pydantic structured output.

    Stage 1: embedding retrieval narrows the reflection to top-k candidate sentences.
    Stage 2: LLM selects the single best evidence span per dimension with explanation.

    Args:
        reflection_text: Full text of the student reflection.
        model: One of 'claude', 'gpt4o', 'gemini'.
        top_k: Number of candidate sentences to pass to the LLM per dimension.
        retries: Extra attempts on JSON/validation failure per dimension.

    Returns:
        ReflectionEvidence with DimensionEvidence for each of WHAT, WHY, HOW.
    """
    # Stage 1 — shared embedding computation (one encode call for all dimensions)
    chunks = chunk_reflection(reflection_text)
    if not chunks:
        chunks = [reflection_text]
    chunk_embeddings = embed_chunks(chunks)

    dim_results: dict[str, DimensionEvidence] = {}
    for dim in DIMENSIONS:
        candidates_with_scores = retrieve_evidence(chunks, chunk_embeddings, dim, top_k=top_k)
        candidates = [sent for sent, _ in candidates_with_scores]

        # Stage 2 — LLM extraction with retries
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                prompt = _build_dimension_prompt(
                    reflection_text, dim, candidates, RUBRIC[dim]
                )
                raw = _call_llm(prompt, model)
                dim_results[dim] = _parse_and_validate(raw, reflection_text)
                break
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(1.0)
        else:
            raise RuntimeError(
                f"LLM evidence extraction failed for '{dim}' after {retries + 1} attempt(s). "
                f"Last error: {last_error}"
            )

    return ReflectionEvidence(
        what=dim_results["what"],
        why=dim_results["why"],
        how=dim_results["how"],
    )


def extract_evidence_batch_llm(
    texts: list[str],
    model: str = EVIDENCE_LLM_MODEL,
    top_k: int = EVIDENCE_TOP_K,
    sleep_between: float = 0.5,
) -> list[ReflectionEvidence]:
    """Score a list of reflections sequentially with a small delay between API calls."""
    results = []
    for i, text in enumerate(texts):
        results.append(extract_evidence_llm(text, model=model, top_k=top_k))
        if i < len(texts) - 1:
            time.sleep(sleep_between)
    return results
