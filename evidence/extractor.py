"""
Evidence extraction: retrieve supporting sentences per rubric dimension
using sentence embeddings and cosine similarity, then verify exact matches.
"""
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL, EVIDENCE_TOP_K

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Run: python -m spacy download en_core_web_sm")

_encoder = SentenceTransformer(EMBEDDING_MODEL)

# Semantic queries used to retrieve dimension-relevant evidence sentences
_DIMENSION_QUERIES = {
    "what": "what specific knowledge, skill, or concept the student learned",
    "why":  "why this learning matters personally, its significance, insight, and importance",
    "how":  "specific actionable plan for applying this learning in future pharmacy practice",
}


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def chunk_reflection(text: str) -> list[str]:
    """Split a reflection into individual sentences (non-empty only)."""
    doc = _nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Embed a list of sentences with the shared SentenceTransformer."""
    return _encoder.encode(chunks, show_progress_bar=False)


def retrieve_evidence(
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    dimension: str,
    top_k: int = EVIDENCE_TOP_K,
) -> list[tuple[str, float]]:
    """Return the top-k most relevant sentences for a rubric dimension.

    Returns:
        List of (sentence, cosine_similarity) sorted by descending relevance.
    """
    if dimension not in _DIMENSION_QUERIES:
        raise ValueError(f"Unknown dimension '{dimension}'. Choose from {list(_DIMENSION_QUERIES)}")
    query_emb = _encoder.encode([_DIMENSION_QUERIES[dimension]])
    sims = cosine_similarity(query_emb, chunk_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def verify_exact_match(evidence: str, original_text: str) -> bool:
    """Check that evidence is an exact substring of the original reflection text."""
    return evidence.strip() in original_text


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def extract_evidence_for_all_dimensions(
    reflection_text: str,
    top_k: int = EVIDENCE_TOP_K,
) -> dict[str, list[dict]]:
    """Extract top-k evidence sentences for WHAT, WHY, and HOW from one reflection.

    Returns:
        {
          "what": [{"sentence": ..., "similarity": ..., "is_exact_match": ...}, ...],
          "why":  [...],
          "how":  [...],
        }
    """
    chunks = chunk_reflection(reflection_text)
    if not chunks:
        return {"what": [], "why": [], "how": []}

    embeddings = embed_chunks(chunks)
    results: dict[str, list[dict]] = {}
    for dim in ("what", "why", "how"):
        candidates = retrieve_evidence(chunks, embeddings, dim, top_k=top_k)
        results[dim] = [
            {
                "sentence":      sent,
                "similarity":    sim,
                "is_exact_match": verify_exact_match(sent, reflection_text),
            }
            for sent, sim in candidates
        ]
    return results


def extract_evidence_batch(
    texts: list[str],
    top_k: int = EVIDENCE_TOP_K,
) -> list[dict[str, list[dict]]]:
    """Run extract_evidence_for_all_dimensions for a list of reflections."""
    return [extract_evidence_for_all_dimensions(t, top_k=top_k) for t in texts]
