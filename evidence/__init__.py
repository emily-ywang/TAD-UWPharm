from evidence.extractor import extract_evidence_batch, extract_evidence_for_all_dimensions
from evidence.extractor_llm import (
    DimensionEvidence,
    ReflectionEvidence,
    extract_evidence_batch_llm,
    extract_evidence_llm,
)

__all__ = [
    # Embedding-based (fast, offline)
    "extract_evidence_for_all_dimensions",
    "extract_evidence_batch",
    # LLM-based (agentic, structured output)
    "extract_evidence_llm",
    "extract_evidence_batch_llm",
    "DimensionEvidence",
    "ReflectionEvidence",
]
