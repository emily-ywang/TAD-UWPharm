"""
Load and validate the reflections CSV, compute total_score,
and optionally split reflection_text into WHAT / WHY / HOW sections.
"""
import re
from pathlib import Path

import pandas as pd

from config import REQUIRED_COLUMNS, SCORE_LABELS, DIMENSIONS


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the reflections CSV and validate its schema.

    Expected columns: reflection_id, reflection_text, what_score, why_score, how_score
    Adds total_score = what_score + why_score + how_score.
    """
    df = pd.read_csv(path)
    _validate_schema(df)
    df = df.copy()
    df["total_score"] = df["what_score"] + df["why_score"] + df["how_score"]
    df["reflection_length"] = df["reflection_text"].str.split().str.len()
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for dim in DIMENSIONS:
        col = f"{dim}_score"
        invalid = df[col][~df[col].isin(SCORE_LABELS)]
        if not invalid.empty:
            raise ValueError(
                f"Column '{col}' has invalid values: {sorted(invalid.unique().tolist())}. "
                f"Expected {SCORE_LABELS}."
            )


# ---------------------------------------------------------------------------
# Optional: split reflection_text into per-section text columns
# ---------------------------------------------------------------------------

# Heuristic patterns that typically introduce each section header
_SECTION_PATTERNS = {
    "what": r"(?:what\s+(?:i\s+)?(?:did\s+(?:i\s+)?)?learn(?:ed)?|what\s*:)",
    "why":  r"(?:why\s+(?:is\s+it\s+|this\s+is\s+)?important|why\s*:)",
    "how":  r"(?:how\s+(?:will\s+(?:i\s+)?|(?:i\s+)?will\s+)?apply|how\s*:)",
}


def split_reflection_sections(text: str) -> dict[str, str]:
    """Attempt to split a reflection into WHAT, WHY, HOW sections using header patterns.

    Returns a dict with keys 'what', 'why', 'how'.
    If fewer than two headers are detected, all keys map to the full text as fallback.
    """
    text_lower = text.lower()
    positions: dict[str, int] = {}
    for section, pattern in _SECTION_PATTERNS.items():
        match = re.search(pattern, text_lower)
        if match:
            positions[section] = match.start()

    if len(positions) < 2:
        return {"what": text, "why": text, "how": text}

    ordered = sorted(positions.items(), key=lambda x: x[1])
    sections: dict[str, str] = {}
    for i, (section, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        sections[section] = text[start:end].strip()

    for dim in ("what", "why", "how"):
        sections.setdefault(dim, text)

    return sections


def add_section_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add what_text, why_text, how_text columns by splitting reflection_text.

    Useful when training dimension-specific classifiers on targeted text spans.
    """
    split = df["reflection_text"].apply(split_reflection_sections)
    df = df.copy()
    df["what_text"] = split.apply(lambda d: d["what"])
    df["why_text"] = split.apply(lambda d: d["why"])
    df["how_text"] = split.apply(lambda d: d["how"])
    return df
