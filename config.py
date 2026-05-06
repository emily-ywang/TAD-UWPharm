"""
Central configuration for paths, model names, rubric definitions, and experiment settings.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Place your CSV here (columns: reflection_id, reflection_text, what_score, why_score, how_score)
RAW_CSV = RAW_DATA_DIR / "SyntheticReflectionData_Experiment.csv"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["reflection_id", "reflection_text", "what_score", "why_score", "how_score"]
SCORE_LABELS = [0, 1, 2]
DIMENSIONS = ["what", "why", "how"]

# ---------------------------------------------------------------------------
# Rubric definitions  (used verbatim in LLM prompts)
# ---------------------------------------------------------------------------
RUBRIC = {
    "what": {
        "name": "WHAT / Description",
        "prompt": "What did you learn?",
        "description": "Describes key learnings, activities, or concepts from the session.",
        "levels": {
            2: "Clear and specific description of key learnings, activities, or concepts.",
            1: "Description is present but vague; lacks details or specificity.",
            0: "No identifiable learning described; off-topic or overly vague.",
        },
    },
    "why": {
        "name": "WHY / Analysis",
        "prompt": "Why is it important for your growth/development?",
        "description": "Explains why the learning mattered; describes new thoughts and/or feelings; examines assumptions, beliefs, or challenges.",
        "levels": {
            2: "Clearly explains why the learning mattered and includes personal insight, supported with examples.",
            1: "States general importance but includes limited personal insight or little supporting evidence.",
            0: "No analysis or reflection; simply restates expectations or requirements.",
        },
    },
    "how": {
        "name": "HOW / Integration",
        "prompt": "How will you apply this learning in future practice (e.g. internship, rotations, or beyond)?",
        "description": "Connects learning to prior experiences or values; clearly describes plan for future professional role.",
        "levels": {
            2: "Clear, specific, and actionable future plan for applying learning.",
            1: "Future intent is mentioned but vague or not actionable.",
            0: "No clear application or plan described.",
        },
    },
}

# ---------------------------------------------------------------------------
# Language markers for EDA and feature extraction
# ---------------------------------------------------------------------------
FUTURE_MARKERS = [
    "will", "plan", "intend", "aim", "hope", "going to", "expect", "commit",
    "want to", "would like to", "shall", "anticipate", "aspire",
]
CAUSAL_MARKERS = [
    "because", "since", "therefore", "thus", "as a result", "consequently",
    "due to", "owing to", "hence", "so that", "in order to",
]
INTROSPECTIVE_MARKERS = [
    "i realize", "i understand", "i feel", "i believe", "i think", "i learned",
    "i noticed", "i recognized", "i discovered", "i became aware", "i reflected",
    "i appreciate", "i see now", "i now know",
]
SPECIFICITY_MARKERS = [
    "specifically", "for example", "for instance", "such as", "in particular",
    "namely", "including", "especially", "mg", "dose", "patient", "drug",
    "medication", "pharmacy", "clinical", "pharmacist",
]

# ---------------------------------------------------------------------------
# LLM model names
# ---------------------------------------------------------------------------
LLM_MODELS = {
    # Proprietary
    "claude-haiku":  "claude-haiku-4-5-20251001",
    "claude":        "claude-sonnet-4-6",
    "gpt4o":         "gpt-4o",
    # Groq-hosted open-source (GROQ_API_KEY)
    "llama-3.3-70b":  "llama-3.3-70b-versatile",
    "llama-4-scout":  "meta-llama/llama-4-scout-17b-16e-instruct",
    # Google Gemini (GEMINI_API_KEY)
    # "gemini-flash":   "gemini-2.5-flash",  # temporarily disabled — high demand
    # Ollama local (no API key — requires `ollama serve` on localhost:11434)
    # "ollama-qwen2.5-72b": "qwen2.5:72b",
    # "ollama-gemma3-12b":  "gemma3:12b",
    # "ollama-llama-4-maverick": "llama4:maverick",
}

# Sentence embedding model (SentenceTransformers) used for approach (b) and evidence extraction
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Consistency analysis
# ---------------------------------------------------------------------------
CONSISTENCY_N_RUNS = 5           # repeated LLM runs per reflection
CONSISTENCY_VARIANCE_THRESHOLD = 0.5  # flag if score std dev exceeds this value

# ---------------------------------------------------------------------------
# Experiment / training settings
# ---------------------------------------------------------------------------
CV_FOLDS = 5
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------
EVIDENCE_TOP_K = 3  # top-k sentences retrieved per rubric dimension
