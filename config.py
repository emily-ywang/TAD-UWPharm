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
RAW_CSV = RAW_DATA_DIR / "reflections.csv"

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
        "description": "Evaluates how clearly and specifically the student describes what they learned.",
        "levels": {
            2: "Clear, specific description of a concrete learning experience with relevant details.",
            1: "Some description of learning but vague, incomplete, or lacking specificity.",
            0: "No meaningful description of learning, or response is off-topic.",
        },
    },
    "why": {
        "name": "WHY / Analysis",
        "prompt": "Why is this learning important for your growth or development?",
        "description": "Evaluates the depth of reflection on why the learning matters.",
        "levels": {
            2: "Clear explanation of personal significance with insight, self-awareness, and concrete examples.",
            1: "Some analysis of importance but surface-level, generic, or lacking personal connection.",
            0: "No meaningful analysis of why the learning mattered.",
        },
    },
    "how": {
        "name": "HOW / Integration",
        "prompt": "How will you apply this learning in future practice?",
        "description": "Evaluates specificity and actionability of the future application plan.",
        "levels": {
            2: "Specific, actionable, and realistic plan for applying learning in future pharmacy practice.",
            1: "Some mention of future application but vague, generic, or not clearly actionable.",
            0: "No mention of future application, or response is purely hypothetical without specifics.",
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
    "claude": "claude-sonnet-4-6",
    "gpt4o": "gpt-4o",
    "gemini": "gemini-2.5-pro",  # update if the Gemini model ID changes
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
