# TAD-UWPharm: Automated Scoring of Pharmacy Student Reflections

**Bella Chang · Emily Wang · Kristina Fujimoto**
University of Washington School of Pharmacy

An automated scoring pipeline for pharmacy student reflective writing. Each reflection is scored on three rubric dimensions (WHAT, WHY, HOW) derived from UW School of Pharmacy course requirements, with scores grounded in exact evidence spans from the student's text. Low-agreement reflections are flagged for instructor review.

---

## Rubric

Each reflection is scored 0–2 on three dimensions (total 0–6):

- **WHAT (Description):** Does the student clearly describe a specific learning experience, skill, or concept?
- **WHY (Analysis):** Does the student explain why this learning is personally significant?
- **HOW (Integration):** Does the student articulate a specific, actionable plan for applying the learning?

---

## Dataset

Due to student data privacy regulations and IRB constraints, all experimentation uses synthetic datasets constructed with faculty guidance and validation. Two CSVs live in `data/raw/`:

| File | Rows | Used for |
|---|---|---|
| `SyntheticReflectionData_Experiment.csv` | 671 | Experiments 1 and 2 (80/20 train/test split, seed 42) |
| `SyntheticReflectionData_Combined.csv` | 594 | Final pipeline synthetic run (`main.py`) |

---

## Codebase Structure

```
TAD-UWPharm/
├── config.py                   # Central config: paths, rubric, model names, seeds
├── main.py                     # Final pipeline synthetic run
├── requirements.txt
│
├── data/
│   ├── raw/                    # Source CSVs
│   └── processed/              # Outputs: results, caches, plots
│
├── preprocessing/
│   ├── loader.py               # Load and validate CSV
│   ├── text_features.py        # TF-IDF and language marker extraction
│   └── eda.py                  # Exploratory data analysis
│
├── scoring/
│   ├── tfidf_classifier.py     # TF-IDF + logistic regression scorer
│   ├── embedding_classifier.py # Sentence embedding + logistic regression scorer
│   └── llm_scorer.py           # LLM-based scorer with persistent cache
│
├── evidence/
│   └── extractor.py            # Evidence retrieval and exact-match verification
│
├── evaluation/
│   ├── metrics.py              # Accuracy, macro-F1, MAE, QWK
│   └── consistency.py          # Cross-model agreement analysis
│
├── experiments/
│   ├── experiment1.py          # Decomposed vs. direct scoring
│   └── experiment2.py          # Cross-approach, cross-model comparison
│
├── app/                        # Teacher-facing web interface (see app/README.md)
└── paper/                      # LaTeX source and references
```

---

## Module Guide

### `config.py`
Central configuration: file paths, the full rubric definition (used verbatim in LLM prompts), lexical marker lists, LLM model name mappings, and experiment hyperparameters (CV folds, random seed, embedding model).

### `preprocessing/loader.py`
Loads either column format (raw export headers or snake_case), validates that all scores are in {0, 1, 2}, and computes `total_score` and `reflection_length`. Optionally splits free-text reflections into WHAT/WHY/HOW sections using header-pattern heuristics.

### `preprocessing/eda.py`
Runs exploratory data analysis: score-level class distributions, word/sentence count by score quartile, lexical marker rates (specificity, causal, future, introspective), and VADER sentiment by total score. Saves plots to a configurable output directory.

### `scoring/tfidf_classifier.py`
Represents each reflection as a TF-IDF vector (up to 5,000 unigram+bigram features, English stopwords removed) and trains a balanced logistic regression per rubric target. Serves as the lexical baseline.

### `scoring/embedding_classifier.py`
Encodes reflections with `all-MiniLM-L6-v2` (384-dimensional sentence embeddings) and trains a logistic regression or calibrated LinearSVC per target. More semantically expressive than TF-IDF and substantially cheaper than LLM inference.

### `scoring/llm_scorer.py`
Prompts LLMs with the full rubric and requests structured JSON output (score + exact evidence span + explanation per dimension). Supports Claude (Haiku, Sonnet), GPT-4o, Llama 3.3-70B, Llama 4 Scout, and Gemini Flash. All models run at temperature 0. Responses are cached to disk keyed by model + SHA-256 of the reflection text to avoid redundant API calls.

Required environment variables (`.env`):
```
ANTHROPIC_API_KEY   # claude-haiku, claude
OPENAI_API_KEY      # gpt4o
GROQ_API_KEY        # llama-3.3-70b, llama-4-scout
GEMINI_API_KEY      # gemini-flash
```

### `evidence/extractor.py`
Retrieves supporting sentences per dimension using cosine similarity between `all-MiniLM-L6-v2` embeddings and fixed dimension queries. Performs post-hoc exact-match verification to guard against hallucinated evidence spans.

### `evaluation/metrics.py`
Computes accuracy, macro-F1, MAE, and quadratic weighted kappa (QWK) for ordinal 0/1/2 predictions. QWK is the primary metric as it penalises large ordinal errors most relevant to fair grading.

### `evaluation/consistency.py`
Scores reflections with two models and reports pairwise exact-agreement rate and weighted κ per dimension. Optionally evaluates each model's QWK against human ground-truth labels.

---

## Experiments

### Experiment 1 — Decomposed vs. Direct Scoring (`experiments/experiment1.py`)
Compares predicting WHAT, WHY, HOW separately and summing (decomposed) against predicting total score directly, using TF-IDF and embedding classifiers. Decomposed scoring was adopted for all subsequent work based on consistently better QWK and MAE.

### Experiment 2 — Cross-Approach and Cross-Model Evaluation (`experiments/experiment2.py`)
Compares TF-IDF, embeddings, and five LLM approaches on per-dimension scoring using decomposed scoring. Saves results as a wide CSV, a timestamped JSON snapshot, and a long-format append log. Also produces QWK bar charts, a model×dimension heatmap, and per-model confusion matrices in `data/processed/experiments/plots/`.

---

## Final Pipeline (`main.py`)

Scores every reflection in `SyntheticReflectionData_Combined.csv` using:

- **Primary grader:** Claude Sonnet across all three dimensions
- **Cross-model validator:** Llama 3.3-70B (different model family for genuine independence)
- **Evidence extraction:** Claude Sonnet returns a verbatim text span per dimension; post-hoc verification confirms it is a strict substring of the original reflection
- **Flagging:** Any reflection where the two models disagree on any dimension is exported to `uncertain_reflections.csv` for instructor review
- **Output:** `doubly_validated_reflections.csv` — reflections with full Claude/Llama agreement, including predicted scores and evidence strings

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env  # fill in API keys
```

```bash
python experiments/experiment1.py     # decomposed vs. direct
python experiments/experiment2.py     # cross-model comparison
python main.py                        # final pipeline run
```
