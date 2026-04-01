# App — Teacher-Facing Web Interface (scaffold, not yet implemented)

This folder will contain the teacher-facing web application described in the project deliverable.

## Planned features

- Batch upload of synthetic reflections (CSV)
- Automated WHAT / WHY / HOW scoring via the NLP pipeline
- Evidence-grounded explanations for each score
- Rubric-aligned feedback generation
- Manual review and override for uncertain/flagged cases
- Exportable scores and comments

## Planned stack (TBD)

- **Backend**: FastAPI or Flask, consuming the modules in `scoring/`, `evidence/`, `evaluation/`
- **Frontend**: React or a lightweight template (Jinja2 + HTMX)
- **State**: SQLite or simple JSON for small-scale use

## Integration points

The app will call:
- `preprocessing/loader.py` — ingest uploaded CSV
- `scoring/llm_scorer.py` — primary scoring
- `evidence/extractor.py` — retrieve supporting sentences
- `evaluation/consistency.py` — flag low-confidence submissions for manual review
