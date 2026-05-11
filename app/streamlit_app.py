"""
Streamlit frontend for TAD-UWPharm reflection scorer.
Professors upload a CSV of student reflections and get WHAT/WHY/HOW scores back.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Make project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Inject API keys from Streamlit secrets into environment before any imports
_KEY_NAMES = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"]
for _k in _KEY_NAMES:
    if hasattr(st, "secrets") and _k in st.secrets:
        os.environ[_k] = st.secrets[_k]

from config import DIMENSIONS, LLM_MODELS, RUBRIC
from scoring.llm_scorer import score_reflection, set_cache_path

# Use a writable temp path so the cache works on Streamlit Cloud
set_cache_path(Path("/tmp/llm_score_cache.json"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Reflection Scorer — UW Pharmacy",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model_options = [k for k in LLM_MODELS if k != "ollama"]  # hide local-only models
    model = st.selectbox(
        "Scoring model",
        options=model_options,
        index=model_options.index("claude") if "claude" in model_options else 0,
        help="Which LLM to use. Claude and GPT-4o are most reliable.",
    )

    sleep_between = st.slider(
        "Delay between API calls (seconds)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Increase if you hit rate-limit errors.",
    )

    st.divider()
    st.subheader("Rubric reference")
    for dim, info in RUBRIC.items():
        with st.expander(info["name"]):
            st.caption(info["description"])
            for score in [2, 1, 0]:
                st.write(f"**{score}** — {info['levels'][score]}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Student Reflection Scorer")
st.write(
    "Upload a CSV with student reflections. Each row is scored on three rubric "
    "dimensions — **WHAT**, **WHY**, and **HOW** — using an AI model."
)

# ── Sample template download ────────────────────────────────────────────────
sample_df = pd.DataFrame({
    "reflection_id": ["REF001", "REF002"],
    "reflection": [
        (
            "I learned how to counsel patients on warfarin therapy during my rotation. "
            "This matters because improper anticoagulation can lead to serious bleeding "
            "or clotting events. Going forward, I will always review INR values and ask "
            "about dietary changes before counseling."
        ),
        (
            "We discussed drug interactions between statins and common OTC medications. "
            "Understanding this is crucial because patients often do not mention OTC use "
            "to their pharmacist. I plan to ask every patient about OTC and supplement "
            "use as part of my standard medication review."
        ),
    ],
})

st.download_button(
    "Download CSV template",
    data=sample_df.to_csv(index=False),
    file_name="reflection_template.csv",
    mime="text/csv",
)

st.info(
    "**Required columns:** `reflection_id`, `reflection`  \n"
    "Ground-truth score columns (`what_score`, `why_score`, `how_score`) are optional."
)

# ── File upload ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload reflections CSV", type="csv")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "reflection" not in df.columns:
        st.error("CSV must contain a `reflection` column.")
        st.stop()

    if "reflection_id" not in df.columns:
        df.insert(0, "reflection_id", [f"REF{i + 1:03d}" for i in range(len(df))])

    df = df[df["reflection"].notna() & (df["reflection"].str.strip() != "")].copy()

    st.success(f"{len(df)} reflection(s) ready to score.")

    with st.expander("Preview uploaded data"):
        st.dataframe(df[["reflection_id", "reflection"]].head(10), use_container_width=True)

    # ── Score button ─────────────────────────────────────────────────────────
    if st.button("Score reflections", type="primary", use_container_width=True):
        results: list[dict] = []
        progress_bar = st.progress(0, text="Starting…")
        status_text = st.empty()
        n = len(df)

        for idx, (_, row) in enumerate(df.iterrows()):
            status_text.text(f"Scoring {idx + 1} / {n}: {row['reflection_id']}")
            try:
                result = score_reflection(row["reflection"], model=model)
            except Exception as exc:
                result = {f"{dim}_score": None for dim in DIMENSIONS}
                result.update({f"{dim}_evidence": "" for dim in DIMENSIONS})
                result.update({f"{dim}_explanation": "" for dim in DIMENSIONS})
                result["error"] = str(exc)

            result["reflection_id"] = row["reflection_id"]
            result["reflection"] = row["reflection"]
            results.append(result)

            progress_bar.progress((idx + 1) / n, text=f"Scored {idx + 1} / {n}")
            if idx < n - 1:
                time.sleep(sleep_between)

        status_text.empty()
        progress_bar.empty()
        st.session_state["results_df"] = pd.DataFrame(results)
        st.session_state["source_model"] = model
        st.success("Scoring complete!")

# ── Results ──────────────────────────────────────────────────────────────────
if "results_df" in st.session_state:
    results_df: pd.DataFrame = st.session_state["results_df"]
    source_model = st.session_state.get("source_model", "")

    st.divider()
    st.subheader(f"Results  ·  model: `{LLM_MODELS.get(source_model, source_model)}`")

    # ── Summary table ─────────────────────────────────────────────────────
    score_cols = [f"{dim}_score" for dim in DIMENSIONS]
    present_score_cols = [c for c in score_cols if c in results_df.columns]

    def _color_score(val):
        if not isinstance(val, (int, float)) or pd.isna(val):
            return "background-color: #f0f0f0; color: #999"
        if val == 2:
            return "background-color: #d4edda"
        if val == 1:
            return "background-color: #fff3cd"
        return "background-color: #f8d7da"

    display_cols = ["reflection_id"] + present_score_cols
    if "error" in results_df.columns:
        display_cols.append("error")

    styled = (
        results_df[display_cols]
        .style.map(_color_score, subset=present_score_cols)
    )
    st.dataframe(styled, use_container_width=True, height=min(400, 40 + 35 * len(results_df)))

    # ── Score distribution ─────────────────────────────────────────────────
    if present_score_cols:
        st.subheader("Score distribution")
        dist_cols = st.columns(len(present_score_cols))
        for col_ui, sc in zip(dist_cols, present_score_cols):
            with col_ui:
                counts = results_df[sc].value_counts().sort_index()
                col_ui.write(f"**{sc.replace('_score', '').upper()}**")
                col_ui.bar_chart(counts)

    # ── Per-reflection detail ─────────────────────────────────────────────
    st.subheader("Evidence & explanations")
    for _, row in results_df.iterrows():
        has_error = "error" in row and pd.notna(row.get("error"))

        scores_str = "  |  ".join(
            f"{dim.upper()}: {row.get(f'{dim}_score', '?')}"
            for dim in DIMENSIONS
        )
        label = f"{row['reflection_id']}  —  {scores_str}"
        if has_error:
            label += "  ⚠️"

        with st.expander(label):
            if has_error:
                st.error(f"Scoring error: {row['error']}")

            st.write("**Reflection:**")
            st.write(row["reflection"])
            st.divider()

            dim_cols = st.columns(3)
            for dim_col, dim in zip(dim_cols, DIMENSIONS):
                with dim_col:
                    score_val = row.get(f"{dim}_score")
                    evidence = row.get(f"{dim}_evidence", "—")
                    explanation = row.get(f"{dim}_explanation", "—")
                    exact = row.get(f"{dim}_evidence_exact_match", True)

                    st.metric(dim.upper(), score_val if score_val is not None else "—")
                    if not exact:
                        st.warning("Evidence may not be an exact quote — review manually.")
                    st.write(f"**Evidence:** {evidence}")
                    st.write(f"**Explanation:** {explanation}")

    # ── Export ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export")

    export_cols = (
        ["reflection_id", "reflection"]
        + [f"{dim}_{field}" for dim in DIMENSIONS for field in ["score", "evidence", "explanation"]]
    )
    export_cols = [c for c in export_cols if c in results_df.columns]

    st.download_button(
        "Download results CSV",
        data=results_df[export_cols].to_csv(index=False),
        file_name="scored_reflections.csv",
        mime="text/csv",
        use_container_width=True,
    )
