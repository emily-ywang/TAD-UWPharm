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

sys.path.insert(0, str(Path(__file__).parent.parent))

_KEY_NAMES = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"]
for _k in _KEY_NAMES:
    if hasattr(st, "secrets") and _k in st.secrets:
        os.environ[_k] = st.secrets[_k]

from config import DIMENSIONS, LLM_MODELS, RUBRIC
from scoring.llm_scorer import score_reflection, set_cache_path

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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Fonts & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #4b2e83 0%, #6a3fa6 100%);
    border-radius: 16px;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    color: white;
}
.hero p {
    font-size: 1.05rem;
    opacity: 0.88;
    margin: 0;
    color: white;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
    color: white;
}

/* ── Score badges ── */
.score-badge {
    display: inline-block;
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 0.5rem;
}
.score-2 { background: #d1fae5; color: #065f46; border: 1.5px solid #6ee7b7; }
.score-1 { background: #fef3c7; color: #92400e; border: 1.5px solid #fcd34d; }
.score-0 { background: #fee2e2; color: #991b1b; border: 1.5px solid #fca5a5; }
.score-none { background: #f3f4f6; color: #6b7280; border: 1.5px solid #d1d5db; }

/* ── Evidence card ── */
.evidence-card {
    background: #f8f7ff;
    border-left: 4px solid #4b2e83;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 0.8rem 0;
    font-size: 0.95rem;
    color: #1f1f2e;
}
.explanation-card {
    background: #f9fafb;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.92rem;
    color: #374151;
    border: 1px solid #e5e7eb;
}

/* ── Dimension label ── */
.dim-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.3rem;
}

/* ── Reflection preview box ── */
.reflection-box {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.93rem;
    color: #374151;
    line-height: 1.65;
    margin-bottom: 1.2rem;
    max-height: 180px;
    overflow-y: auto;
}

/* ── Section header ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1f1f2e;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e5e7eb;
}

/* ── Stat card ── */
.stat-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stat-card .stat-num {
    font-size: 2rem;
    font-weight: 800;
    color: #4b2e83;
}
.stat-card .stat-label {
    font-size: 0.82rem;
    color: #6b7280;
    font-weight: 500;
    margin-top: 0.2rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #f5f3ff;
}
section[data-testid="stSidebar"] h2 {
    color: #4b2e83;
}

/* ── Buttons ── */
div.stButton > button[kind="primary"] {
    background: #4b2e83;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 1.5rem;
}
div.stButton > button[kind="primary"]:hover {
    background: #3a2269;
}

/* ── Download button ── */
div.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_options = [k for k in LLM_MODELS if k != "ollama"]
    model = st.selectbox(
        "Scoring model",
        options=model_options,
        index=model_options.index("claude") if "claude" in model_options else 0,
        help="Which LLM to use. Claude and GPT-4o are most reliable.",
    )

    sleep_between = st.slider(
        "Delay between API calls (s)",
        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
        help="Increase if you hit rate-limit errors.",
    )

    st.divider()
    st.markdown("### 📋 Rubric")
    score_colors = {2: "🟢", 1: "🟡", 0: "🔴"}
    for dim, info in RUBRIC.items():
        with st.expander(f"**{info['name']}**"):
            st.caption(info["description"])
            for score in [2, 1, 0]:
                st.write(f"{score_colors[score]} **{score}** — {info['levels'][score]}")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">UW School of Pharmacy</div>
    <h1>💊 Student Reflection Scorer</h1>
    <p>Upload a CSV of student reflections to automatically score them on the <strong>WHAT</strong>, <strong>WHY</strong>, and <strong>HOW</strong> rubric dimensions using AI.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
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

col_info, col_download = st.columns([3, 1])
with col_info:
    st.info("**Required columns:** `reflection_id`, `reflection`  \n"
            "Ground-truth score columns are optional.")
with col_download:
    st.download_button(
        "⬇️ Download template",
        data=sample_df.to_csv(index=False),
        file_name="reflection_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

uploaded = st.file_uploader("Upload reflections CSV", type="csv", label_visibility="collapsed")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "reflection" not in df.columns:
        st.error("CSV must contain a `reflection` column.")
        st.stop()

    if "reflection_id" not in df.columns:
        df.insert(0, "reflection_id", [f"REF{i + 1:03d}" for i in range(len(df))])

    df = df[df["reflection"].notna() & (df["reflection"].str.strip() != "")].copy()

    # Stats row
    st.markdown('<div class="section-header">Uploaded Data</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{len(df)}</div><div class="stat-label">Reflections</div></div>', unsafe_allow_html=True)
    with s2:
        avg_words = int(df["reflection"].str.split().str.len().mean())
        st.markdown(f'<div class="stat-card"><div class="stat-num">{avg_words}</div><div class="stat-label">Avg. words per reflection</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{LLM_MODELS.get(model, model).split("-")[0].upper()}</div><div class="stat-label">Selected model</div></div>', unsafe_allow_html=True)

    with st.expander("Preview data"):
        st.dataframe(df[["reflection_id", "reflection"]].head(10), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶ Score Reflections", type="primary", use_container_width=True):
        results: list[dict] = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        n = len(df)

        for idx, (_, row) in enumerate(df.iterrows()):
            status_text.markdown(f"⏳ Scoring **{idx + 1} / {n}** — `{row['reflection_id']}`")
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
            progress_bar.progress((idx + 1) / n)
            if idx < n - 1:
                time.sleep(sleep_between)

        status_text.empty()
        progress_bar.empty()
        st.session_state["results_df"] = pd.DataFrame(results)
        st.session_state["source_model"] = model
        st.success("✅ Scoring complete! See results below.")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "results_df" in st.session_state:
    results_df: pd.DataFrame = st.session_state["results_df"]
    source_model = st.session_state.get("source_model", "")

    score_cols = [f"{dim}_score" for dim in DIMENSIONS]
    present_score_cols = [c for c in score_cols if c in results_df.columns]

    # ── Summary table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Score Summary</div>', unsafe_allow_html=True)

    def _color_score(val):
        if not isinstance(val, (int, float)) or pd.isna(val):
            return "background-color: #f3f4f6; color: #9ca3af"
        if val == 2:
            return "background-color: #d1fae5; color: #065f46; font-weight: 700"
        if val == 1:
            return "background-color: #fef3c7; color: #92400e; font-weight: 700"
        return "background-color: #fee2e2; color: #991b1b; font-weight: 700"

    display_cols = ["reflection_id"] + present_score_cols
    if "error" in results_df.columns:
        display_cols.append("error")

    styled = results_df[display_cols].style.map(_color_score, subset=present_score_cols)
    st.dataframe(styled, use_container_width=True, height=min(420, 45 + 38 * len(results_df)))

    # ── Score distribution ─────────────────────────────────────────────────
    if present_score_cols:
        st.markdown('<div class="section-header">📈 Score Distribution</div>', unsafe_allow_html=True)
        dist_cols = st.columns(len(present_score_cols))
        for col_ui, sc in zip(dist_cols, present_score_cols):
            with col_ui:
                counts = results_df[sc].value_counts().sort_index().reindex([0, 1, 2], fill_value=0)
                st.markdown(f"**{sc.replace('_score','').upper()}**")
                st.bar_chart(counts, color="#4b2e83", height=180)

    # ── Per-reflection detail ──────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Evidence & Explanations</div>', unsafe_allow_html=True)

    for _, row in results_df.iterrows():
        has_error = "error" in row and pd.notna(row.get("error"))

        scores_str = "  ·  ".join(
            f"{dim.upper()}: {row.get(f'{dim}_score', '?')}"
            for dim in DIMENSIONS
        )
        label = f"{row['reflection_id']}  —  {scores_str}"
        if has_error:
            label += "  ⚠️"

        with st.expander(label):
            if has_error:
                st.error(f"Scoring error: {row['error']}")

            st.markdown(f'<div class="reflection-box">{row["reflection"]}</div>', unsafe_allow_html=True)

            dim_cols = st.columns(3)
            for dim_col, dim in zip(dim_cols, DIMENSIONS):
                with dim_col:
                    score_val = row.get(f"{dim}_score")
                    evidence = row.get(f"{dim}_evidence", "—")
                    explanation = row.get(f"{dim}_explanation", "—")
                    exact = row.get(f"{dim}_evidence_exact_match", True)

                    score_class = f"score-{int(score_val)}" if score_val is not None else "score-none"
                    score_display = score_val if score_val is not None else "—"

                    st.markdown(f'<div class="dim-label">{dim.upper()}</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="score-badge {score_class}">Score: {score_display}</span>', unsafe_allow_html=True)

                    if not exact:
                        st.warning("Evidence may not be exact — review manually.")

                    st.markdown("**Evidence**")
                    st.markdown(f'<div class="evidence-card">"{evidence}"</div>', unsafe_allow_html=True)
                    st.markdown("**Explanation**")
                    st.markdown(f'<div class="explanation-card">{explanation}</div>', unsafe_allow_html=True)

    # ── Export ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬇️ Export Results</div>', unsafe_allow_html=True)

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
