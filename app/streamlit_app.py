"""
Streamlit frontend for TAD-UWPharm reflection scorer.
Dual-model scoring, uncertainty flagging, EDA, and export.
"""
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
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
    page_title="Reflection Scorer",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, rgba(167,139,250,0.25) 0%, rgba(196,181,253,0.15) 100%);
    border-radius: 16px;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(167,139,250,0.3);
}
.hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 0.4rem 0; }
.hero p  { font-size: 1.05rem; margin: 0; opacity: 0.85; }

.score-badge {
    display: inline-block; border-radius: 8px;
    padding: 0.3rem 0.8rem; font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem;
}
.score-2    { background:#d1fae5; color:#065f46; border:1.5px solid #6ee7b7; }
.score-1    { background:#fef3c7; color:#92400e; border:1.5px solid #fcd34d; }
.score-0    { background:#fee2e2; color:#991b1b; border:1.5px solid #fca5a5; }
.score-none { background:rgba(128,128,128,0.15); border:1.5px solid rgba(128,128,128,0.3); }

.uncertain-badge {
    display:inline-block; background:#fff7ed; color:#9a3412;
    border:1.5px solid #fdba74; border-radius:6px;
    padding:0.2rem 0.7rem; font-size:0.82rem; font-weight:600;
}
.agreed-badge {
    display:inline-block; background:#f0fdf4; color:#166534;
    border:1.5px solid #86efac; border-radius:6px;
    padding:0.2rem 0.7rem; font-size:0.82rem; font-weight:600;
}

.evidence-card {
    background:rgba(167,139,250,0.1); border-left:4px solid #7c5cbf;
    border-radius:0 10px 10px 0; padding:0.9rem 1.1rem;
    margin:0.5rem 0 0.8rem 0; font-size:0.95rem;
}
.explanation-card {
    background:rgba(128,128,128,0.08); border-radius:8px;
    padding:0.7rem 1rem; font-size:0.92rem;
    border:1px solid rgba(128,128,128,0.2);
}
.dim-label {
    font-size:0.75rem; font-weight:700; letter-spacing:0.08em;
    text-transform:uppercase; opacity:0.6; margin-bottom:0.3rem;
}
.reflection-box {
    background:rgba(128,128,128,0.08); border:1px solid rgba(128,128,128,0.2);
    border-radius:10px; padding:1rem 1.2rem; font-size:0.93rem;
    line-height:1.65; margin-bottom:1.2rem; max-height:180px; overflow-y:auto;
}
.section-header {
    font-size:1.2rem; font-weight:700; margin:1.5rem 0 0.8rem 0;
    padding-bottom:0.4rem; border-bottom:2px solid rgba(128,128,128,0.2);
}
.stat-card {
    background:rgba(128,128,128,0.08); border:1px solid rgba(128,128,128,0.2);
    border-radius:12px; padding:1.1rem 1.2rem; text-align:center;
}
.stat-card .stat-num { font-size:2rem; font-weight:800; color:#7c5cbf; }
.stat-card .stat-label { font-size:0.82rem; opacity:0.65; font-weight:500; margin-top:0.2rem; }

div.stButton > button[kind="primary"] {
    background:#7c5cbf; border:none; border-radius:8px;
    font-weight:600; font-size:1rem; padding:0.6rem 1.5rem;
}
div.stButton > button[kind="primary"]:hover { background:#6a4aad; }
div.stDownloadButton > button { border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Settings")

    model_options = [k for k in LLM_MODELS if k != "ollama"]
    primary_model = st.selectbox(
        "Primary model",
        options=model_options,
        index=model_options.index("claude") if "claude" in model_options else 0,
        help="Main model used for scoring and evidence extraction.",
    )
    cv_model = st.selectbox(
        "Cross-validation model",
        options=model_options,
        index=model_options.index("llama-3.3-70b") if "llama-3.3-70b" in model_options else 1,
        help="Second model used to flag uncertain reflections.",
    )
    sleep_between = st.slider(
        "Delay between API calls (s)",
        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
    )
    max_retries = st.slider(
        "Max retries for exact-match evidence",
        min_value=1, max_value=8, value=5, step=1,
    )

    st.divider()
    st.subheader("📋 Rubric")
    for dim, info in RUBRIC.items():
        with st.expander(info["name"]):
            st.caption(info["description"])
            for score in [2, 1, 0]:
                st.write(f"**{score}** — {info['levels'][score]}")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Student Reflection Scorer</h1>
    <p>Upload a CSV of student reflections to score them on the <strong>WHAT</strong>, <strong>WHY</strong>,
    and <strong>HOW</strong> rubric dimensions using two AI models, with uncertainty flagging and analysis.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
sample_df = pd.DataFrame({
    "reflection_id": ["REF001", "REF002"],
    "reflection": [
        "I learned how to counsel patients on warfarin therapy during my rotation. "
        "This matters because improper anticoagulation can lead to serious bleeding or clotting events. "
        "Going forward, I will always review INR values and ask about dietary changes before counseling.",
        "We discussed drug interactions between statins and common OTC medications. "
        "Understanding this is crucial because patients often do not mention OTC use to their pharmacist. "
        "I plan to ask every patient about OTC and supplement use as part of my standard medication review.",
    ],
})

col_info, col_dl = st.columns([3, 1])
with col_info:
    st.info("**Required columns:** `reflection_id`, `reflection`")
with col_dl:
    st.download_button("⬇️ Download template", sample_df.to_csv(index=False),
                       "reflection_template.csv", "text/csv", use_container_width=True)

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
        df.insert(0, "reflection_id", [f"REF{i+1:03d}" for i in range(len(df))])

    df = df[df["reflection"].notna() & (df["reflection"].str.strip() != "")].copy()
    df["word_count"] = df["reflection"].str.split().str.len()

    # Stats row
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{len(df)}</div><div class="stat-label">Reflections</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{int(df["word_count"].mean())}</div><div class="stat-label">Avg words</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{primary_model.split("-")[0].upper()}</div><div class="stat-label">Primary model</div></div>', unsafe_allow_html=True)

    with st.expander("Preview data"):
        st.dataframe(df[["reflection_id", "reflection"]].head(10), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶ Score & Analyze", type="primary", use_container_width=True):
        results = []
        n = len(df)
        progress = st.progress(0)
        status = st.empty()

        # ── Stage 1: primary model ──────────────────────────────────────────
        status.markdown(f"**Stage 1 of 2** — Scoring with `{primary_model}`…")
        primary_results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            status.markdown(f"**Stage 1 of 2** — `{primary_model}` · {idx+1}/{n}: `{row['reflection_id']}`")
            result = None
            for attempt in range(max_retries):
                try:
                    r = score_reflection(row["reflection"], model=primary_model, retries=0)
                    all_exact = all(r.get(f"{dim}_evidence_exact_match", True) for dim in DIMENSIONS)
                    if all_exact or attempt == max_retries - 1:
                        result = r
                        break
                except Exception as exc:
                    if attempt == max_retries - 1:
                        result = {f"{dim}_score": None for dim in DIMENSIONS}
                        result.update({f"{dim}_evidence": "" for dim in DIMENSIONS})
                        result.update({f"{dim}_explanation": "" for dim in DIMENSIONS})
                        result["error"] = str(exc)
                if result is None and attempt < max_retries - 1:
                    time.sleep(1.0)
            primary_results.append(result or {})
            progress.progress((idx + 1) / (n * 2))
            if idx < n - 1:
                time.sleep(sleep_between)

        # ── Stage 2: cross-validation model ────────────────────────────────
        status.markdown(f"**Stage 2 of 2** — Cross-validating with `{cv_model}`…")
        cv_results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            status.markdown(f"**Stage 2 of 2** — `{cv_model}` · {idx+1}/{n}: `{row['reflection_id']}`")
            try:
                r = score_reflection(row["reflection"], model=cv_model, retries=3)
            except Exception as exc:
                r = {f"{dim}_score": None for dim in DIMENSIONS}
                r["error"] = str(exc)
            cv_results.append(r)
            progress.progress(0.5 + (idx + 1) / (n * 2))
            if idx < n - 1:
                time.sleep(sleep_between)

        progress.empty()
        status.empty()

        # ── Combine results ─────────────────────────────────────────────────
        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            p = primary_results[i]
            c = cv_results[i]

            def _score(res, dim):
                v = res.get(f"{dim}_score")
                return None if v is None or (isinstance(v, float) and pd.isna(v)) else int(v)

            uncertain = any(
                _score(p, dim) != _score(c, dim)
                for dim in DIMENSIONS
                if _score(p, dim) is not None and _score(c, dim) is not None
            )

            rows.append({
                "reflection_id":   row["reflection_id"],
                "reflection":      row["reflection"],
                "word_count":      row["word_count"],
                "uncertain":       uncertain,
                **{f"{dim}_score":        _score(p, dim) for dim in DIMENSIONS},
                **{f"{dim}_score_cv":     _score(c, dim) for dim in DIMENSIONS},
                **{f"{dim}_evidence":     p.get(f"{dim}_evidence", "") for dim in DIMENSIONS},
                **{f"{dim}_explanation":  p.get(f"{dim}_explanation", "") for dim in DIMENSIONS},
                **{f"{dim}_exact_match":  p.get(f"{dim}_evidence_exact_match", True) for dim in DIMENSIONS},
                "primary_error":   p.get("error", ""),
                "cv_error":        c.get("error", ""),
            })

        results_df = pd.DataFrame(rows)
        results_df["total_score"] = results_df[[f"{d}_score" for d in DIMENSIONS]].sum(axis=1)
        st.session_state["results_df"] = results_df
        st.session_state["primary_model"] = primary_model
        st.session_state["cv_model"] = cv_model
        st.success("✅ Scoring complete!")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "results_df" in st.session_state:
    results_df: pd.DataFrame = st.session_state["results_df"]
    pm = st.session_state.get("primary_model", "")
    cvm = st.session_state.get("cv_model", "")

    n_uncertain = int(results_df["uncertain"].sum())
    n_agreed = len(results_df) - n_uncertain

    tab_scores, tab_eda, tab_evidence, tab_export = st.tabs(
        ["📋 Scores", "📈 Analysis", "🔍 Evidence", "⬇️ Export"]
    )

    # ── Tab 1: Scores ───────────────────────────────────────────────────────
    with tab_scores:
        st.markdown('<div class="section-header">Score Summary</div>', unsafe_allow_html=True)

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{len(results_df)}</div><div class="stat-label">Total reflections</div></div>', unsafe_allow_html=True)
        with a2:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{n_agreed}</div><div class="stat-label">Models agreed</div></div>', unsafe_allow_html=True)
        with a3:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{n_uncertain}</div><div class="stat-label">Uncertain (review)</div></div>', unsafe_allow_html=True)
        with a4:
            pct = f"{100*n_agreed/len(results_df):.0f}%"
            st.markdown(f'<div class="stat-card"><div class="stat-num">{pct}</div><div class="stat-label">Agreement rate</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        def _color(val):
            if not isinstance(val, (int, float)) or pd.isna(val):
                return "background-color:rgba(128,128,128,0.1)"
            if val == 2: return "background-color:#d1fae5; color:#065f46; font-weight:700"
            if val == 1: return "background-color:#fef3c7; color:#92400e; font-weight:700"
            return "background-color:#fee2e2; color:#991b1b; font-weight:700"

        def _flag(val):
            return "background-color:#fff7ed; color:#9a3412; font-weight:600" if val else ""

        score_cols = [f"{d}_score" for d in DIMENSIONS] + [f"{d}_score_cv" for d in DIMENSIONS]
        display_cols = ["reflection_id", "uncertain"] + score_cols + ["total_score"]
        display_cols = [c for c in display_cols if c in results_df.columns]

        rename_map = {
            **{f"{d}_score": f"{d.upper()} ({pm[:5]})" for d in DIMENSIONS},
            **{f"{d}_score_cv": f"{d.upper()} ({cvm[:5]})" for d in DIMENSIONS},
        }
        renamed_score_cols = [rename_map.get(c, c) for c in display_cols if "_score" in c]

        styled = (
            results_df[display_cols]
            .rename(columns=rename_map)
            .style
            .map(_color, subset=renamed_score_cols)
            .map(_flag, subset=["uncertain"])
        )
        st.dataframe(styled, use_container_width=True, height=min(450, 45 + 38*len(results_df)))

        if n_uncertain > 0:
            st.warning(f"⚠️ {n_uncertain} reflection(s) flagged as uncertain — the two models disagreed on at least one dimension. These should be reviewed manually.")

    # ── Tab 2: Analysis (EDA) ───────────────────────────────────────────────
    with tab_eda:
        st.markdown('<div class="section-header">Score Distributions</div>', unsafe_allow_html=True)
        dist_cols = st.columns(3)
        for col_ui, dim in zip(dist_cols, DIMENSIONS):
            with col_ui:
                counts = results_df[f"{dim}_score"].value_counts().sort_index().reindex([0,1,2], fill_value=0)
                st.markdown(f"**{dim.upper()}**")
                st.bar_chart(counts, color="#7c5cbf", height=180)

        st.markdown('<div class="section-header">Word Count vs Total Score</div>', unsafe_allow_html=True)
        valid = results_df[results_df["total_score"].notna()].copy()
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            groups = valid.groupby("total_score")["word_count"]
            labels = sorted(valid["total_score"].dropna().unique())
            ax.boxplot([groups.get_group(s).values for s in labels], labels=[int(s) for s in labels])
            ax.set_xlabel("Total Score")
            ax.set_ylabel("Word Count")
            ax.set_title("Word Count by Total Score")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown('<div class="section-header">Model Agreement by Dimension</div>', unsafe_allow_html=True)
        agree_data = {}
        for dim in DIMENSIONS:
            p_col, c_col = f"{dim}_score", f"{dim}_score_cv"
            if p_col in results_df.columns and c_col in results_df.columns:
                valid_rows = results_df[[p_col, c_col]].dropna()
                if not valid_rows.empty:
                    agree_data[dim.upper()] = (valid_rows[p_col] == valid_rows[c_col]).mean() * 100
        if agree_data:
            agree_df = pd.DataFrame.from_dict(agree_data, orient="index", columns=["Agreement %"])
            st.bar_chart(agree_df, color="#7c5cbf", height=220)

        st.markdown('<div class="section-header">Uncertain Reflections</div>', unsafe_allow_html=True)
        if n_uncertain > 0:
            uncertain_df = results_df[results_df["uncertain"]][
                ["reflection_id"] + [f"{d}_score" for d in DIMENSIONS] + [f"{d}_score_cv" for d in DIMENSIONS]
            ]
            st.dataframe(uncertain_df.rename(columns={
                **{f"{d}_score": f"{d.upper()} ({pm[:5]})" for d in DIMENSIONS},
                **{f"{d}_score_cv": f"{d.upper()} ({cvm[:5]})" for d in DIMENSIONS},
            }), use_container_width=True)
        else:
            st.success("Both models agreed on all reflections — no manual review needed.")

    # ── Tab 3: Evidence ─────────────────────────────────────────────────────
    with tab_evidence:
        st.markdown('<div class="section-header">Evidence & Explanations</div>', unsafe_allow_html=True)
        filter_uncertain = st.checkbox("Show uncertain reflections only")
        view_df = results_df[results_df["uncertain"]] if filter_uncertain else results_df

        for _, row in view_df.iterrows():
            scores_str = "  ·  ".join(
                f"{dim.upper()}: {row.get(f'{dim}_score', '?')}"
                for dim in DIMENSIONS
            )
            flag = " ⚠️ uncertain" if row.get("uncertain") else ""
            with st.expander(f"{row['reflection_id']}  —  {scores_str}{flag}"):
                if row.get("uncertain"):
                    st.warning("Models disagreed — review evidence carefully.")
                st.markdown(f'<div class="reflection-box">{row["reflection"]}</div>', unsafe_allow_html=True)
                dim_cols = st.columns(3)
                for dim_col, dim in zip(dim_cols, DIMENSIONS):
                    with dim_col:
                        p_score = row.get(f"{dim}_score")
                        c_score = row.get(f"{dim}_score_cv")
                        exact = row.get(f"{dim}_exact_match", True)
                        valid = p_score is not None and not (isinstance(p_score, float) and pd.isna(p_score))
                        score_class = f"score-{int(p_score)}" if valid else "score-none"
                        score_display = int(p_score) if valid else "—"
                        cv_display = int(c_score) if (c_score is not None and not (isinstance(c_score, float) and pd.isna(c_score))) else "—"

                        st.markdown(f'<div class="dim-label">{dim.upper()}</div>', unsafe_allow_html=True)
                        st.markdown(f'<span class="score-badge {score_class}">Score: {score_display}</span>', unsafe_allow_html=True)
                        st.caption(f"Cross-val ({cvm[:8]}): {cv_display}")
                        if not exact:
                            st.warning("Evidence may not be exact — review.")
                        st.markdown("**Evidence**")
                        st.markdown(f'<div class="evidence-card">"{row.get(f"{dim}_evidence", "—")}"</div>', unsafe_allow_html=True)
                        st.markdown("**Explanation**")
                        st.markdown(f'<div class="explanation-card">{row.get(f"{dim}_explanation", "—")}</div>', unsafe_allow_html=True)

    # ── Tab 4: Export ───────────────────────────────────────────────────────
    with tab_export:
        st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)

        export_cols = (
            ["reflection_id", "reflection", "uncertain", "total_score"]
            + [f"{d}_score" for d in DIMENSIONS]
            + [f"{d}_score_cv" for d in DIMENSIONS]
            + [f"{d}_evidence" for d in DIMENSIONS]
            + [f"{d}_explanation" for d in DIMENSIONS]
        )
        export_cols = [c for c in export_cols if c in results_df.columns]
        export_df = results_df[export_cols].rename(columns={
            **{f"{d}_score": f"{d}_{pm[:8]}" for d in DIMENSIONS},
            **{f"{d}_score_cv": f"{d}_{cvm[:8]}" for d in DIMENSIONS},
        })

        st.dataframe(export_df.head(5), use_container_width=True)
        st.caption(f"Showing first 5 of {len(export_df)} rows.")

        st.download_button(
            "⬇️ Download full results CSV",
            data=export_df.to_csv(index=False),
            file_name="scored_reflections.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if n_uncertain > 0:
            uncertain_export = results_df[results_df["uncertain"]][export_cols]
            st.download_button(
                "⬇️ Download uncertain reflections only",
                data=uncertain_export.to_csv(index=False),
                file_name="uncertain_reflections.csv",
                mime="text/csv",
                use_container_width=True,
            )
