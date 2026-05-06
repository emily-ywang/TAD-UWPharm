"""
Main script for synthetic reflection scoring pipeline.

Loads SyntheticReflectionData_Combined.csv, runs preprocessing (EDA),
scores each reflection using Claude (primary) across all three dimensions,
cross-validates against Llama-3.3-70b, flags uncertain reflections,
and computes consistency metrics.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DIMENSIONS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from evaluation.metrics import compute_metrics
from preprocessing.eda import run_full_eda
from scoring.llm_scorer import score_reflection, set_cache_path

set_cache_path(PROCESSED_DATA_DIR / "synthetic_run" / "llm_score_cache.json")

load_dotenv()

# Paths
SYNTHETIC_CSV = RAW_DATA_DIR / "SyntheticReflectionData_Combined.csv"
SYNTHETIC_RUN_DIR = PROCESSED_DATA_DIR / "synthetic_run"
UNCERTAIN_CSV = PROCESSED_DATA_DIR / "uncertain_reflections.csv"
RESULTS_JSON = PROCESSED_DATA_DIR / "synthetic_scoring_results.json"
SCORES_CSV = PROCESSED_DATA_DIR / "doubly_validated_reflections.csv"

def main():
    # Load synthetic data
    print("Loading synthetic data...")
    df_synthetic = pd.read_csv(SYNTHETIC_CSV)
    # Rename columns to match expected
    df_synthetic = df_synthetic.rename(columns={
        "Reflection ID": "reflection_id",
        "Reflection": "reflection_text",
        "Score 1 (WHAT)": "what_score",
        "Score 2 (WHY)": "why_score",
        "Score 3 (HOW)": "how_score",
        "Overall Score": "total_score"
    })
    # Add required columns
    df_synthetic["total_score"] = df_synthetic["what_score"] + df_synthetic["why_score"] + df_synthetic["how_score"]
    df_synthetic["reflection_length"] = df_synthetic["reflection_text"].str.split().str.len()

    # Run preprocessing (EDA)
    print("Running preprocessing (EDA)...")
    SYNTHETIC_RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_full_eda(df_synthetic, save=True, output_dir=SYNTHETIC_RUN_DIR)

    # Score each synthetic reflection
    print("Scoring synthetic reflections...")
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON) as f:
                results = json.load(f)
            print(f"  Resuming — loaded {len(results)} previously scored reflections.")
        except json.JSONDecodeError:
            print("  Warning: results file is corrupted, starting fresh.")
            results = []
    else:
        results = []
    done_ids = {r["reflection_id"] for r in results}
    uncertain_results = [r for r in results if any(
        r["claude_scores"][dim] != r["llama_scores"][dim] for dim in ("what", "why", "how")
    )]

    for idx, row in df_synthetic.iterrows():
        reflection_id = row["reflection_id"]
        if reflection_id in done_ids:
            print(f"Skipping reflection {reflection_id} (already scored).")
            continue
        text = row["reflection_text"]
        true_scores = {dim: int(row[f"{dim}_score"]) for dim in DIMENSIONS}

        print(f"Scoring reflection {reflection_id}...")

        # Score all dimensions with Claude (primary), verify exact-match evidence for all dims.
        max_retries = 5
        for attempt in range(max_retries):
            try:
                claude_result = score_reflection(text, model="claude", retries=0)
                claude_scores = {
                    "what": claude_result["what_score"],
                    "why":  claude_result["why_score"],
                    "how":  claude_result["how_score"],
                }
                evidence = {
                    "what": claude_result["what_evidence"],
                    "why":  claude_result["why_evidence"],
                    "how":  claude_result["how_evidence"],
                }
                all_exact = all(claude_result[f"{dim}_evidence_exact_match"] for dim in ("what", "why", "how"))
                if all_exact:
                    break
                else:
                    print(f"  Attempt {attempt+1}: Evidence not exact match, retrying...")
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise

        # Cross-validate with Llama-3.3-70b across all dimensions.
        llama_result = score_reflection(text, model="llama-3.3-70b", retries=5)
        llama_scores = {
            "what": llama_result["what_score"],
            "why":  llama_result["why_score"],
            "how":  llama_result["how_score"],
        }

        pred_scores = dict(claude_scores)

        result = {
            "reflection_id": reflection_id,
            "reflection_text": text,
            "true_scores": true_scores,
            "pred_scores": pred_scores,
            "claude_scores": claude_scores,
            "llama_scores": llama_scores,
            "evidence": evidence,
        }
        results.append(result)
        tmp = RESULTS_JSON.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        tmp.replace(RESULTS_JSON)

        # Flag as uncertain if Claude and Llama-3.3-70b disagree on any dimension.
        disagrees = any(claude_scores[dim] != llama_scores[dim] for dim in ("what", "why", "how"))
        if disagrees:
            uncertain_results.append(result)

    # Save flat scores CSV — only reflections where Claude and Llama-3.3-70b fully agree.
    agreed_results = [
        r for r in results
        if all(r["claude_scores"][dim] == r["llama_scores"][dim] for dim in ("what", "why", "how"))
    ]
    scores_df = pd.DataFrame([
        {
            "reflection_id": r["reflection_id"],
            "reflection_text": r["reflection_text"],
            "true_what": r["true_scores"]["what"],
            "true_why": r["true_scores"]["why"],
            "true_how": r["true_scores"]["how"],
            "pred_what": r["pred_scores"]["what"],
            "pred_why": r["pred_scores"]["why"],
            "pred_how": r["pred_scores"]["how"],
            "claude_what": r["claude_scores"]["what"],
            "claude_why": r["claude_scores"]["why"],
            "claude_how": r["claude_scores"]["how"],
            "llama_what": r["llama_scores"]["what"],
            "llama_why": r["llama_scores"]["why"],
            "llama_how": r["llama_scores"]["how"],
            "evidence_what": r["evidence"]["what"],
            "evidence_why": r["evidence"]["why"],
            "evidence_how": r["evidence"]["how"],
        }
        for r in agreed_results
    ])
    scores_df.to_csv(SCORES_CSV, index=False)
    print(f"Scores CSV → {SCORES_CSV} ({len(agreed_results)}/{len(results)} reflections with full agreement)")

    # Save uncertain reflections
    uncertain_df = pd.DataFrame([
        {
            "reflection_id": r["reflection_id"],
            "reflection_text": r["reflection_text"],
            "claude_what": r["claude_scores"]["what"],
            "llama_what": r["llama_scores"]["what"],
            "claude_why": r["claude_scores"]["why"],
            "llama_why": r["llama_scores"]["why"],
            "claude_how": r["claude_scores"]["how"],
            "llama_how": r["llama_scores"]["how"],
            "evidence_what": r["evidence"]["what"],
            "evidence_why": r["evidence"]["why"],
            "evidence_how": r["evidence"]["how"]
        }
        for r in uncertain_results
    ])
    uncertain_df.to_csv(UNCERTAIN_CSV, index=False)
    print(f"Uncertain CSV → {UNCERTAIN_CSV} ({len(uncertain_results)}/{len(results)} reflections flagged)")

    # Compute Claude vs Llama-3.3-70b agreement across all dimensions.
    print("Computing agreement metrics...")
    for dim in ("what", "why", "how"):
        claude_dim = [r["claude_scores"][dim] for r in results]
        llama_dim = [r["llama_scores"][dim] for r in results]
        dim_metrics = compute_metrics(llama_dim, claude_dim)
        print(f"{dim.upper()} agreement metrics (Claude vs Llama-3.3-70b):", dim_metrics)

    exact_agreement = sum(
        1 for r in results
        if all(r["claude_scores"][dim] == r["llama_scores"][dim] for dim in ("what", "why", "how"))
    )
    print(
        f"Per-reflection exact agreement under chosen comparisons: {exact_agreement}/{len(results)} "
        f"({exact_agreement / len(results):.2%})"
    )

    print(f"Processed {len(results)} reflections, {len(uncertain_results)} uncertain.")

if __name__ == "__main__":
    main()