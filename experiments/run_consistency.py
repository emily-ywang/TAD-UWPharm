"""
Consistency analysis runner.

Usage (single model):
    python experiments/run_consistency.py --mode repeated --model ollama_llama2 --n-samples 5
    python experiments/run_consistency.py --mode repeated --model ollama_mistral --n-samples 5
    python experiments/run_consistency.py --mode repeated --model claude

Usage (multiple Ollama models):
    python experiments/run_consistency.py --models ollama_llama2 ollama_mistral ollama_neural_chat --n-samples 5
    python experiments/run_consistency.py --models ollama_llama2 ollama_mistral --n-samples 10

Usage (cross-model agreement across all models):
    python experiments/run_consistency.py --mode cross-model --n-samples 10
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from config import PROCESSED_DATA_DIR, RAW_CSV, LLM_MODELS
from evaluation.consistency import (
    compute_score_variance,
    cross_model_agreement,
    flag_unstable_cases,
    run_repeated_scoring,
)
from preprocessing.loader import load_dataset


def main(args: argparse.Namespace) -> None:
    df = load_dataset(args.data_path)
    texts = df["reflection_text"].tolist()
    if args.n_samples:
        texts = texts[: args.n_samples]

    # If multiple models requested, run repeated analysis for each
    if args.models and args.mode == "repeated":
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model in args.models:
            print(f"\n{'='*60}")
            print(f"Running repeated-run stability — model: {model}")
            print(f"{'='*60}")
            repeated = run_repeated_scoring(texts, model=model)
            variance_df = compute_score_variance(repeated)
            variance_df = flag_unstable_cases(variance_df)
            out_path = out_dir / f"repeated_variance_{model}_{run_ts}.csv"
            variance_df.to_csv(out_path, index=False)
            print(variance_df.to_string())
            n_unstable = variance_df["is_unstable"].sum()
            print(f"\nUnstable cases: {n_unstable}/{len(variance_df)}")
            print(f"Saved → {out_path}")
        return

    print(f"Running consistency analysis on {len(texts)} reflections.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode in ("repeated", "both"):
        print(f"\n[1] Repeated-run stability — model: {args.model}")
        repeated = run_repeated_scoring(texts, model=args.model)
        variance_df = compute_score_variance(repeated)
        variance_df = flag_unstable_cases(variance_df)
        out_path = out_dir / f"repeated_variance_{run_ts}.csv"
        variance_df.to_csv(out_path, index=False)
        print(variance_df.to_string())
        print(f"\nSaved → {out_path}")
        n_unstable = variance_df["is_unstable"].sum()
        print(f"Unstable cases: {n_unstable}/{len(variance_df)}")

    if args.mode in ("cross-model", "both"):
        print("\n[2] Cross-model agreement")
        result = cross_model_agreement(texts)
        out_path = out_dir / f"cross_model_agreement_{run_ts}.json"
        with open(out_path, "w") as f:
            json.dump(result["agreement_metrics"], f, indent=2)
        print(json.dumps(result["agreement_metrics"], indent=2))
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistency analysis for LLM scorer")
    parser.add_argument(
        "--mode",
        choices=["repeated", "cross-model", "both"],
        default="repeated",
        help="Which analysis to run",
    )
    parser.add_argument(
        "--model",
        choices=list(LLM_MODELS.keys()),
        default="claude",
        help="Model to use for repeated-run analysis (single model)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(LLM_MODELS.keys()),
        default=None,
        help="Multiple models to compare (runs repeated-run for each; overrides --model)",
    )
    parser.add_argument(
        "--data-path", default=str(RAW_CSV), help="Path to reflections CSV"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit to first N reflections (useful for quick tests)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_DATA_DIR / "consistency"),
        help="Directory to save results",
    )
    main(parser.parse_args())
