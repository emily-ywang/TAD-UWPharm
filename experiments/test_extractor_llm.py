"""
Quick smoke test for the LLM-based evidence extractor.

Usage:
    python experiments/test_extractor_llm.py
    python experiments/test_extractor_llm.py --model gpt4o
    python experiments/test_extractor_llm.py --model claude --data-path data/raw/reflections.csv --n 3
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence import extract_evidence_llm, extract_evidence_batch_llm

# ---------------------------------------------------------------------------
# Sample reflections (used when no --data-path is given)
# ---------------------------------------------------------------------------
SAMPLE_REFLECTIONS = [
    (
        "Warfarin counseling",
        "I learned how to counsel patients on warfarin therapy during my rotation. "
        "This matters because improper anticoagulation can lead to serious bleeding or clotting events. "
        "Going forward, I will always review INR values and ask about dietary changes before counseling.",
    ),
    (
        "Drug interaction awareness",
        "During my clinical rotation I discovered that many common OTC medications interact with statins. "
        "Understanding this is crucial because patients often don't mention OTC use to their pharmacist. "
        "I plan to ask every patient about OTC and supplement use as part of my standard medication review.",
    ),
]


def print_result(label: str, text: str, result) -> None:
    flags = result.exact_match_flags(text)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for dim in ("what", "why", "how"):
        de = getattr(result, dim)
        match = "✓" if flags[dim] else "✗ NO EXACT MATCH"
        print(f"\n  [{dim.upper()}] {match}")
        print(f"  Evidence   : {de.evidence}")
        print(f"  Explanation: {de.explanation}")
        print(f"  Confidence : {de.confidence:.2f}")
    print()


def main(args: argparse.Namespace) -> None:
    if args.data_path:
        import pandas as pd
        df = pd.read_csv(args.data_path)
        texts = df["reflection_text"].head(args.n).tolist()
        ids = df["reflection_id"].head(args.n).tolist()

        print(f"\nRunning batch extraction on {len(texts)} reflection(s) from {args.data_path}...")
        results = extract_evidence_batch_llm(texts, model=args.model)
        for rid, text, result in zip(ids, texts, results):
            print_result(f"reflection_id={rid}", text, result)
    else:
        print(f"\nRunning on {len(SAMPLE_REFLECTIONS)} built-in sample reflection(s)...")
        for label, text in SAMPLE_REFLECTIONS:
            result = extract_evidence_llm(text, model=args.model)
            print_result(label, text, result)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for LLM evidence extractor")
    parser.add_argument(
        "--model", default="gpt4o", choices=["claude", "gpt4o", "gemini"],
        help="LLM to use for extraction (default: claude)",
    )
    parser.add_argument(
        "--data-path", default=None,
        help="Path to reflections CSV; if omitted, uses built-in samples",
    )
    parser.add_argument(
        "--n", type=int, default=3,
        help="Number of reflections to test from the CSV (default: 3)",
    )
    main(parser.parse_args())
