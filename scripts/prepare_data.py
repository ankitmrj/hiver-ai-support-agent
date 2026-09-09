import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.pairs import build_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--brand-author-id", required=True)
    parser.add_argument("--max-pairs", type=int, default=20_000)

    args = parser.parse_args()

    n = build_pairs(
        input_csv=args.input,
        output_csv=args.output,
        brand_author_id=args.brand_author_id,
        max_pairs=args.max_pairs,
    )

    print(f"Wrote {n} pairs to {args.output}")