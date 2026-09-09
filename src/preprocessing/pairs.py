from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def _split_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def find_company_author_ids(input_csv: str) -> pd.DataFrame:
    """
    Find anonymized company author IDs by counting outbound tweets.

    In this dataset:
      inbound=True  -> customer tweet
      inbound=False -> company/support tweet
    """
    counts: dict[str, int] = {}

    for chunk in pd.read_csv(
        input_csv,
        usecols=USECOLS,
        chunksize=150_000,
        low_memory=False,
    ):
        outbound = chunk[chunk["inbound"] == False]

        for author_id, count in outbound["author_id"].value_counts().items():
            author_id = str(author_id)
            counts[author_id] = counts.get(author_id, 0) + int(count)

    return (
        pd.DataFrame(
            [
                {"author_id": author_id, "outbound_count": count}
                for author_id, count in counts.items()
            ]
        )
        .sort_values("outbound_count", ascending=False)
        .reset_index(drop=True)
    )


def build_pairs(
    input_csv: str,
    output_csv: str,
    brand_author_id: str,
    max_pairs: int = 20_000,
) -> int:
    """
    Build high-confidence customer -> selected brand response pairs.

    brand_author_id must be the anonymized author_id belonging to
    the selected support account.
    """
    rows: list[dict] = []

    for chunk in pd.read_csv(
        input_csv,
        usecols=USECOLS,
        chunksize=150_000,
        low_memory=False,
    ):
        chunk["tweet_id"] = chunk["tweet_id"].astype(str)

        inbound = chunk[
            (chunk["inbound"] == True)
            & chunk["response_tweet_id"].notna()
        ].copy()

        if inbound.empty:
            continue

        target_ids: set[str] = set()

        for value in inbound["response_tweet_id"]:
            target_ids.update(_split_ids(value))

        if not target_ids:
            continue

        outbound = chunk[
            (chunk["inbound"] == False)
            & chunk["tweet_id"].isin(target_ids)
            & (chunk["author_id"].astype(str) == str(brand_author_id))
        ].copy()

        if outbound.empty:
            continue

        out_by_id = outbound.set_index("tweet_id").to_dict("index")

        for _, customer in inbound.iterrows():

            for response_id in _split_ids(customer["response_tweet_id"]):

                if response_id not in out_by_id:
                    continue

                brand = out_by_id[response_id]

                rows.append(
                    {
                        "conversation_root_id": str(customer["tweet_id"]),
                        "customer_tweet_id": str(customer["tweet_id"]),
                        "customer_text": str(customer["text"]),
                        "brand_tweet_id": str(response_id),
                        "brand_text": str(brand["text"]),
                        "created_at": str(customer["created_at"]),
                        "brand_author_id": str(brand_author_id),
                    }
                )

                if len(rows) >= max_pairs:
                    break

            if len(rows) >= max_pairs:
                break

        if len(rows) >= max_pairs:
            break

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.drop_duplicates(
            subset=["customer_tweet_id", "brand_tweet_id"]
        )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    return len(out)


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