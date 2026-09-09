from __future__ import annotations

import pandas as pd


INPUT = "data/raw/twcs.csv"

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "text",
    "response_tweet_id",
]


# Step 1:
# Find customer tweets that explicitly mention @AmazonHelp
response_ids = set()
mention_count = 0

print("Scanning for @AmazonHelp mentions...")

for chunk in pd.read_csv(
    INPUT,
    usecols=USECOLS,
    chunksize=150_000,
    low_memory=False,
):
    inbound = chunk[
        (chunk["inbound"] == True)
        & chunk["text"].fillna("").str.contains(
            "@AmazonHelp",
            case=False,
            regex=False,
        )
    ]

    mention_count += len(inbound)

    for value in inbound["response_tweet_id"].dropna():
        for rid in str(value).split(","):
            rid = rid.strip()
            if rid:
                response_ids.add(rid)

print(f"AmazonHelp customer mentions: {mention_count}")
print(f"Response tweet IDs found: {len(response_ids)}")


# Step 2:
# Find who authored those response tweets
author_counts = {}
examples = {}

print("Finding the AmazonHelp response author...")

for chunk in pd.read_csv(
    INPUT,
    usecols=USECOLS,
    chunksize=150_000,
    low_memory=False,
):
    chunk["tweet_id"] = chunk["tweet_id"].astype(str)

    responses = chunk[
        (chunk["inbound"] == False)
        & chunk["tweet_id"].isin(response_ids)
    ]

    for _, row in responses.iterrows():
        author = str(row["author_id"])

        author_counts[author] = author_counts.get(author, 0) + 1

        examples.setdefault(author, [])

        if len(examples[author]) < 3:
            examples[author].append(str(row["text"]))


print("\nPossible AmazonHelp author IDs:\n")

for author, count in sorted(
    author_counts.items(),
    key=lambda x: x[1],
    reverse=True,
):
    print("=" * 70)
    print("author_id:", author)
    print("linked responses:", count)

    for text in examples[author]:
        print("  ", text)