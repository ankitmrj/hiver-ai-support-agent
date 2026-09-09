import argparse
from src.preprocessing.pairs import find_company_author_ids


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)

    args = p.parse_args()

    df = find_company_author_ids(args.input)

    print(df.head(50).to_string(index=False))