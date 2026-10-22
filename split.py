import sys
from pathlib import Path
from src.ft_load import ft_load


def main() -> None:
    if len(sys.argv) != 2:
        print("Error: use as python3 split.py <data.csv>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]

    df = ft_load(path)

    split_ratio = 0.8
    split_idx = int(len(df) * split_ratio)

    df_train = df.iloc[:split_idx, :].copy()
    df_test = df.iloc[split_idx:, :].copy()

    try:
        Path("data").mkdir(parents=True, exist_ok=True)
        df_train.to_csv('data/training_data.csv', index=False)
        df_test.to_csv('data/testing_data.csv', index=False)
        print("Data split completed successfully:")
        print("  - Training set: data/training_data.csv "
              f"({len(df_train)} rows)")
        print(f"  - Testing set:  data/testing_data.csv ({len(df_test)} rows)")
    except Exception as e:
        print(f"Error while saving split data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
