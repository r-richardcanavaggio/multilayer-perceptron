import pandas
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Error: use as python3 split.py <data.csv>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]

    try:
        df = pandas.read_csv(path, header=None)
        if df.shape[1] < 2:
            print(f"Error: '{path}' does not contain enough columns.", file=sys.stderr)
            sys.exit(1)
        df = df.iloc[:, 1:]
        print(f"Data from '{path}' successfully loaded.")
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied for '{path}'.", file=sys.stderr)
        sys.exit(1)
    except pandas.errors.EmptyDataError:
        print(f"Error: File '{path}' is empty.", file=sys.stderr)
        sys.exit(1)
    except pandas.errors.ParserError:
        print(f"Error: Unable to parse '{path}', file may be corrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while reading '{path}': {e}", file=sys.stderr)
        sys.exit(1)

    df[df.columns[0]] = df[df.columns[0]].replace({'M': 1, 'B': 0})

    split_ratio = 0.8
    split_idx = int(len(df) * split_ratio)

    df_train = df.iloc[:split_idx, :].copy()
    df_test = df.iloc[split_idx:, :].copy()

    first_col = df_train.iloc[:, 0]
    first_col_test = df_test.iloc[:, 0]

    df_train = df_train.drop(columns=df_train.columns[0])
    df_test = df_test.drop(columns=df_test.columns[0])

    mean_train = df_train.mean()
    std_train = df_train.std()

    df_train = (df_train - mean_train) / std_train
    df_test = (df_test - mean_train) / std_train

    df_train.insert(0, first_col.name, first_col)
    df_test.insert(0, first_col_test.name, first_col_test)

    try:
        Path("data").mkdir(parents=True, exist_ok=True)
        df_train.to_csv('data/training_data.csv', index=False)
        df_test.to_csv('data/testing_data.csv', index=False)
        print("Data split completed successfully:")
        print(f"  - Training set: data/training_data.csv ({len(df_train)} rows)")
        print(f"  - Testing set:  data/testing_data.csv ({len(df_test)} rows)")
    except Exception as e:
        print(f"Error while saving split data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
