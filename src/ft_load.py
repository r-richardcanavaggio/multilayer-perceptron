import pandas
import sys


def ft_load(path: str) -> pandas.DataFrame:
    try:
        df = pandas.read_csv(path)
        print(f"Successfully loaded data in {path}")
        return df
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
        print(f"Error: Unable to parse '{path}', file may be corrupted.",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while reading '{path}': {e}", file=sys.stderr)
        sys.exit(1)
