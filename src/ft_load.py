import pandas
import sys


def ft_load(path: str) -> pandas.DataFrame:
    try:
        df = pandas.read_csv(path)
        print(f"Successfully loaded data in {path}")
        return df
    except FileNotFoundError:
        print(f"FileNotFoundError: {path}")
        sys.exit()
    except PermissionError:
        print(f"PermissionError: denied {path}")
        sys.exit()
    except pandas.errors.ParserError:
        print(f"Error: Data Parsing Error. Data might be corrupted in {path}")
        sys.exit()
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: File could not be decoded at {path}")
        sys.exit()
    except Exception as e:
        print(f"Unexepected error while reading file {e}")
        sys.exit()
