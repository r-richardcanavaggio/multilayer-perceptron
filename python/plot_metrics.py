import pandas as pd
import matplotlib as plt


def main():
    cols = pd.read_csv('data.csv', header=None, nrows=0).columns
    df = pd.read_csv('data.csv', header=None, usecols=cols[1:])

    df_malign = df[df[df.columns[0]] == 'M']
    df_benign = df[df[df.columns[0]] == 'B']

    # minima = df.min(numeric_only=True)
    # min_idx = df.idxmin(numeric_only=True)
    # first_col_values = df.loc[min_idx, df.columns[0]]
    # first_col_values.index = min_idx.index

    # maxima = df.max(numeric_only=True)
    # max_idx = df.idxmax(numeric_only=True)
    # first_col_max_values = df.loc[max_idx, df.columns[0]]
    # first_col_max_values.index = max_idx.index

    # result_maxs = pd.DataFrame({
    #     'minimum': minima,
    #     'result_min': first_col_values,
    #     'max_value': maxima,
    #     'result_max': first_col_max_values
    # })

    # print(result_maxs)

    mean_malign = df_malign.mean(numeric_only=True)
    std_malign = df_malign.std(numeric_only=True)

    mean_benign = df_benign.mean(numeric_only=True)
    std_benign = df_benign.std(numeric_only=True)

    result = pd.DataFrame({
        'mean_M': mean_malign,
        'std_M': std_malign,
        'mean_B': mean_benign,
        'std_B': std_benign
    })
    
    print(result.iloc[0])
    print()
    print(result.iloc[6])


if __name__ == "__main__":
    main()