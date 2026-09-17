import pandas
import matplotlib.pyplot as plt


def main():
    cols = pandas.read_csv('data.csv', header=None, nrows=0).columns
    df = pandas.read_csv('data.csv', header=None, usecols=cols[1:])

    df_malign = df[df[df.columns[0]] == 'M']
    df_benign = df[df[df.columns[0]] == 'B']

    radius_malign = df_malign.iloc[:, 1]
    radius_benign = df_benign.iloc[:, 1]

    # plt.boxplot([radius_benign, radius_malign])
    radius_malign.hist()
    plt.show()

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

    z_score = (df_malign - mean_malign) / std_malign

    
    z_score.iloc[:, 1].hist()
    plt.show()

    result = pandas.DataFrame({
        'mean_M': mean_malign,
        'std_M': std_malign,
        'mean_B': mean_benign,
        'std_B': std_benign
    })

    print(z_score)
    print(result.iloc[0])
    print()
    print(result.iloc[6] * 100)
    print()

    print(result.iloc[3])


if __name__ == "__main__":
    main()