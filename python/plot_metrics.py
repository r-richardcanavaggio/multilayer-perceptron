import pandas
import matplotlib.pyplot as plt


def main():
    cols = pandas.read_csv('data.csv', header=None, nrows=0).columns
    df = pandas.read_csv('data.csv', header=None, usecols=cols[1:])

    df[df.columns[0]] = df[df.columns[0]].replace({'M':1}, regex=True)
    df[df.columns[0]] = df[df.columns[0]].replace({'B':0}, regex=True)
    # df = df[df[df.columns[0]]].replace({'M':'1'}, regex=True).astype(int)
    print(df)

    df_train = df.iloc[:455,:]
    df_test = df.iloc[455:,:]

    df_train.to_csv('training_data.csv', index=False)
    df_test.to_csv('testing_data.csv', index=False)


    # df_malign = df[df[df.columns[0]] == 'M']
    # df_benign = df[df[df.columns[0]] == 'B']

    # mean_malign = df_malign.mean(numeric_only=True)
    # std_malign = df_malign.std(numeric_only=True)

    # mean_benign = df_benign.mean(numeric_only=True)
    # std_benign = df_benign.std(numeric_only=True)

    # df_mean = df.mean(numeric_only=True)
    # df_std = df.std(numeric_only=True)

    # z_score = (df - df_mean) / df_std
    
    # z_score.iloc[:, 1].hist()
    # plt.show()

    # # result = pandas.DataFrame({
    # #     'mean_M': mean_malign,
    # #     'std_M': std_malign,
    # #     'mean_B': mean_benign,
    # #     'std_B': std_benign
    # # })

    # print(z_score)
    # print(result.iloc[0])
    # print()
    # print(result.iloc[6] * 100)
    # print()

    # print(result.iloc[3])


if __name__ == "__main__":
    main()