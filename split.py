import pandas


def main():
    cols = pandas.read_csv('data/data.csv', header=None, nrows=0).columns
    df = pandas.read_csv('data/data.csv', header=None, usecols=cols[1:])

    df[df.columns[0]] = df[df.columns[0]].replace({'M':1}, regex=True)
    df[df.columns[0]] = df[df.columns[0]].replace({'B':0}, regex=True)

    df_train = df.iloc[:455,:]
    df_test = df.iloc[455:,:]

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

    df_train.to_csv('data/training_data.csv', index=False)
    df_test.to_csv('data/testing_data.csv', index=False)


if __name__ == "__main__":
    main()