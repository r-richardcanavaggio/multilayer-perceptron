def main():
    df = pandas.read_csv('training_data.csv')

    y_true = df.iloc[:, 0].to_numpy()
    y_true_inv = 1 - y_true
    y_true_double = np.stack((y_true, y_true_inv), axis=1)

    df = df.drop(columns=df.columns[0])

    X = df.to_numpy()

    """(Number of Inputs, Number of Perceptrons)"""
    W1 = glorot(30, 15)
    B1 = np.zeros(15)
    Z1 = np.dot(X, W1) + B1
    A1 = sigmoid(Z1)

    W2 = glorot(15, 15)
    print(f"W2 shape : {W2.shape}")
    B2 = np.zeros(15)

    W3 = glorot(15, 2)
    print(f"W3 shape : {W3.shape}")
    B3 = np.zeros(2)

    Z2 = np.dot(A1, W2) + B2
    A2 = sigmoid(Z2)
    print(f"Z2 shape : {Z2.shape}")

    Z3 = np.dot(A2, W3) + B3
    A3 = softmax(Z3)
    print(f"Z3 shape : {Z3.shape}")
    # print(Z3)

    loss = binary_cross_entropy(y_true, A3)
    print(loss)

    delta_z3 = A3 - y_true_double


    delta_z2 = (np.dot(delta_z3, W3.transpose())) * (A2 * (1 - A2)) # Matrix_Input * Matrix_Current * F' (here, Sigmoid'(Z) = Sigmoid(Z) * (1 - Sigmoid(Z)))
    d_W2 = np.dot(A1.transpose(), delta_z2) / 455
    
    
    delta_z1 = (np.dot(delta_z2, W2.transpose())) * (A1 * (1 - A1))


    d_W3 = np.dot(A2.transpose(), delta_z3) / 455
    d_W1 = np.dot(X.transpose(), delta_z1) / 455

    d_B3 = delta_z3.sum(axis=0) / 455
    d_B2 = delta_z2.sum(axis=0) / 455
    d_B1 = delta_z1.sum(axis=0) / 455

    W1 = W1 - (learning_rate * d_W1)
    W2 = W2 - (learning_rate * d_W2)
    W3 = W3 - (learning_rate * d_W3)
    B1 = B1 - (learning_rate * d_B1)
    B2 = B2 - (learning_rate * d_B2)
    B3 = B3 - (learning_rate * d_B3)