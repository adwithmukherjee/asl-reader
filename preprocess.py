import numpy as np


def preprocess_fn():
    train_data = np.genfromtxt('../archive/sign_mnist_train/sign_mnist_train.csv',
                               delimiter=",", skip_header=1)
    test_data = np.genfromtxt('../archive/sign_mnist_test/sign_mnist_test.csv',
                              delimiter=",", skip_header=1)

    print(train_data)
    train_labels = train_data[:, 0]
    test_labels = test_data[:, 0]
    print(train_data[:, 0])
