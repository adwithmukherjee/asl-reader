import numpy as np
import pandas as pd

def preprocess_fn():
    train_data = np.genfromtxt('./archive/sign_mnist_train/sign_mnist_train.csv',
                               delimiter=",", skip_header=1)
    test_data = np.genfromtxt('./archive/sign_mnist_test/sign_mnist_test.csv',
                              delimiter=",", skip_header=1)

    print(train_data)
    train_labels = train_data[:, 0]
    train_data = np.reshape(train_data[:,1:], (-1,28,28,1))
    
    
    test_labels = test_data[:, 0]
    test_data = np.reshape(test_data[:,1:], (-1,28,28,1))
    
    train_data = train_data / 255
    test_data = test_data / 255
    #TODO: preprocess, data augmentation, etc. 

    return train_data, train_labels, test_data, test_labels