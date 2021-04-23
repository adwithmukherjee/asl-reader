import numpy as np
import tensorflow as tf


def preprocess_fn():
    train_data = np.genfromtxt('./archive/sign_mnist_train/sign_mnist_train.csv',
                               delimiter=",", skip_header=1)
    test_data = np.genfromtxt('./archive/sign_mnist_test/sign_mnist_test.csv',
                              delimiter=",", skip_header=1)

    print(train_data)
    train_labels = train_data[:, 0]
    train_data = np.reshape(train_data[:, 1:], (-1, 28, 28, 1))

    test_labels = test_data[:, 0]
    test_data = np.reshape(test_data[:, 1:], (-1, 28, 28, 1))

    train_data = train_data / 255
    test_data = test_data / 255

    # TODO: preprocess, data augmentation, etc.
    return train_data, train_labels, test_data, test_labels

def alternate_preprocess_fn():
    train_data = np.genfromtxt('./archive/sign_mnist_train/sign_mnist_train.csv',
                               delimiter=",", skip_header=1)
    test_data = np.genfromtxt('./archive/sign_mnist_test/sign_mnist_test.csv',
                              delimiter=",", skip_header=1)

    train_labels = train_data[:, 0]
    train_data = np.reshape(train_data[:,1:], (-1,28,28,1))
    
    
    test_labels = test_data[:, 0]
    test_data = np.reshape(test_data[:,1:], (-1,28,28,1))
    
    train_data = train_data / 255
    test_data = test_data / 255

    no_bg_train = np.zeros(train_data.shape)
    no_bg_test = np.zeros(test_data.shape)

    for i in range(train_data.shape[0]):
        no_bg_train[i] = np.reshape(process(train_data[i]), (28,28,1))
    for i in range(test_data.shape[0]):
        no_bg_test[i] = np.reshape(process(test_data[i]), (28,28,1))
   

    #TODO: preprocess, data augmentation, etc. 

    return no_bg_train, train_labels, no_bg_test, test_labels




def process(img):
    # classifications = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    print(img.shape)
    gray = np.reshape(img, (28,28))
    gray = np.array(gray * 255, dtype = np.uint8)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    threshed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    masked = cv2.bitwise_and(gray, gray, mask=threshold)
    return masked