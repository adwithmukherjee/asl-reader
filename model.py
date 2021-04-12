import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.learning_rate)

        self.architecture = [
            Conv2D(32, kernel_size=5, activation='relu'),
            Conv2D(32, kernel_size=5, activation='relu'),
            MaxPool2D(3, 2, padding="same"),
            Conv2D(64, kernel_size=5, activation='relu'),
            Conv2D(64, kernel_size=5, activation='relu'),
            MaxPool2D(3, 2, padding="same"),
            Flatten(),
            Dropout(0.35),
            Dense(60, activation='relu'),
            Dropout(0.35),
            #      Dense(160, activation='relu'),
            #      Dropout(0.3),
            Dense(50, activation='relu'),
            Dropout(0.35),
            Dense(hp.num_classes, activation='softmax')
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
