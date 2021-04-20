import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp

def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28,28,1)),
            Conv2D(32, kernel_size=5, activation='relu'),
            #Conv2D(32, kernel_size=5, activation='relu'),
            MaxPool2D(2, 2, padding="same"),
            #Conv2D(64, kernel_size=5, activation='relu'),
            Conv2D(64, kernel_size=5, activation='relu'),
            MaxPool2D(2, 2, padding="same"),
            Flatten(),
            Dropout(0.35),
            Dense(64, activation='relu'),
            Dropout(0.2),
            #      Dense(160, activation='relu'),
            #      Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(hp.num_classes, activation='softmax')
        ]
    )
    return model


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
            #Conv2D(32, kernel_size=5, activation='relu'),
            MaxPool2D(2, 2, padding="same"),
            #Conv2D(64, kernel_size=5, activation='relu'),
            Conv2D(64, kernel_size=5, activation='relu'),
            MaxPool2D(2, 2, padding="same"),
            Flatten(),
            Dropout(0.35),
            Dense(64, activation='relu'),
            Dropout(0.2),
            #      Dense(160, activation='relu'),
            #      Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(hp.num_classes, activation='softmax')
        ]

        # a = Conv2D(64, (3,3), strides = 2, activation='relu') #32 feature layers to start
        # b = Conv2D(128, (3,3), strides = 2, activation='relu')
        # c = Conv2D(200, (3,3), strides =2, activation = 'relu')
        # c1 = MaxPool2D((2,2))
        # d = Flatten() #make it 1D so dense layers can process
        # e = Dropout(0.2)
        # f = Dense(96, activation = 'relu')
        # g = Dropout(0.2)
        # h = Dense(32, activation='relu')
        # i = Dense(hp.num_classes, activation='softmax') #10 is number of output classes
        # self.architecture = [a,b,c,c1,d,e,f,g,h,i]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """


        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
