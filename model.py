import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
from preprocess import preprocess_fn

import hyperparameters as hp
import os
from datetime import datetime
from tensorboard import ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver


def retrieve_saved_model():
    model = tf.keras.models.load_model('./trained_model/model1.h5')
    model.summary()
    return model


def train_and_test_model():
    train_data, train_labels, test_data, test_labels = preprocess_fn()
    model = get_model()
    model.summary()

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    checkpoint_path = "checkpoints" + os.sep + \
        "your_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "your_model" + \
        os.sep + timestamp + os.sep

    idx_to_class = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
                       10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
                       18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, train_data, idx_to_class),
        CustomModelSaver(checkpoint_path, hp.max_num_weights)
    ]

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=["sparse_categorical_accuracy"])

    model.fit(
        x=train_data,
        y=train_labels,
        validation_data=(test_data, test_labels),
        epochs=hp.num_epochs,
        callbacks=callback_list,
    )
    model.evaluate(
        x=test_data,
        y=test_labels,
        verbose=1
    )


def train_and_save_model():

    train_data, train_labels, test_data, test_labels = preprocess_fn()
    model = get_model()
    model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=["sparse_categorical_accuracy"])

    model.fit(
        x=train_data,
        y=train_labels,
        validation_data=(test_data, test_labels),
        epochs=hp.num_epochs,
    )

    model.evaluate(
        x=test_data,
        y=test_labels,
        verbose=1
    )

    model.save("model1.h5")
    return model


def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
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
