from preprocess import preprocess_fn
from model import YourModel
import hyperparameters as hp
import tensorflow as tf

def main():
    print("hello")
    train_data, train_labels, test_data, test_labels = preprocess_fn()

    model = YourModel()
    model(tf.keras.Input(shape=(28,28,1)))

    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
 
    model.fit(
        x=train_data,
        y=train_labels, 
        validation_data=(test_data, test_labels),
        epochs=hp.num_epochs,
    )

    model.evaluate(
        x=test_data,
        y=test_data,
        verbose=1
    )




main()
