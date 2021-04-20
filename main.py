from preprocess import preprocess_fn
from model import YourModel, get_model
import hyperparameters as hp
import tensorflow as tf

def main():
    print("hello")
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

main()
