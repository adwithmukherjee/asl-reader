from preprocess import preprocess_fn
from model import YourModel
import hyperparameters as hp
import tensorflow as tf
import cv2

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

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
