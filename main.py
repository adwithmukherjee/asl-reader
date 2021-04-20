from preprocess import preprocess_fn
from model import YourModel, get_model
import hyperparameters as hp
import tensorflow as tf
import cv2

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

