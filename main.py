# from preprocess import preprocess_fn
# from handleResult import text2Speach
from model import train_and_save_model,  retrieve_saved_model
import hyperparameters as hp
import tensorflow as tf
import cv2
import numpy as np


def main():

    model = retrieve_saved_model()
    print(" ")

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("ASL Letter Classification")
    img_counter = 0

    while True:
        ret, frame = cam.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break
        cv2.imshow("ASL Letter Classification", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "live_frames/frame_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            classify(img_name, model)
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()


def classify(img_name, model):
    classifications = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    img = cv2.imread(img_name)

    print(" ")
    r = cv2.selectROI(img)
    cv2.destroyWindow("ROI selector")
    print("Region of Interest: ", r)
    print(" ")

    imgCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (41, 41), 0)

    _, threshold = cv2.threshold(
        blur, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(gray, gray, mask=threshold)

    final = cv2.resize(masked, (28, 28))
    final = np.reshape(final, (1, 28, 28, 1))

    prediction = model.predict_classes(final)
    print("–––––––––––––––––––––––")
    print("CLASSIFICATION: " + classifications[prediction[0]])
    print("–––––––––––––––––––––––")
    print(" ")


if __name__ == '__main__':
    main()
