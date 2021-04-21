from preprocess import preprocess_fn
from handleResult import text2Speach
from model import train_and_save_model,  retrieve_saved_model
import hyperparameters as hp
import tensorflow as tf
import cv2
import numpy as np
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.io import imread

def main():    
    model = retrieve_saved_model()
    model.summary()

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
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            classify(img_name, model)
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

def classify(img_name, model):
    classifications = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    img = imread(img_name)
    img = rgb2grey(img)
    img = resize(img, (1,28,28,1))
    prediction = model.predict_classes(img)
    print("Classification: " + classifications[prediction[0]])
    
if __name__ == '__main__':
    main()

