# facial_recognition.py

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import mtcnn
from mtcnn.mtcnn import MTCNN

from os import listdir
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, constraints
from tensorflow.keras import models
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import random

def run_facial_recognition():
    def get_sample_test_image():
        expected_class = random.randint(1, 15)
        random_sample = random.randint(1, 3)
        image_path = f"data/train/class_{expected_class}/class_{expected_class}_{random_sample}.png"
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return img, expected_class

    def preprocess_image(img):
        image = Image.fromarray(img)
        image = image.resize((160, 160))
        face_array = np.asarray(image)
        face_array = face_array.reshape(160, 160, 1)
        face_array = face_array.astype('float32')
        scaled_image = np.expand_dims(face_array, axis=0)
        return scaled_image

    def prediction(image, debug=True):
        plt.imshow(image)
        plt.show()
        input_sample = preprocess_image(img)
        results = cnnmodel.predict(input_sample)
        result = np.argmax(results, axis=1)
        index = result[0]
        confidence = results[0][index] * 100
        classes = np.load(os.path.join("model", class_names_file), allow_pickle=True).item()
        if type(classes) is dict:
            for k, v in classes.items():
                if k == index:
                    class_name = v
        if debug:
            print(results)
            print("Detected class is {} with {:.2f}% confidence".format(class_name, round(confidence, 2)))
        return class_name, confidence

    THRESHOLD = 90

    ALLOWED_USERS = ["class_1", "class_2", "class_3", "class_4", "class_5"]

    def authenticate(img, debug=False):
        class_name, confidence = prediction(img, debug)
        if (confidence < THRESHOLD):
            print("Face not recognized")
        elif (class_name in ALLOWED_USERS):
            print("Access Granted for {}".format(class_name))
        else:
            print("You are not permitted. Detected class: {}, Confidence: {:.2f}%".format(class_name,
                                                                                          round(confidence, 2)))

    img, expected_class = get_sample_test_image()
    print(f"expected class {expected_class}")

    authenticate(img)


