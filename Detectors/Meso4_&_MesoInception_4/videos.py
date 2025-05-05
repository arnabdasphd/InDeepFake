# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.stats import mode
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

IMGWIDTH = 256

class Classifier:
    def __init__(self):
        self.model = 0
    
    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

class Meso1(Classifier):
    """
    Feature extraction + Classification
    """
    def __init__(self, learning_rate=0.001, dl_rate=1):
        self.model = self.init_model(dl_rate)
        optimizer = Adam(lr=learning_rate)  # Using 'lr' instead of 'learning_rate'
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
    def init_model(self, dl_rate):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(16, (3, 3), dilation_rate=dl_rate, strides=1, padding='same', activation='relu')(x)
        x1 = Conv2D(4, (1, 1), padding='same', activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(8, 8), padding='same')(x1)

        y = Flatten()(x1)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)
        return KerasModel(inputs=x, outputs=y)

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)  # Using 'lr' instead of 'learning_rate'
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
    def init_model(self): 
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)  # Correct usage of LeakyReLU
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)

class MesoInception4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)  # Using 'lr' instead of 'learning_rate'
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

            y = Concatenate(axis=-1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def init_model(self):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
        
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)  # Correct usage of LeakyReLU
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)

def extract_faces_from_frame(frame, face_cascade):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Extract and return face regions
    face_regions = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMGWIDTH, IMGWIDTH))
        face_regions.append(face)
    return face_regions

def process_video(video_path, model, face_cascade):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions = []

    with tqdm(total=frame_count, desc='Processing {}'.format(os.path.basename(video_path))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            faces = extract_faces_from_frame(frame, face_cascade)
            if faces:
                faces = np.array([img_to_array(face) for face in faces])
                faces = faces.astype('float32') / 255.0
                preds = model.predict(faces)
                binary_preds = (preds > 0.5).astype(int).flatten()  # Convert predictions to binary
                predictions.extend(binary_preds)
            pbar.update(1)  # Update the progress bar

    cap.release()
    if predictions:
        mode_prediction = mode(predictions).mode[0]  # Calculate the mode of the predictions
    else:
        mode_prediction = 0  # Default to 0 if no predictions were made

    return mode_prediction


def process_video_directory(video_directory, model):
    face_cascade_path = 'E:/RN_Mam/MesoNet-master/MesoNet-master/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    video_predictions = {}

    for filename in os.listdir(video_directory):
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            video_path = os.path.join(video_directory, filename)
            avg_pred = process_video(video_path, model, face_cascade)
            video_predictions[filename] = avg_pred
    return video_predictions

def evaluate_model(real_video_directory, df_video_directory, model):
    real_video_predictions = process_video_directory(real_video_directory, model)
    df_video_predictions = process_video_directory(df_video_directory, model)

    y_true = [1] * len(real_video_predictions) + [0] * len(df_video_predictions)
    y_pred = list(real_video_predictions.values()) + list(df_video_predictions.values())

    auc_score = roc_auc_score(y_true, y_pred)

    print("AUC Score: {}".format(auc_score))
    # for video, prediction in real_video_predictions.items():
    #    print("Video: {}, Mode Prediction: {}".format(video, prediction))
    #for video, prediction in df_video_predictions.items():
    #    print("Video: {}, Mode Prediction: {}".format(video, prediction))

if __name__ == "__main__":
    real_video_directory = "E:/RN_Mam/MesoNet-master/MesoNet-master/test_videos/test_1"
    df_video_directory = "E:/RN_Mam/MesoNet-master/MesoNet-master/test_videos/test_2"
    weights_path = "E:/RN_Mam/MesoNet-master/MesoNet-master/weights/MesoInception_F2F.h5"

    model = MesoInception4()  # Change to Meso1 or MesoInception4 if needed
    model.load(weights_path)

    # Evaluate model
    evaluate_model(real_video_directory, df_video_directory, model)
