from collections import defaultdict
import cv2
import numpy as np
import tensorflow as tf

with open('model/emotion.json', 'r') as json_file:
    json_savedModel = json_file.read()

# load the emotion detector model's architecture
emotionDetModel = tf.keras.models.model_from_json(json_savedModel)
emotionDetModel.load_weights('model/weights_emotions.hdf5')
emotionDetModel.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

#loading openCV's pre-trained face detection data
trained_face_data = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

#creating a dictionary with the labels
label_to_text = {0:'angry', 1:'disgusted', 2:'sad', 3:'happy', 4: 'surprised'}

def prediction_to_text(prediction):
    prediction = list(map(float, iter(prediction[0])))
    d = defaultdict(int)
    for idx, em in enumerate(prediction):
        em = round(em*100, 2)
        d[idx] = em
    return d

def scan_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_frame, minNeighbors=6, minSize=(100, 100))
    for (x, y, w, h) in face_coordinates:
        face = frame[y:y + h, x:x + w]
        face = np.asarray(face).astype('float32')
        face = cv2.resize(face, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.stack(face, axis=0)
        face = face.reshape(1, 96, 96, 1)
        face = face / 255
        
        return prediction_to_text(emotionDetModel.predict(face))






