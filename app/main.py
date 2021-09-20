from collections import defaultdict
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow.keras as keras
from threading import Thread
from time import sleep


app = Flask(__name__)

with open('app/model/emotion.json', 'r') as json_file:
    json_savedModel = json_file.read()

# load the emotion detector model's architecture
emotionDetModel = keras.models.model_from_json(json_savedModel)
emotionDetModel.load_weights('app/model/weights_emotions.hdf5')
emotionDetModel.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

#loading openCV's pre-trained face detection data
trained_face_data = cv2.CascadeClassifier('app/model/haarcascade_frontalface_default.xml')

#creating a dictionary with the labels
label_to_text = {0:'angry', 1:'disgusted', 2:'sad', 3:'happy', 4: 'surprised'}

def prediction_to_text(prediction):
    prediction = list(map(float, iter(prediction[0])))
    d = defaultdict(int)
    for idx, em in enumerate(prediction):
        em = round(em*100, 2)
        d[idx] = em
    return d


lastReadFrame = None

def gen_frames():  
    camera = cv2.VideoCapture(0)
    global lastReadFrame
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            lastReadFrame = frame
            ret, frame = cv2.imencode('.jpg', frame)
            frame = frame.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def det_emotion():
    while True:
        if lastReadFrame is not None:
            gray_frame = cv2.cvtColor(lastReadFrame, cv2.COLOR_BGR2GRAY)
            face_coordinates = trained_face_data.detectMultiScale(gray_frame, minNeighbors=6, minSize=(100, 100))
            for idx, (x, y, w, h) in enumerate(face_coordinates):
                face = lastReadFrame[y:y + h, x:x + w]
                face = np.asarray(face).astype('float32')
                face = cv2.resize(face, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = np.stack(face, axis=0)
                face = face.reshape(1, 96, 96, 1)
                face = face / 255

                #print(emotionDetModel.predict(face))

                emotion_prediction = emotionDetModel.predict(face)
                emotionList = (prediction_to_text(emotion_prediction))
                
                # df_emotion = np.argmax(emotion_prediction, axis=-1)
                
                # df_emotion = np.expand_dims(df_emotion, axis=1)
                # emotion = label_to_text[df_emotion[0][0]]
                # print(df_emotion)

                yield "data: " + str(idx) + "&&&" + str(emotionList[0])  + "&&&" + str(emotionList[1]) + "&&&" + str(emotionList[2]) + "&&&" + str(emotionList[3]) + "&&&" + str(emotionList[4]) + "\n\n"
        
        sleep(0.5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def emotion():
    return Response(det_emotion(), mimetype='text/event-stream')


