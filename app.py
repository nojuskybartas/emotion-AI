from collections import defaultdict
from flask import Flask, render_template, request
from model import scan_frame, label_to_text
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np

ai_app = Flask(__name__)
socketio = SocketIO(ai_app)

def bytes_to_array(bytes):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(bytes, np.uint8), -1), cv2.COLOR_BGR2RGB)

emotionDict = defaultdict(int)

@socketio.on('image')
def image(data_image):

    global emotionDict

    headers, image = data_image.split(',', 1) 

    frame_array = bytes_to_array(base64.b64decode(image))
    frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

    emotionDict = scan_frame(frame)

    if emotionDict is not None:
        emit('emotionDict', dict(emotionDict))

@ai_app.route('/')
def index():
    return render_template('index.html', address=request.host, numEmotions=5, label_to_text=label_to_text)

if __name__=="__main__":
    socketio.run(ai_app)