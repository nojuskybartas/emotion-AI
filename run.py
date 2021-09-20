from collections import defaultdict
from flask import Flask, render_template, Response
from app.main import scan_frame
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
    
    # TODO: future implementation of facial key point model
    
    # imgencode = cv2.imencode('.jpg', frame)[1]
    
    # # base64 encode
    # stringData = base64.b64encode(imgencode).decode('utf-8')
    # b64_src = 'data:image/jpg;base64,'
    # stringData = b64_src + stringData

    # # emit the frame back
    # emit('response_back', stringData)

def send_emotion():
    while True:
        if emotionDict is not None:
            yield "data: " + str(0) + "&&&" + str(emotionDict[0])  + "&&&" + str(emotionDict[1]) + "&&&" + str(emotionDict[2]) + "&&&" + str(emotionDict[3]) + "&&&" + str(emotionDict[4]) + "\n\n"

@ai_app.route('/')
def index():
    return render_template('index.html')

@ai_app.route('/emotion')
def emotion():
    return Response(send_emotion(), mimetype='text/event-stream')

if __name__=="__main__":
    socketio.run(ai_app, debug=True)