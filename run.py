from flask import Flask, render_template, Response, jsonify
from app.main import gen_frames, det_emotion

ai_app = Flask(__name__)

@ai_app.route('/')
def index():
    return render_template('index.html')

@ai_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@ai_app.route('/emotion')
def emotion():
    return Response(det_emotion(), mimetype='text/event-stream')

if __name__=="__main__":
    ai_app.run()