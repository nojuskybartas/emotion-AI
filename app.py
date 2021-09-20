from flask import Flask, render_template, Response, jsonify
from app.main import gen_frames, det_emotion

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def emotion():
    return Response(det_emotion(), mimetype='text/event-stream')

if __name__=="__main__":
    app.run()