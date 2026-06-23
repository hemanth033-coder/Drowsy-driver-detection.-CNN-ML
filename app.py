import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response

app = Flask(__name__)

model = load_model("model/drowsy_model.h5")

labels = ["yawn", "no_yawn", "opened", "closed"]

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            face = cv2.resize(frame, (64,64))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face)
            label = labels[np.argmax(pred)]

            cv2.putText(frame, f"Status: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if label in ["yawn", "closed"]:
                cv2.putText(frame, "Drowsy Detected !", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
