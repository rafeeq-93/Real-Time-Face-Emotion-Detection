from flask import Flask, render_template, Response
from facial_emotion_recognition import EmotionRecognition
import cv2

app = Flask(__name__)
er = EmotionRecognition(device='cpu')  # Initialize Emotion Recognition
cam = cv2.VideoCapture(0)  # Open webcam

def generate_frames():
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            frame = er.recognise_emotion(frame, return_type='BGR')  # Detect emotion
            _, buffer = cv2.imencode('.jpg', frame)  # Encode frame to JPEG
            frame_bytes = buffer.tobytes()  # Convert to bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # Send frame

@app.route('/')
def index():
    return render_template('index.html')  # Load frontend

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
