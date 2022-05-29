from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pygame
import threading
from motiondetector import SingleMotionDetector
from flask import Flask, render_template, Response

outputFrame = None
lock = threading.Lock()

vs = VideoStream(0 + cv2.CAP_DSHOW).start()
time.sleep(2.0)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('user_screen_detail.html')

def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def detect_drowsiness(frameCount):
    global vs, outputFrame, lock

    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    if args.alarm > 0:
        from gpiozero import TrafficHat

        th = TrafficHat()
        print("[INFO] using TrafficHat alarm...")

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 16

    COUNTER = 0
    ALARM_ON = False

    print("[INFO] loading facial landmark predictor...")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600, height=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True

                        pygame.mixer.init()
                        pygame.mixer.music.load('nomal_alarm.wav')
                        pygame.mixer.music.play()

                    cv2.putText(frame, "Wake Up!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
               COUNTER = 0
               ALARM_ON = False

        if total > frameCount:
            motion = md.detect(gray)
            if motion is not None:
                (thresh) = motion

        md.update(gray)
        total += 1

        with lock:
            outputFrame = frame.copy()

def gen_frames():
    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cascade", help="path to where the face cascade resides")
    ap.add_argument("-p", "--shape_predictor", help="path to facial landmark predictor")
    ap.add_argument("-a", "--alarm", type=int, default=0, help="boolean used to indicate if TraffHat should be used")
    ap.add_argument("-f", "--frame_count", type=int, default=32, help="# of frames used to construct the background model")
    args = ap.parse_args()

    t = threading.Thread(target=detect_drowsiness,args=(args.frame_count,))
    t.daemon = True
    t.start()

    #app.run(host=args.ip, port=args.port, debug=True, threaded=True, use_reloader=False)
    app.run(host="127.0.0.1", port="9900")

vs.stop()