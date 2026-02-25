import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
names = {}
label_id = 0

# Load dataset
for name in os.listdir("dataset"):
    names[label_id] = name
    for img in os.listdir(f"dataset/{name}"):
        img_path = f"dataset/{name}/{img}"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        labels.append(label_id)
    label_id += 1

recognizer.train(faces, np.array(labels))

# Attendance file
file = "attendance.csv"
if not os.path.exists(file):
    pd.DataFrame(columns=["Name","Date","Time","Status"]).to_csv(file, index=False)

df = pd.read_csv(file)
today = datetime.now().strftime("%Y-%m-%d")

url = "http://192.168.29.124:8080/video"
cap = cv2.VideoCapture(url)
marked = set()

print("Scanning faces... Press q to quit")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = names[label]

            if name not in marked:
                time_now = datetime.now().strftime("%H:%M:%S")
                df.loc[len(df)] = [name, today, time_now, "Present"]
                df.to_csv(file, index=False)
                marked.add(name)

            cv2.putText(frame, name, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Unknown", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Corporate Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()