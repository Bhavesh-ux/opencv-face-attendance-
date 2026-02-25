import cv2

# Load cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# IP Webcam URL
url = "http://192.168.29.124:8080/video"
camera = cv2.VideoCapture(url)

while True:
    ret, frame = camera.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        if len(eyes) > 0:
            cv2.putText(frame, "Eyes Detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Smile detection
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        if len(smile) > 0:
            cv2.putText(frame, "Smiling", (x, y-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Smart Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
