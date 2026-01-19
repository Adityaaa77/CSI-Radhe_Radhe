import cv2
import time
import numpy as np
import winsound
import yaml

# ---------------- ALERT ----------------
def alert_beep():
    winsound.Beep(1000, 5000)  # 5 sec beep

# ---------------- LOAD MODELS ----------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

net = cv2.dnn.readNet("yolov5n.onnx")

with open("coco.yaml", "r") as f:
    coco = yaml.safe_load(f)
classes = coco["names"]

cap = cv2.VideoCapture(0)

eye_closed_start = None
alert_active = False
alert_start = 0

print("AI Proctoring System Started")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ---------------- OBJECT DETECTION ----------------
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), swapRB=True)
    net.setInput(blob)
    preds = net.forward()[0]

    for det in preds:
        conf = det[4]
        if conf > 0.45:
            class_id = int(np.argmax(det[5:]))
            label = classes[class_id]

            if label in ["cell phone", "book"]:
                cv2.putText(frame, f"{label.upper()} DETECTED",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

                if not alert_active:
                    alert_beep()
                    alert_active = True
                    alert_start = time.time()

    # ---------------- FACE & EYE LOGIC ----------------
    for (x, y, fw, fh) in faces:
        face = gray[y:y+fh, x:x+fw]
        eyes = eye_cascade.detectMultiScale(face, 1.3, 5)

        # -------- EYES CLOSED / SLEEP --------
        if len(eyes) == 0:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > 2:
                cv2.putText(frame, "SLEEPING DETECTED",
                            (50, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)
                if not alert_active:
                    alert_beep()
                    alert_active = True
                    alert_start = time.time()
        else:
            eye_closed_start = None

        # -------- HEAD DIRECTION --------
        center_x = x + fw // 2
        center_y = y + fh // 2

        if center_x < w * 0.4:
            direction = "LEFT"
        elif center_x > w * 0.6:
            direction = "RIGHT"
        elif center_y < h * 0.4:
            direction = "UP"
        elif center_y > h * 0.6:
            direction = "DOWN"
        else:
            direction = "CENTER"

        cv2.putText(frame, f"HEAD: {direction}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (255, 0, 0), 2)

    # ---------------- ALERT RESET AFTER 5s ----------------
    if alert_active and time.time() - alert_start > 5:
        alert_active = False

    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
