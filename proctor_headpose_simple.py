import cv2
import time
import numpy as np
import winsound

# ---------------- ALERT ----------------
def alert_beep():
    winsound.Beep(1000, 1500)  # 1.5 sec beep (safe for demo)

# ---------------- LOAD MODELS ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# YOLOv5 ONNX model
net = cv2.dnn.readNet("yolov5n.onnx")

# COCO class names (HARDCODED – hackathon safe)
classes = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

eye_closed_start = None
alert_active = False
alert_start = 0

print("AI Proctoring System Started... Press Q to Exit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------- FACE DETECTION ----------------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ---------------- OBJECT DETECTION ----------------
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), swapRB=True)
    net.setInput(blob)
    detections = net.forward()[0]

    for det in detections:
        confidence = det[4]
        if confidence > 0.45:
            class_id = int(np.argmax(det[5:]))
            label = classes[class_id]

            if label in ["cell phone", "book"]:
                cv2.putText(frame, f"{label.upper()} DETECTED",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

                if not alert_active:
                    alert_beep()
                    alert_active = True
                    alert_start = time.time()

    # ---------------- FACE, EYE & HEAD LOGIC ----------------
    for (x, y, fw, fh) in faces:
        face_gray = gray[y:y+fh, x:x+fw]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 5)

        # ---- EYES CLOSED (SLEEP / INATTENTIVE) ----
        if len(eyes) == 0:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > 2:
                cv2.putText(frame, "EYES CLOSED",
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

                if not alert_active:
                    alert_beep()
                    alert_active = True
                    alert_start = time.time()
        else:
            eye_closed_start = None

        # ---- HEAD DIRECTION ----
        cx = x + fw // 2
        cy = y + fh // 2

        if cx < w * 0.4:
            direction = "LEFT"
        elif cx > w * 0.6:
            direction = "RIGHT"
        elif cy < h * 0.4:
            direction = "UP"
        elif cy > h * 0.6:
            direction = "DOWN"
        else:
            direction = "CENTER"

        cv2.putText(frame, f"HEAD: {direction}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + fw, y + fh),
                      (255, 0, 0), 2)

    # ---------------- ALERT RESET ----------------
    if alert_active and time.time() - alert_start > 5:
        alert_active = False

    # ---------------- DISPLAY ----------------
    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
