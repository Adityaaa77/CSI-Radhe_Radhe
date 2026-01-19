import cv2
import numpy as np
import time
from mediapipe.python.solutions import face_mesh

face_mesh = face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

violation_count = 0
look_start_time = None

print("AI Proctoring (MediaPipe Head Pose) Started... Press Q to exit")

def get_direction(landmarks, w, h):
    nose = landmarks[1]      # nose tip
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    nose_x = int(nose.x * w)
    nose_y = int(nose.y * h)

    eye_center_x = int((left_eye.x + right_eye.x) / 2 * w)
    eye_center_y = int((left_eye.y + right_eye.y) / 2 * h)

    dx = nose_x - eye_center_x
    dy = nose_y - eye_center_y

    if dx > 18:
        return "RIGHT"
    elif dx < -18:
        return "LEFT"
    elif dy > 15:
        return "DOWN"
    elif dy < -15:
        return "UP"
    else:
        return "CENTER"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status = "OK"
    direction = "CENTER"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            direction = get_direction(face_landmarks.landmark, w, h)

            if direction != "CENTER":
                if look_start_time is None:
                    look_start_time = time.time()
                elif time.time() - look_start_time > 1.5:
                    status = f"LOOKING {direction}"
                    violation_count += 1
            else:
                look_start_time = None

            # draw face box (simple)
            x_list = []
            y_list = []
            for lm in face_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x1, x2 = min(x_list), max(x_list)
            y1, y2 = min(y_list), max(y_list)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(frame, f"Dir: {direction}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Status: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.putText(frame, f"Violations: {violation_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("AI Proctor - MediaPipe Head Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
