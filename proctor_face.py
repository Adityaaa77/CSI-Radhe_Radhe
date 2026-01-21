# ============================================================
# AI PROCTORING SYSTEM – FINAL ENHANCED EDITION (<=2000 lines)
# ============================================================

import cv2
import time
import numpy as np
import winsound
import threading
from collections import deque, defaultdict
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pygetwindow as gw
from PIL import ImageGrab, Image, ImageTk
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sounddevice as sd
import queue
import pickle
from fpdf import FPDF
import json
from datetime import datetime
import platform
import psutil
import pyautogui
from pynput import keyboard, mouse
from io import BytesIO
import random

# ============================================================
# OPTIONAL IMPORTS
# ============================================================
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except:
    FACE_RECOGNITION_AVAILABLE = False
    print("face_recognition not available - identity verification will be basic")

try:
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False
    print("scikit-learn not available - using heuristic behavioral anomaly")

# ============================================================
# GLOBALS
# ============================================================
APP_TITLE = "AI Proctoring System - Final Enhanced Edition"
os.makedirs("screenshots", exist_ok=True)
os.makedirs("replay_data", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
os.makedirs("identity_data", exist_ok=True)

# Runtime state
running = True
current_user_id = None
current_test = None
test_questions = []
test_duration = 600  # 10 minutes
test_start_time = 0
user_answers = []
user_theory_answer = ""
disqualified = False
disqualification_reason = ""
violation_count = 0
tab_switches = 0
phone_time = 0

# Identity Verification
identity_verified = False
reference_face_encoding = None
face_verification_failures = 0

# Behavioral Analytics
mouse_movements = []
behavioral_anomaly_score = 0.0

# Browser Control
clipboard_access_attempts = 0
right_click_attempts = 0
screen_capture_attempts = 0
fullscreen_violations = 0

# Eye Tracking
pupil_positions = []
attention_drops = 0
eye_velocity_high_count = 0

# Screen Recording
screen_recorder = None
recording_fps = 10
max_recording_minutes = 15

# Replay & audit
frame_buffer = deque(maxlen=300)
violation_events = []
risk_history = []
visited_windows = {}
active_window_title = "Initializing..."
window_switch_count = 0

# Camera detections
phone_detected = False
eyes_closed = False
head_direction = "CENTER"
gaze_down = False
multiple_faces = 0
eye_closed_start = None

# Voice detection
voice_queue = queue.Queue()
voice_threshold = 0.02
voice_alert = False
music_detected = False  # NEW: robust music detection

# Alerts
alert_active = False
alert_start_time = 0

# Heatmap
heatmap_accumulator = np.zeros((480, 640), dtype=np.float32)
eye_heatmap = np.zeros((480, 640), dtype=np.float32)

# Forbidden apps
forbidden_apps = ["Discord", "Slack", "WhatsApp", "YouTube", "Spotify",
                  "TeamViewer", "AnyDesk", "Telegram", "Messenger"]

# Network monitoring
suspicious_connections = []

# Organizer storage
candidate_reports = defaultdict(list)

# Risk Weights
RISK_WEIGHTS = {
    'phone': 45,
    'multiple_faces': 35,
    'background_voice': 25,
    'music': 35,
    'forbidden_app': 35,
    'identity_mismatch': 50,
    'behavioral_anomaly': 30,
    'eyes_closed': 10,
    'head_movement': 10,
    'gaze_down': 15,
    'voice_alert': 10,
    'clipboard_attempt': 20,
    'screen_capture': 25,
    'fullscreen_exit': 15,
    'attention_drop': 12,
    'network_suspicious': 25
}

# ============================================================
# UTILITIES
# ============================================================
def play_alert():
    try:
        winsound.Beep(1000, 300)
    except:
        pass

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def calculate_multi_modal_risk():
    risk = 0
    factors = []

    if phone_detected:
        risk += RISK_WEIGHTS['phone']
        factors.append("Phone")
    if multiple_faces >= 2:
        risk += RISK_WEIGHTS['multiple_faces']
        factors.append(f"{multiple_faces} Faces")
    if voice_alert:
        risk += RISK_WEIGHTS['background_voice']
        factors.append("Background Voice")
    if music_detected:
        risk += RISK_WEIGHTS['music']
        factors.append("Music")
    if eyes_closed:
        risk += RISK_WEIGHTS['eyes_closed']
        factors.append("Eyes Closed")
    if head_direction != "CENTER":
        risk += RISK_WEIGHTS['head_movement']
        factors.append(f"Head {head_direction}")
    if gaze_down:
        risk += RISK_WEIGHTS['gaze_down']
        factors.append("Gaze Down")
    if face_verification_failures > 0:
        risk += RISK_WEIGHTS['identity_mismatch']
        factors.append("ID Mismatch")
    if behavioral_anomaly_score > 0.7:
        risk += RISK_WEIGHTS['behavioral_anomaly']
        factors.append("Behavior Anomaly")
    if clipboard_access_attempts > 0:
        risk += RISK_WEIGHTS['clipboard_attempt']
        factors.append("Clipboard")
    if screen_capture_attempts > 0:
        risk += RISK_WEIGHTS['screen_capture']
        factors.append("Screen Capture")
    if fullscreen_violations > 0:
        risk += RISK_WEIGHTS['fullscreen_exit']
        factors.append("Fullscreen Exit")
    if attention_drops > 0:
        risk += RISK_WEIGHTS['attention_drop']
        factors.append("Attention Drop")
    if len(suspicious_connections) > 0:
        risk += RISK_WEIGHTS['network_suspicious']
        factors.append("Network")

    return min(100, risk), factors

def save_violation_event(frame, violations_text, risk_score):
    try:
        timestamp = time.time() - test_start_time
        screenshot = ImageGrab.grab()
        event = {
            'timestamp': timestamp,
            'time_str': time.strftime("%H:%M:%S"),
            'frame': frame.copy(),
            'screenshot': np.array(screenshot),
            'risk_score': risk_score,
            'violations': violations_text,
            'active_window': active_window_title,
            'behavioral_score': behavioral_anomaly_score,
            'identity_confidence': 1.0 - (face_verification_failures * 0.1)
        }
        violation_events.append(event)

        if len(violation_events) % 10 == 0:
            with open(os.path.join("replay_data", f'violations_{int(timestamp)}.pkl'), 'wb') as f:
                pickle.dump(violation_events[-10:], f)

        img_path = os.path.join("screenshots", f"violation_{len(violation_events)}_{event['time_str'].replace(':','-')}.png")
        Image.fromarray(event['screenshot']).save(img_path)
    except Exception as e:
        print("Error saving violation:", e)

# ============================================================
# IDENTITY VERIFICATION
# ============================================================
def capture_identity_snapshot():
    global reference_face_encoding, identity_verified

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    countdown = 2
    while countdown > 0:
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, f"Position your face clearly... {countdown}", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Identity Verification", frame)
            cv2.waitKey(1000)
            countdown -= 1

    ret, frame = cap.read()
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        identity_verified = False
        return False

    if FACE_RECOGNITION_AVAILABLE:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 1:
            encs = face_recognition.face_encodings(rgb_frame, face_locations)
            if len(encs) > 0:
                reference_face_encoding = encs[0]
                timestamp = int(time.time())
                ref_path = os.path.join("identity_data", f"{current_user_id}_reference_{timestamp}.jpg")
                cv2.imwrite(ref_path, frame)
                identity_verified = True
                cv2.putText(frame, "✓ VERIFIED", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Identity Verification", frame)
                cv2.waitKey(1200)
                cap.release()
                cv2.destroyAllWindows()
                return True

    # Fallback: single face presence
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
    identity_verified = len(faces) == 1
    cv2.putText(frame, "✓ VERIFIED" if identity_verified else "✗ TRY AGAIN", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if identity_verified else (0, 0, 255), 2)
    cv2.imshow("Identity Verification", frame)
    cv2.waitKey(1200)
    cap.release()
    cv2.destroyAllWindows()
    return identity_verified

def verify_identity_periodic(frame):
    global face_verification_failures
    if not FACE_RECOGNITION_AVAILABLE or reference_face_encoding is None:
        return True
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            return True
        encs = face_recognition.face_encodings(rgb_frame, face_locations)
        if len(encs) > 0:
            matches = face_recognition.compare_faces([reference_face_encoding], encs[0], tolerance=0.6)
            if not matches[0]:
                face_verification_failures += 1
                return False
            else:
                face_verification_failures = max(0, face_verification_failures - 1)
                return True
    except:
        pass
    return True

# ============================================================
# BEHAVIORAL ANALYTICS (HEURISTIC + OPTIONAL IF)
# ============================================================
class BehavioralAnalyzer:
    def __init__(self):
        self.mouse_velocities = deque(maxlen=200)
        self.model = IsolationForest(n_estimators=50, contamination=0.1) if ML_AVAILABLE else None
        self.buffer = deque(maxlen=200)

    def add_mouse_movement(self, x, y, timestamp):
        if len(mouse_movements) > 0:
            px, py, pt = mouse_movements[-1]
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            dt = max(1e-3, timestamp - pt)
            vel = dist / dt
            self.mouse_velocities.append(vel)
            self.buffer.append([vel])
        mouse_movements.append((x, y, timestamp))

    def calculate_anomaly_score(self):
        if ML_AVAILABLE and len(self.buffer) >= 30:
            try:
                X = np.array(self.buffer)
                preds = self.model.fit_predict(X)
                # fraction of anomalies in recent window
                frac = (preds == -1).mean()
                return float(min(1.0, max(0.0, frac)))
            except:
                pass
        # heuristic: high variance in velocity suggests inconsistency
        if len(self.mouse_velocities) >= 10:
            var = np.std(self.mouse_velocities)
            return float(min(1.0, var / 1000.0))
        return 0.0

behavioral_analyzer = BehavioralAnalyzer()

# ============================================================
# BROWSER LOCKDOWN
# ============================================================
class BrowserLockdown:
    def __init__(self):
        self.keyboard_listener = None
        self.mouse_listener = None

    def start(self):
        def on_press(key):
            global clipboard_access_attempts, screen_capture_attempts
            try:
                if key == keyboard.Key.print_screen:
                    screen_capture_attempts += 1
                    return False
                # Detect Ctrl+C/V via key combination state
                if hasattr(key, 'char'):
                    pass
                if hasattr(key, 'name'):
                    ctrl_pressed = keyboard.Controller().pressed(keyboard.Key.ctrl)
                    if ctrl_pressed and key.name in ['c', 'v']:
                        clipboard_access_attempts += 1
                        return False
            except:
                pass

        def on_click(x, y, button, pressed):
            global right_click_attempts
            if pressed:
                behavioral_analyzer.add_mouse_movement(x, y, time.time())
                if button == mouse.Button.right:
                    right_click_attempts += 1
                    return False

        def on_move(x, y):
            behavioral_analyzer.add_mouse_movement(x, y, time.time())

        self.keyboard_listener = keyboard.Listener(on_press=on_press, suppress=True)
        self.mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move, suppress=False)
        self.keyboard_listener.start()
        self.mouse_listener.start()

    def stop(self):
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()

browser_lockdown = BrowserLockdown()

# ============================================================
# SCREEN RECORDING
# ============================================================
class ScreenRecorder:
    def __init__(self, fps=10):
        self.fps = fps
        self.frames = []
        self.recording = False
        self.max_frames = fps * 60 * max_recording_minutes

    def start(self):
        self.recording = True
        threading.Thread(target=self._record_loop, daemon=True).start()

    def _record_loop(self):
        while self.recording and running:
            try:
                screenshot = ImageGrab.grab()
                frame = np.array(screenshot)
                self.frames.append(frame)
                if len(self.frames) > self.max_frames:
                    self.frames.pop(0)
                time.sleep(1 / self.fps)
            except:
                pass

    def stop(self):
        self.recording = False

    def save_video(self, filepath):
        if len(self.frames) == 0:
            return
        try:
            h, w, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))
            for frame in self.frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"Screen recording saved: {filepath}")
        except Exception as e:
            print(f"Error saving recording: {e}")

    def get_highlight_frames(self):
        highlights = []
        for event in violation_events:
            idx = int(event['timestamp'] * self.fps)
            if 0 <= idx < len(self.frames):
                highlights.append({
                    'frame': self.frames[idx],
                    'timestamp': event['timestamp'],
                    'violations': event['violations']
                })
        return highlights

# ============================================================
# NETWORK MONITORING
# ============================================================
def monitor_network():
    global suspicious_connections
    try:
        connections = psutil.net_connections(kind='inet')
        for conn in connections:
            if conn.status == 'ESTABLISHED' and conn.raddr:
                remote_ip = conn.raddr.ip
                remote_port = conn.raddr.port
                suspicious_ports = [3389, 5900, 22, 23, 445, 139]
                if remote_port in suspicious_ports:
                    if remote_ip not in suspicious_connections:
                        suspicious_connections.append(remote_ip)
    except:
        pass

# ============================================================
# VOICE & MUSIC MONITOR
# ============================================================
def voice_monitor_thread():
    global voice_alert, music_detected, running

    # Rolling window for audio energy and spectral features
    energy_window = deque(maxlen=20)
    flatness_window = deque(maxlen=20)

    def audio_callback(indata, frames, time_, status):
        # Normalize energy
        audio = indata[:, 0].astype(np.float32)
        energy = float(np.linalg.norm(audio) / max(frames, 1))
        # FFT-based spectral flatness (simple proxy for music vs speech)
        try:
            spectrum = np.abs(np.fft.rfft(audio))
            spectrum = spectrum + 1e-8
            geo_mean = np.exp(np.mean(np.log(spectrum)))
            arith_mean = np.mean(spectrum)
            flatness = float(geo_mean / arith_mean)  # closer to 1 => noise-like; lower => tonal/harmonic
        except:
            flatness = 0.5

        energy_window.append(energy)
        flatness_window.append(flatness)

        # Voice alert: short-term energy above threshold
        voice_alert_local = energy > voice_threshold
        if voice_alert_local:
            voice_queue.put(energy)

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
            while running:
                try:
                    # Voice alert if recent energy spikes exist
                    voice_alert = not voice_queue.empty()
                    if not voice_queue.empty():
                        _ = voice_queue.get()

                    # Music detection: sustained energy + low spectral flatness (tonal)
                    if len(energy_window) >= 10 and len(flatness_window) >= 10:
                        avg_energy = np.mean(energy_window)
                        avg_flatness = np.mean(flatness_window)
                        # Heuristic: music tends to have sustained energy and lower flatness
                        music_detected = (avg_energy > (voice_threshold * 1.5)) and (avg_flatness < 0.4)
                    else:
                        music_detected = False
                except:
                    pass
                time.sleep(0.1)
    except Exception as e:
        print("Voice stream error:", e)

# ============================================================
# CAMERA PROCTORING (FAST PHONE + SMOOTH MULTI-FACE)
# ============================================================
def camera_proctoring_thread():
    global phone_detected, eyes_closed, head_direction, gaze_down, multiple_faces
    global alert_active, alert_start_time, eye_closed_start, risk_history, running
    global phone_time, violation_count, eye_heatmap, pupil_positions, attention_drops
    global eye_velocity_high_count

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # YOLO (fast path): use smaller input + NMS
    yolo_available = False
    classes = []
    try:
        net = cv2.dnn.readNet("yolov5n.onnx")
        yolo_available = True
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
            "toaster","sink","refrigerator","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush"
        ]
    except:
        print("YOLO model not found")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    last_violation_save = 0
    last_identity_check = time.time()
    last_pupil_pos = None
    last_network_check = time.time()
    face_count_smooth = 0  # EMA for multiple faces

    ret, init_frame = cap.read()
    if ret:
        h, w = init_frame.shape[:2]
        global heatmap_accumulator, eye_heatmap
        heatmap_accumulator = np.zeros((h, w), dtype=np.float32)
        eye_heatmap = np.zeros((h, w), dtype=np.float32)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reset flags
        phone_detected = False
        eyes_closed = False
        head_direction = "CENTER"
        gaze_down = False

        # Periodic identity verification (every 30s)
        if time.time() - last_identity_check > 30:
            if not verify_identity_periodic(frame):
                save_violation_event(frame, "Identity Mismatch", 50)
            last_identity_check = time.time()

        # Network monitoring (every 10s)
        if time.time() - last_network_check > 10:
            monitor_network()
            last_network_check = time.time()

        # YOLO phone detection (fast)
        if yolo_available:
            try:
                # Resize to 416x416 for speed
                blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
                net.setInput(blob)
                outputs = net.forward()[0]
                boxes = []
                confidences = []
                class_ids = []

                for det in outputs:
                    conf = float(det[4])
                    if conf < 0.25:
                        continue
                    class_probs = det[5:]
                    class_id = int(np.argmax(class_probs))
                    score = conf * class_probs[class_id]
                    if score < 0.35:
                        continue
                    if classes[class_id] == "cell phone":
                        # Extract bbox
                        cx, cy, w_box, h_box = det[0]*width, det[1]*height, det[2]*width, det[3]*height
                        x = int(cx - w_box/2)
                        y = int(cy - h_box/2)
                        boxes.append([x, y, int(w_box), int(h_box)])
                        confidences.append(float(score))
                        class_ids.append(class_id)

                # NMS to reduce duplicates
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.4)
                if len(idxs) > 0:
                    phone_detected = True
                    phone_time += 1
            except:
                pass

        # Face/Eye/Gaze + smoothing for multiple faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))
        # Smooth face count via EMA
        face_count_current = len(faces)
        face_count_smooth = int(0.6 * face_count_smooth + 0.4 * face_count_current)
        multiple_faces = face_count_smooth

        for (x, y, w_box, h_box) in faces:
            face_gray = gray[y:y+h_box, x:x+w_box]
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

            if len(eyes) == 0:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start > 3:
                    eyes_closed = True
                    attention_drops += 1
            else:
                eyes_closed = False
                eye_closed_start = None

                # Eye tracking
                for (ex, ey, ew, eh) in eyes:
                    eye_cx = x + ex + ew // 2
                    eye_cy = y + ey + eh // 2
                    if 0 <= eye_cy < height and 0 <= eye_cx < width:
                        eye_heatmap[eye_cy, eye_cx] += 1
                    pupil_positions.append((eye_cx, eye_cy, time.time()))
                    if last_pupil_pos and len(pupil_positions) > 1:
                        dx = eye_cx - last_pupil_pos[0]
                        dy = eye_cy - last_pupil_pos[1]
                        dt = time.time() - last_pupil_pos[2]
                        if dt > 0:
                            velocity = np.sqrt(dx**2 + dy**2) / dt
                            if velocity > 500:
                                eye_velocity_high_count += 1
                    last_pupil_pos = (eye_cx, eye_cy, time.time())
                    cv2.circle(frame, (eye_cx, eye_cy), 2, (0, 255, 255), -1)

            cx, cy = x + w_box // 2, y + h_box // 2
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            if 0 <= cy < height and 0 <= cx < width:
                heatmap_accumulator[cy, cx] += 1

            # Head direction
            if cx < width * 0.4:
                head_direction = "LEFT"
            elif cx > width * 0.6:
                head_direction = "RIGHT"
            else:
                head_direction = "CENTER"

            # Gaze down
            if cy > height * 0.65:
                gaze_down = True

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)

        # Behavioral anomaly
        global behavioral_anomaly_score
        behavioral_anomaly_score = behavioral_analyzer.calculate_anomaly_score()

        # Multi-modal risk
        violation_score, factors = calculate_multi_modal_risk()
        risk_history.append(violation_score)

        # Major violation counting
        if phone_detected or multiple_faces >= 2 or voice_alert or music_detected or \
           face_verification_failures > 2 or len(suspicious_connections) > 0:
            violation_count += 1

        # Save violations
        if violation_score > 30 and time.time() - last_violation_save > 1.5:
            save_violation_event(frame, ", ".join(factors), violation_score)
            last_violation_save = time.time()
            if not alert_active:
                play_alert()
                alert_active = True
                alert_start_time = time.time()

        if alert_active and time.time() - alert_start_time > 3:
            alert_active = False

        # Overlay
        y0, dy = 30, 25
        overlay = [
            f"Head: {head_direction}",
            f"Gaze Down: {gaze_down}",
            f"Phone: {phone_detected}",
            f"Faces (smooth): {multiple_faces}",
            f"Eyes Closed: {eyes_closed}",
            f"Voice: {voice_alert}",
            f"Music: {music_detected}",
            f"ID Fails: {face_verification_failures}",
            f"Behavior: {behavioral_anomaly_score:.2f}",
            f"Network: {len(suspicious_connections)}",
            f"Risk: {violation_score}"
        ]
        for i, text in enumerate(overlay):
            color = (0, 255, 0) if ("False" in text or "CENTER" in text or "0" in text) else (0, 0, 255)
            cv2.putText(frame, text, (20, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Risk bar
        bar_width, bar_height = 250, 20
        bar_x, bar_y = width - 270, 30
        fill = int(bar_width * violation_score / 100)
        bar_color = (0, 255, 0) if violation_score < 30 else ((0, 255, 255) if violation_score < 60 else (0, 0, 255))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_height), bar_color, -1)
        cv2.putText(frame, "RISK", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Flash banner
        if violation_score > 0 and int(time.time() * 2) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 255), -1)
            cv2.putText(frame, "⚠ VIOLATION DETECTED", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Camera Proctoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# WINDOW MONITOR
# ============================================================
def window_monitor_thread():
    global active_window_title, window_switch_count, visited_windows
    global violation_count, running, fullscreen_violations

    last_title = None
    while running:
        try:
            active_window = gw.getActiveWindow()
            if active_window:
                title = active_window.title.strip() or "Unknown Window"
                # Fullscreen check
                screen_width = pyautogui.size().width
                screen_height = pyautogui.size().height
                if active_window.width < screen_width * 0.9 or active_window.height < screen_height * 0.9:
                    fullscreen_violations += 1

                if title != last_title:
                    last_title = title
                    active_window_title = title
                    window_switch_count += 1
                    if title not in visited_windows:
                        visited_windows[title] = now_ts()

                    # Forbidden app
                    if any(app.lower() in title.lower() for app in forbidden_apps):
                        violation_count += 1
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        path = os.path.join("screenshots", f"{ts}_{title.replace(' ','_')}.png")
                        ImageGrab.grab().save(path)
                        save_violation_event(
                            frame_buffer[-1] if frame_buffer else np.zeros((480, 640, 3), dtype=np.uint8),
                            f"Forbidden App: {title}",
                            60
                        )
                        play_alert()
        except:
            pass
        time.sleep(0.4)

# ============================================================
# PDF REPORT
# ============================================================
def generate_pdf_report(user_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "AI PROCTORED EXAM REPORT - FINAL", 0, 1, 'C', 1)
    pdf.ln(5)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Candidate: {user_id}", 0, 1)
    pdf.cell(0, 10, f"Test: {current_test}", 0, 1)
    pdf.cell(0, 10, f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(test_start_time))}", 0, 1)
    pdf.ln(5)

    # Score
    score = 0
    for i, ans in enumerate(user_answers):
        if i < len(test_questions) and ans == test_questions[i]['answer']:
            score += 1

    if disqualified:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"STATUS: DISQUALIFIED - {disqualification_reason}", 0, 1, 'C')
    else:
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 10, f"SCORE: {score}/{len(test_questions)} ({int(score*100/len(test_questions))}%)", 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Proctoring Summary", 0, 1)
    pdf.set_font("Arial", size=10)

    summary_data = [
        f"Total Major Violations: {violation_count}",
        f"Tab Switches: {window_switch_count}",
        f"Phone Detection Time: {phone_time}s",
        f"Identity Verification Failures: {face_verification_failures}",
        f"Behavioral Anomaly Score: {behavioral_anomaly_score:.2f}",
        f"Clipboard Access Attempts: {clipboard_access_attempts}",
        f"Screen Capture Attempts: {screen_capture_attempts}",
        f"Fullscreen Violations: {fullscreen_violations}",
        f"Attention Drops: {attention_drops}",
        f"Suspicious Network Connections: {len(suspicious_connections)}",
        f"High-Velocity Eye Movements: {eye_velocity_high_count}",
        f"Music Detected: {'Yes' if music_detected else 'No'}"
    ]
    for item in summary_data:
        pdf.cell(0, 7, item, 0, 1)
    pdf.ln(3)

    # Detailed violations
    if violation_events:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Violation Log", 0, 1)
        pdf.set_font("Arial", size=9)
        for i, event in enumerate(violation_events[:10]):
            pdf.cell(0, 6, f"#{i+1} | {event['time_str']} | Risk: {event['risk_score']}% | {event['violations']}", 0, 1)
            pdf.cell(0, 6, f"Window: {event['active_window']} | Behavior: {event['behavioral_score']:.2f} | ID: {event['identity_confidence']:.2f}", 0, 1)
            img_path = os.path.join("screenshots", f"report_violation_{i+1}.png")
            Image.fromarray(event['screenshot']).save(img_path)
            pdf.image(img_path, w=100)
            pdf.ln(4)

    # Risk trend graph
    if risk_history:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.plot(risk_history, color='red', linewidth=2)
        plt.title("Session Risk Trend", fontsize=14, fontweight='bold')
        plt.xlabel("Time (frames)")
        plt.ylabel("Risk Score")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        graph_path = os.path.join("reports", f"{user_id}_risk_graph.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()

        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Session Risk Trend Analysis", 0, 1)
        pdf.image(graph_path, w=170)

    # Heatmaps
    if heatmap_accumulator.sum() > 0:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(heatmap_accumulator, cmap='hot')
        ax1.set_title("Face Position Heatmap")
        ax1.axis('off')
        ax2.imshow(eye_heatmap, cmap='cool')
        ax2.set_title("Eye Tracking Heatmap")
        ax2.axis('off')
        heatmap_path = os.path.join("heatmaps", f"{user_id}_heatmaps.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Attention Heatmaps", 0, 1)
        pdf.image(heatmap_path, w=180)

    # Theory answer
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Theory Question Response", 0, 1)
    pdf.set_font("Arial", size=10)
    wrapped = user_theory_answer.replace("\r", "").split("\n")
    for line in wrapped:
        pdf.multi_cell(0, 6, line)

    report_path = os.path.join("reports", f"{user_id}_{current_test}_final_report.pdf")
    pdf.output(report_path)
    candidate_reports[user_id].append(report_path)
    return report_path

# ============================================================
# TEST DATABASE (MCQ + THEORY)
# ============================================================
TEST_DATABASE = {
    "Math": [
        {"question": "What is the value of π to two decimal places?", "options": ["3.14", "3.16", "3.18", "3.12"], "answer": 0},
        {"question": "Solve for x: 2x + 5 = 15", "options": ["x=5", "x=10", "x=7.5", "x=6"], "answer": 0},
        {"question": "What is the derivative of x²?", "options": ["2x", "x", "2", "x²"], "answer": 0},
        {"question": "Calculate the area of a circle with radius 5", "options": ["78.5", "31.4", "15.7", "25.0"], "answer": 0},
        {"question": "What is 7 × 8?", "options": ["56", "64", "42", "49"], "answer": 0}
    ],
    "English": [
        {"question": "Which word is a synonym for 'happy'?", "options": ["Joyful", "Sad", "Angry", "Tired"], "answer": 0},
        {"question": "Identify the verb: 'The cat sleeps peacefully'", "options": ["Cat", "Sleeps", "Peacefully", "The"], "answer": 1},
        {"question": "What is the past tense of 'go'?", "options": ["Went", "Goed", "Gone", "Going"], "answer": 0},
        {"question": "Which is a proper noun?", "options": ["city", "London", "river", "mountain"], "answer": 1},
        {"question": "What literary device is 'as brave as a lion'?", "options": ["Metaphor", "Simile", "Hyperbole", "Personification"], "answer": 1}
    ],
    "Science": [
        {"question": "What is the chemical symbol for gold?", "options": ["Au", "Ag", "Fe", "Cu"], "answer": 0},
        {"question": "Which planet is known as the Red Planet?", "options": ["Mars", "Venus", "Jupiter", "Saturn"], "answer": 0},
        {"question": "What is the powerhouse of the cell?", "options": ["Mitochondria", "Nucleus", "Ribosome", "Golgi"], "answer": 0},
        {"question": "What is H₂O?", "options": ["Water", "Hydrogen", "Oxygen", "Helium"], "answer": 0},
        {"question": "What force keeps us on the ground?", "options": ["Gravity", "Magnetism", "Friction", "Inertia"], "answer": 0}
    ]
}

THEORY_QUESTIONS = {
    "Math": "Explain the difference between differentiation and integration with a real-world example.",
    "English": "Discuss how tone and mood differ in literature, with examples from any text you’ve read.",
    "Science": "Describe how photosynthesis works and why it’s essential for life on Earth."
}

# ============================================================
# UI THEME (Professional website-like)
# ============================================================
def setup_style():
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TFrame", background="#0f172a")
    style.configure("Card.TFrame", background="#1e293b")
    style.configure("Accent.TButton", font=("Arial", 12, "bold"), foreground="white", background="#10b981")
    style.map("Accent.TButton", background=[("active", "#059669")])
    style.configure("Primary.TButton", font=("Arial", 12, "bold"), foreground="white", background="#3b82f6")
    style.map("Primary.TButton", background=[("active", "#2563eb")])
    style.configure("Warn.TButton", font=("Arial", 12, "bold"), foreground="white", background="#ef4444")
    style.map("Warn.TButton", background=[("active", "#dc2626")])
    style.configure("TLabel", background="#0f172a", foreground="white", font=("Arial", 12))
    style.configure("Card.TLabel", background="#1e293b", foreground="#bfdbfe", font=("Arial", 11))

# ============================================================
# START SCREEN
# ============================================================
class StartScreen:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1000x650")
        self.root.configure(bg="#0f172a")

        tk.Label(root, text="AI Proctoring System", font=("Arial", 28, "bold"), fg="white", bg="#0f172a").pack(pady=20)
        tk.Label(root, text="Final Enhanced Edition", font=("Arial", 14), fg="#60a5fa", bg="#0f172a").pack(pady=5)

        features_frame = ttk.Frame(root, style="Card.TFrame")
        features_frame.pack(pady=20, padx=40, fill="x")
        tk.Label(features_frame, text="🔹 Fast Phone Detection  🔹 Music Identification  🔹 Smooth Multi-Face",
                 font=("Arial", 11), fg="#bfdbfe", bg="#1e293b").pack(pady=4)
        tk.Label(features_frame, text="🔹 Identity Verification  🔹 Behavioral Analytics  🔹 Screen Recording",
                 font=("Arial", 11), fg="#bfdbfe", bg="#1e293b").pack(pady=4)
        tk.Label(features_frame, text="🔹 Browser Lockdown  🔹 Eye Tracking  🔹 Network Monitoring  🔹 Random MCQs + Theory",
                 font=("Arial", 11), fg="#bfdbfe", bg="#1e293b").pack(pady=4)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=30)

        ttk.Button(btn_frame, text="👤 Candidate (Client)", style="Accent.TButton",
                   command=self.open_candidate).grid(row=0, column=0, padx=20, ipadx=20, ipady=10)
        ttk.Button(btn_frame, text="🗂 Organizer (Server)", style="Primary.TButton",
                   command=self.open_organizer_login).grid(row=0, column=1, padx=20, ipadx=20, ipady=10)

        tk.Label(root, text="Next-Generation Integrity Monitoring • Forensic Reporting • ML-Powered Detection",
                 font=("Arial", 12), fg="#94a3b8", bg="#0f172a").pack(pady=10)

    def open_candidate(self):
        CandidateLogin(self.root)

    def open_organizer_login(self):
        OrganizerLogin(self.root)

# ============================================================
# Candidate Login + Test Selection
# ============================================================
class CandidateLogin:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Candidate Login")
        self.window.geometry("600x400")
        self.window.configure(bg="#0f172a")

        tk.Label(self.window, text="Candidate Sign-in", font=("Arial", 22, "bold"), fg="white", bg="#0f172a").pack(pady=20)
        self.user_entry = tk.Entry(self.window, font=("Arial", 14))
        self.user_entry.pack(pady=10)
        self.user_entry.insert(0, "user_001")

        ttk.Button(self.window, text="Continue", style="Accent.TButton",
                   command=self.continue_to_tests).pack(pady=20)

    def continue_to_tests(self):
        global current_user_id
        current_user_id = self.user_entry.get().strip() or f"user_{int(time.time())}"

        if messagebox.askyesno("Identity Verification", "Proceed with face verification?"):
            if not capture_identity_snapshot():
                messagebox.showerror("Verification Failed", "Could not verify identity. Please try again.")
                return

        TestSelectionScreen(self.window)

class TestSelectionScreen:
    def __init__(self, parent):
        self.root = tk.Toplevel(parent)
        self.root.title("Select Test")
        self.root.geometry("1000x650")
        self.root.configure(bg="#1e3a8a")

        tk.Label(self.root, text="AI PROCTORED EXAMINATION SYSTEM", font=("Arial", 26, "bold"), fg="white", bg="#1e3a8a").pack(pady=20)
        tk.Label(self.root, text="Secure • Automated • Intelligent • Final Edition", font=("Arial", 14), fg="#bfdbfe", bg="#1e3a8a").pack()

        card_frame = ttk.Frame(self.root)
        card_frame.pack(fill="both", expand=True, padx=50, pady=20)

        def start_test(name):
            global current_test, test_questions, test_start_time, user_answers, disqualified, disqualification_reason
            global screen_recorder

            current_test = name
            # Randomize MCQs each click: shuffle questions and options
            base_questions = TEST_DATABASE[name][:]
            random.shuffle(base_questions)
            for q in base_questions:
                opts = q['options'][:]
                correct_index = q['answer']
                correct_value = opts[correct_index]
                random.shuffle(opts)
                q['options'] = opts
                q['answer'] = opts.index(correct_value)
            test_questions = base_questions

            test_start_time = time.time()
            user_answers = []
            disqualified = False
            disqualification_reason = ""

            threading.Thread(target=camera_proctoring_thread, daemon=True).start()
            threading.Thread(target=window_monitor_thread, daemon=True).start()
            threading.Thread(target=voice_monitor_thread, daemon=True).start()

            browser_lockdown.start()

            screen_recorder = ScreenRecorder(fps=recording_fps)
            screen_recorder.start()

            TestScreen(self.root)

        subjects = ["Math", "English", "Science"]
        for idx, name in enumerate(subjects):
            card = ttk.Frame(card_frame, style="Card.TFrame")
            card.grid(row=0, column=idx, padx=20, pady=20, sticky="nsew")
            tk.Label(card, text=f"{name} TEST", font=("Arial", 18, "bold"), fg="white", bg="#1e293b").pack(pady=10)
            tk.Label(card, text="5 Questions | 10 Minutes", font=("Arial", 12), fg="white", bg="#1e293b").pack(pady=5)
            ttk.Button(card, text="START TEST", style="Accent.TButton",
                       command=lambda n=name: start_test(n)).pack(pady=20, ipadx=10, ipady=6)

        card_frame.columnconfigure(0, weight=1)
        card_frame.columnconfigure(1, weight=1)
        card_frame.columnconfigure(2, weight=1)

# ============================================================
# Test Screen (MCQ + Theory)
# ============================================================
class TestScreen:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title(f"AI Proctored Test - {current_test}")
        self.window.geometry("1200x800")
        self.window.configure(bg="#0f172a")

        try:
            self.window.attributes('-fullscreen', True)
        except:
            pass

        header = ttk.Frame(self.window)
        header.pack(fill="x", pady=(20, 10))
        tk.Label(header, text=f"{current_test} TEST - FINAL MONITORING", font=("Arial", 24, "bold"), fg="white", bg="#0f172a").pack()

        status_frame = ttk.Frame(self.window, style="Card.TFrame")
        status_frame.pack(fill="x", padx=20, pady=10)
        self.timer_label = tk.Label(status_frame, text="Time Left: 10:00", font=("Arial", 14), fg="white", bg="#1e293b")
        self.timer_label.pack(side="left")
        self.violation_label = tk.Label(status_frame, text="Violations: 0", font=("Arial", 14), fg="white", bg="#1e293b")
        self.violation_label.pack(side="right")
        self.risk_label = tk.Label(status_frame, text="Risk: 0%", font=("Arial", 14), fg="#10b981", bg="#1e293b")
        self.risk_label.pack(side="right", padx=20)

        # MCQ frame
        question_frame = ttk.Frame(self.window, style="Card.TFrame")
        question_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.question_label = tk.Label(question_frame, text="", font=("Arial", 16), fg="white", bg="#1e293b", wraplength=800, justify="left")
        self.question_label.pack(anchor="w", pady=(0, 20))

        self.option_vars = []
        self.option_buttons = []
        options_frame = ttk.Frame(question_frame, style="Card.TFrame")
        options_frame.pack(fill="both", expand=True)

        for i in range(4):
            var = tk.IntVar()
            btn = tk.Radiobutton(options_frame, text="", variable=var, value=i, font=("Arial", 14), fg="white", bg="#1e293b",
                                 selectcolor="#1e293b", activebackground="#1e293b", command=lambda v=i: self.select_answer(v))
            btn.pack(anchor="w", pady=10, padx=50)
            self.option_vars.append(var)
            self.option_buttons.append(btn)

        # Theory question
        theory_frame = ttk.Frame(self.window, style="Card.TFrame")
        theory_frame.pack(fill="x", padx=20, pady=10)
        tk.Label(theory_frame, text="Theory Question", font=("Arial", 16, "bold"), fg="white", bg="#1e293b").pack(anchor="w", pady=(10, 5))
        self.theory_label = tk.Label(theory_frame, text=THEORY_QUESTIONS.get(current_test, "Write your explanation."),
                                     font=("Arial", 12), fg="#bfdbfe", bg="#1e293b", wraplength=1000, justify="left")
        self.theory_label.pack(anchor="w", pady=(0, 10))
        self.theory_text = scrolledtext.ScrolledText(theory_frame, height=6, font=("Consolas", 11), bg="#0f172a", fg="white", insertbackground="white")
        self.theory_text.pack(fill="x", pady=(0, 10))

        nav_frame = ttk.Frame(self.window)
        nav_frame.pack(fill="x", padx=20, pady=10)
        self.prev_btn = ttk.Button(nav_frame, text="◀ Previous", style="Primary.TButton", command=self.prev_question)
        self.prev_btn.pack(side="left", padx=10)
        self.next_btn = ttk.Button(nav_frame, text="Next ▶", style="Accent.TButton", command=self.next_question)
        self.next_btn.pack(side="right", padx=10)

        self.current_question = 0
        self.load_question()
        self.update_timer()
        self.check_violations()

    def load_question(self):
        q = test_questions[self.current_question]
        self.question_label.config(text=f"Question {self.current_question+1}/{len(test_questions)}:\n{q['question']}")
        for i, btn in enumerate(self.option_buttons):
            btn.config(text=q['options'][i])
            self.option_vars[i].set(0)
        if len(user_answers) > self.current_question and user_answers[self.current_question] is not None:
            self.option_vars[user_answers[self.current_question]].set(1)
        self.prev_btn.config(state="normal" if self.current_question > 0 else "disabled")
        self.next_btn.config(text="Next ▶" if self.current_question < len(test_questions)-1 else "Finish Test")

    def select_answer(self, option):
        if len(user_answers) <= self.current_question:
            user_answers.extend([None] * (self.current_question - len(user_answers) + 1))
        user_answers[self.current_question] = option

    def prev_question(self):
        if self.current_question > 0:
            self.current_question -= 1
            self.load_question()

    def next_question(self):
        if self.current_question < len(test_questions)-1:
            # Randomize next MCQ on each click (reshuffle options for next question)
            self.current_question += 1
            q = test_questions[self.current_question]
            opts = q['options'][:]
            correct_value = opts[q['answer']]
            random.shuffle(opts)
            q['options'] = opts
            q['answer'] = opts.index(correct_value)
            self.load_question()
        else:
            self.finish_test()

    def update_timer(self):
        elapsed = time.time() - test_start_time
        remaining = max(0, test_duration - elapsed)
        mins, secs = divmod(int(remaining), 60)
        self.timer_label.config(text=f"Time Left: {mins:02d}:{secs:02d}")

        if len(risk_history) > 0:
            current_risk = risk_history[-1]
            color = "#10b981" if current_risk < 30 else ("#facc15" if current_risk < 60 else "#ef4444")
            self.risk_label.config(text=f"Risk: {current_risk}%", fg=color)

        if remaining <= 0:
            self.finish_test()
        else:
            self.window.after(1000, self.update_timer)

    def check_violations(self):
        global disqualified, disqualification_reason

        if violation_count >= 3:
            disqualified = True
            disqualification_reason = "Major violation limit reached (3)"
        elif face_verification_failures > 3:
            disqualified = True
            disqualification_reason = "Multiple identity verification failures"
        elif behavioral_anomaly_score > 0.85:
            disqualified = True
            disqualification_reason = "Highly anomalous behavior detected"
        elif len(suspicious_connections) > 2:
            disqualified = True
            disqualification_reason = "Multiple suspicious network connections"

        self.violation_label.config(text=f"Violations: {violation_count}")

        if disqualified:
            messagebox.showerror("Disqualified", disqualification_reason)
            self.finish_test()
        else:
            self.window.after(1000, self.check_violations)

    def finish_test(self):
        global user_theory_answer
        user_theory_answer = self.theory_text.get("1.0", tk.END).strip()

        browser_lockdown.stop()
        if screen_recorder:
            screen_recorder.stop()
            video_path = os.path.join("recordings", f"{current_user_id}_{current_test}_recording.mp4")
            screen_recorder.save_video(video_path)

        self.window.destroy()
        report_path = generate_pdf_report(current_user_id)
        ResultsScreen(report_path)

# ============================================================
# Results Screen
# ============================================================
class ResultsScreen:
    def __init__(self, report_path):
        win = tk.Toplevel()
        win.title("Test Results")
        win.geometry("900x700")
        win.configure(bg="#0f172a")

        tk.Label(win, text="TEST RESULTS - FINAL", font=("Arial", 24, "bold"), fg="white", bg="#0f172a").pack(pady=20)

        card = ttk.Frame(win, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=50, pady=20)

        tk.Label(card, text=f"{current_test} Test", font=("Arial", 20), fg="white", bg="#1e293b").pack(pady=10)

        score = 0
        for i, ans in enumerate(user_answers):
            if i < len(test_questions) and ans == test_questions[i]['answer']:
                score += 1

        if disqualified:
            result_text = f"DISQUALIFIED\nReason: {disqualification_reason}"
            color = "#ef4444"
        else:
            result_text = f"SCORE: {score}/{len(test_questions)} ({int(score*100/len(test_questions))}%)"
            color = "#10b981"

        tk.Label(card, text=result_text, font=("Arial", 24, "bold"), fg=color, bg="#1e293b").pack(pady=20)

        stats_frame = ttk.Frame(card, style="Card.TFrame")
        stats_frame.pack(pady=15)

        stats = [
            f"Total Violations: {violation_count}",
            f"Identity Failures: {face_verification_failures}",
            f"Behavioral Score: {behavioral_anomaly_score:.2f}",
            f"Attention Drops: {attention_drops}",
            f"Music Detected: {'Yes' if music_detected else 'No'}"
        ]
        for stat in stats:
            tk.Label(stats_frame, text=stat, font=("Arial", 11), fg="#bfdbfe", bg="#1e293b").pack()

        tk.Label(card, text=f"Report saved: {os.path.basename(report_path)}", font=("Arial", 12), fg="#bfdbfe", bg="#1e293b").pack(pady=15)

        btn_frame = ttk.Frame(card, style="Card.TFrame")
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Open Report", style="Primary.TButton",
                   command=lambda: os.startfile(report_path)).pack(side="left", padx=10)

        if screen_recorder and len(screen_recorder.get_highlight_frames()) > 0:
            ttk.Button(btn_frame, text="View Highlights", style="Accent.TButton",
                       command=self.view_highlights).pack(side="left", padx=10)

    def view_highlights(self):
        highlights_win = tk.Toplevel()
        highlights_win.title("Violation Highlights")
        highlights_win.geometry("800x600")
        highlights_win.configure(bg="#0f172a")

        tk.Label(highlights_win, text="Violation Highlights", font=("Arial", 20, "bold"),
                 fg="white", bg="#0f172a").pack(pady=10)

        canvas = tk.Canvas(highlights_win, bg="#1e293b")
        scrollbar = ttk.Scrollbar(highlights_win, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Card.TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        highlights = screen_recorder.get_highlight_frames()
        for i, highlight in enumerate(highlights[:10]):
            frame_rgb = cv2.cvtColor(highlight['frame'], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((400, 300))
            photo = ImageTk.PhotoImage(img)

            label = tk.Label(scrollable_frame, image=photo, bg="#1e293b")
            label.image = photo
            label.pack(pady=10)

            tk.Label(scrollable_frame,
                     text=f"Time: {int(highlight['timestamp'])}s | {highlight['violations']}",
                     font=("Arial", 11), fg="#bfdbfe", bg="#1e293b").pack()

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

# ============================================================
# Organizer Login + Dashboard
# ============================================================
class OrganizerLogin:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Organizer Login")
        self.window.geometry("500x300")
        self.window.configure(bg="#0f172a")

        tk.Label(self.window, text="Organizer Sign-in", font=("Arial", 22, "bold"), fg="white", bg="#0f172a").pack(pady=20)
        self.user_entry = tk.Entry(self.window, font=("Arial", 14))
        self.user_entry.pack(pady=10)
        self.user_entry.insert(0, "admin")

        self.pass_entry = tk.Entry(self.window, font=("Arial", 14), show="*")
        self.pass_entry.pack(pady=10)
        self.pass_entry.insert(0, "admin123")

        ttk.Button(self.window, text="Login", style="Primary.TButton",
                   command=self.login).pack(pady=20)

    def login(self):
        user = self.user_entry.get().strip()
        pwd = self.pass_entry.get().strip()
        # Simple hardcoded auth for demo
        if user == "admin" and pwd == "admin123":
            OrganizerDashboard(self.window)
        else:
            messagebox.showerror("Login Failed", "Invalid credentials")

class OrganizerDashboard:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Organizer Dashboard - Final")
        self.window.geometry("1200x800")
        self.window.configure(bg="#0f172a")

        tk.Label(self.window, text="Organizer Dashboard", font=("Arial", 26, "bold"), fg="white", bg="#0f172a").pack(pady=20)
        tk.Label(self.window, text="Advanced Analytics & Reports", font=("Arial", 14), fg="#bfdbfe", bg="#0f172a").pack()

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=20, pady=20)

        # Reports tab
        reports_tab = ttk.Frame(notebook, style="Card.TFrame")
        notebook.add(reports_tab, text="📄 Reports")

        self.listbox = tk.Listbox(reports_tab, bg="#1e293b", fg="white", font=("Consolas", 11), height=25)
        self.listbox.pack(fill="both", expand=True, padx=20, pady=20)

        btn_frame = ttk.Frame(reports_tab, style="Card.TFrame")
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="🔄 Refresh", style="Primary.TButton",
                   command=self.refresh).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="📂 Open Report", style="Accent.TButton",
                   command=self.open_selected).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="📊 Analytics", style="Primary.TButton",
                   command=self.show_analytics).pack(side="left", padx=10)

        # Analytics tab
        analytics_tab = ttk.Frame(notebook, style="Card.TFrame")
        notebook.add(analytics_tab, text="📊 Analytics")

        self.analytics_text = scrolledtext.ScrolledText(analytics_tab, bg="#1e293b", fg="white",
                                                        font=("Consolas", 10), height=30)
        self.analytics_text.pack(fill="both", expand=True, padx=20, pady=20)

        self.refresh()

    def refresh(self):
        self.listbox.delete(0, tk.END)
        for user_id, reports in candidate_reports.items():
            for rp in reports:
                self.listbox.insert(tk.END, f"{user_id} | {os.path.basename(rp)}")

    def open_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        item = self.listbox.get(sel[0])
        rp = item.split("|")[-1].strip()
        path = os.path.join("reports", rp)
        if os.path.exists(path):
            os.startfile(path)

    def show_analytics(self):
        self.analytics_text.delete(1.0, tk.END)
        avg_risk = np.mean(risk_history) if risk_history else 0.0
        analytics = f"""
{'='*60}
COMPREHENSIVE ANALYTICS REPORT
{'='*60}

Total Candidates: {len(candidate_reports)}
Total Reports Generated: {sum(len(reports) for reports in candidate_reports.values())}

VIOLATION STATISTICS:
{'─'*60}
Average Violations per Candidate: {violation_count / max(1, len(candidate_reports)):.2f}
Total Identity Failures: {face_verification_failures}
Average Behavioral Anomaly: {behavioral_anomaly_score:.3f}

SECURITY METRICS:
{'─'*60}
Clipboard Access Attempts: {clipboard_access_attempts}
Screen Capture Attempts: {screen_capture_attempts}
Fullscreen Violations: {fullscreen_violations}
Suspicious Network Connections: {len(suspicious_connections)}

ATTENTION & AUDIO:
{'─'*60}
Total Attention Drops: {attention_drops}
High-Velocity Eye Movements: {eye_velocity_high_count}
Music Detected (last): {'Yes' if music_detected else 'No'}
Average Risk Score: {avg_risk:.1f}%

CANDIDATE BREAKDOWN:
{'─'*60}
"""
        for user_id, reports in candidate_reports.items():
            analytics += f"\n{user_id}: {len(reports)} report(s)"

        analytics += f"""

{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
"""
        self.analytics_text.insert(1.0, analytics)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("AI PROCTORING SYSTEM - FINAL ENHANCED EDITION")
    print("=" * 70)
    print("\n🎯 FINAL FEATURES:")
    print("   ✅ Fast Phone Detection (YOLO + NMS + smaller input)")
    print("   ✅ Music Identification (spectral flatness + energy)")
    print("   ✅ Smooth Multiple-Face Handling (EMA)")
    print("   ✅ Identity Verification")
    print("   ✅ Behavioral Analytics (IF heuristic)")
    print("   ✅ Screen Recording & Highlights")
    print("   ✅ Browser Lockdown Mode")
    print("   ✅ Enhanced Eye Tracking")
    print("   ✅ Network Monitoring")
    print("   ✅ Random MCQs each click + Theory Question")
    print("   ✅ Organizer Login + Report Access")
    print("\n" + "=" * 70 + "\n")

    root = tk.Tk()
    setup_style()
    StartScreen(root)
    root.mainloop()
    running = False
