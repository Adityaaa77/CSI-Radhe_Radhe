# ============================================================
# AI PROCTORING SYSTEM – ONE-GO CLIENT/ORGANIZER VERSION
# ============================================================

import cv2
import time
import numpy as np
import winsound
import threading
from collections import deque, defaultdict
import tkinter as tk
from tkinter import ttk, messagebox
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

# ============================================================
# GLOBALS
# ============================================================
APP_TITLE = "AI Proctoring System"
os.makedirs("screenshots", exist_ok=True)
os.makedirs("replay_data", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)

# Runtime state
running = True
current_user_id = None
current_test = None
test_questions = []
test_duration = 600  # 10 minutes
test_start_time = 0
user_answers = []
disqualified = False
disqualification_reason = ""
violation_count = 0
tab_switches = 0
phone_time = 0
voice_talking_seconds = 0
voice_talking_start = None
voice_background_detected = False

# Replay & audit
frame_buffer = deque(maxlen=300)
violation_events = []  # dicts with timestamp, frame, screenshot, risk_score, violations, active_window
risk_history = []
visited_windows = {}
active_window_title = "Initializing..."
window_switch_count = 0

# Camera detections
phone_detected = False
eyes_closed = False
book_detected = False  # deprecated but kept for UI; not used for termination
head_direction = "CENTER"
gaze_down = False
multiple_faces = 0
eye_closed_start = None

# Voice detection
voice_queue = queue.Queue()
voice_threshold = 0.02  # sensitive
voice_alert = False

# Alerts
alert_active = False
alert_start_time = 0

# Heatmap (face center positions)
heatmap_accumulator = np.zeros((480, 640), dtype=np.float32)  # default size; will resize dynamically

# Forbidden apps
forbidden_apps = ["Discord", "Slack", "WhatsApp", "VSCode", "Chrome", "YouTube", "Spotify"]

# Organizer storage (simple local registry)
candidate_reports = defaultdict(list)  # user_id -> list of report paths

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
            'active_window': active_window_title
        }
        violation_events.append(event)
        # Persist every 10
        if len(violation_events) % 10 == 0:
            with open(os.path.join("replay_data", f'violations_{int(timestamp)}.pkl'), 'wb') as f:
                pickle.dump(violation_events[-10:], f)
        # Save screenshot with violation index
        img_path = os.path.join("screenshots", f"violation_{len(violation_events)}_{event['time_str'].replace(':','-')}.png")
        Image.fromarray(event['screenshot']).save(img_path)
    except Exception as e:
        print("Error saving violation:", e)

# ============================================================
# VOICE MONITOR
# ============================================================
def voice_monitor_thread():
    global voice_alert, voice_talking_seconds, voice_talking_start, voice_background_detected, running
    # Simple voice segregation heuristic:
    # - Continuous energy above threshold -> "talking"
    # - If sustained > 10s -> major violation (background talking)
    # - If intermittent -> warning only
    def audio_callback(indata, frames, time_, status):
        volume_norm = float(np.linalg.norm(indata) / max(frames, 1))
        if volume_norm > voice_threshold:
            voice_queue.put(volume_norm)

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
            while running:
                try:
                    if not voice_queue.empty():
                        voice_alert = True
                        _ = voice_queue.get()
                        if voice_talking_start is None:
                            voice_talking_start = time.time()
                        else:
                            voice_talking_seconds = int(time.time() - voice_talking_start)
                            if voice_talking_seconds >= 10:
                                voice_background_detected = True
                    else:
                        voice_alert = False
                        voice_talking_start = None
                        voice_talking_seconds = 0
                        voice_background_detected = False
                except:
                    pass
                time.sleep(0.1)
    except Exception as e:
        print("Voice stream error:", e)

# ============================================================
# CAMERA PROCTORING
# ============================================================
def camera_proctoring_thread():
    global phone_detected, eyes_closed, head_direction, gaze_down, multiple_faces
    global alert_active, alert_start_time, eye_closed_start, risk_history, running
    global phone_time, violation_count

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # YOLO (optional)
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
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush"
        ]
    except:
        print("YOLO model not found—phone detection will be limited.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    last_violation_save = 0
    last_phone_tick = time.time()

    # Heatmap size from first frame
    ret, init_frame = cap.read()
    if ret:
        h, w = init_frame.shape[:2]
        # Resize heatmap accumulator to match camera
        global heatmap_accumulator
        heatmap_accumulator = np.zeros((h, w), dtype=np.float32)

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
        multiple_faces = 0

        # YOLO phone detection
        if yolo_available:
            try:
                blob = cv2.dnn.blobFromImage(frame, 1/255, (640,640), swapRB=True)
                net.setInput(blob)
                outputs = net.forward()[0]
                for det in outputs:
                    conf = float(det[4])
                    if conf < 0.35:
                        continue
                    class_probs = det[5:]
                    class_id = int(np.argmax(class_probs))
                    label = classes[class_id]
                    score = conf * class_probs[class_id]
                    if label == "cell phone" and score > 0.45:
                        phone_detected = True
            except:
                pass

        # Face/Eye/Gaze/Head
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80,80))
        multiple_faces = len(faces)

        # Multiple faces -> major violation
        # Head down detection: face center y > 0.65*height or eyes not found for >3s
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20,20))

            if len(eyes) == 0:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start > 3:  # slower eye detection
                    eyes_closed = True
            else:
                eyes_closed = False
                eye_closed_start = None

            cx, cy = x + w//2, y + h//2
            # Heatmap accumulate
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            if 0 <= cy < height and 0 <= cx < width:
                heatmap_accumulator[cy, cx] += 1

            # Head direction by horizontal position
            if cx < width*0.4:
                head_direction = "LEFT"
            elif cx > width*0.6:
                head_direction = "RIGHT"
            else:
                head_direction = "CENTER"

            # Gaze down heuristic
            if cy > height*0.65:
                gaze_down = True

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        # Voice continuous talking major violation handled in voice thread

        # Violation scoring & logging
        violations_list = []
        violation_score = 0

        # Major violations (count towards termination)
        major_triggered = False
        if phone_detected:
            violations_list.append("Phone Detected")
            violation_score += 40
            major_triggered = True
            # phone time tick per second
            if time.time() - last_phone_tick > 1:
                phone_time += 1
                last_phone_tick = time.time()

        if multiple_faces >= 2:
            violations_list.append(f"Multiple Faces: {multiple_faces}")
            violation_score += 40
            major_triggered = True

        if voice_background_detected:
            violations_list.append("Background Voice >10s")
            violation_score += 40
            major_triggered = True

        # Minor warnings
        if eyes_closed:
            violations_list.append("Eyes Closed")
            violation_score += 10
        if head_direction != "CENTER":
            violations_list.append(f"Head {head_direction}")
            violation_score += 10
        if gaze_down:
            violations_list.append("Gaze Down")
            violation_score += 15
        if voice_alert and not voice_background_detected:
            violations_list.append("Voice Detected")
            violation_score += 10

        violation_score = min(100, violation_score)
        risk_history.append(violation_score)

        # Count major violations
        if major_triggered:
            violation_count += 1

        # Save violation event if score > 30 and cooldown
        if violation_score > 30 and time.time() - last_violation_save > 1.5:
            save_violation_event(frame, ", ".join(violations_list), violation_score)
            last_violation_save = time.time()
            if not alert_active:
                play_alert()
                alert_active = True
                alert_start_time = time.time()

        if alert_active and time.time() - alert_start_time > 3:
            alert_active = False

        # UI overlay
        y0, dy = 30, 30
        overlay = [
            f"Head: {head_direction}",
            f"Gaze Down: {gaze_down}",
            f"Phone: {phone_detected}",
            f"Faces: {multiple_faces}",
            f"Eyes Closed: {eyes_closed}",
            f"Voice Alert: {voice_alert}",
            f"Voice >10s: {voice_background_detected}",
            f"Risk: {violation_score}"
        ]
        for i, text in enumerate(overlay):
            color = (0,255,0) if ("False" in text or "CENTER" in text or "0" in text) else (0,0,255)
            cv2.putText(frame, text, (20, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Risk bar
        bar_width, bar_height = 250, 20
        bar_x, bar_y = width - 270, 30
        fill = int(bar_width * violation_score / 100)
        bar_color = (0,255,0) if violation_score < 30 else ((0,255,255) if violation_score < 60 else (0,0,255))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_width, bar_y+bar_height), (50,50,50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_height), bar_color, -1)
        cv2.putText(frame, "RISK", (bar_x, bar_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Flash banner
        if violation_score > 0 and int(time.time()*2) % 2 == 0:
            cv2.rectangle(frame, (0,0), (width, 40), (0,0,255), -1)
            cv2.putText(frame, "⚠ VIOLATION DETECTED", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

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
    global violation_count, running
    last_title = None
    while running:
        try:
            active_window = gw.getActiveWindow()
            if active_window:
                title = active_window.title.strip() or "Unknown Window"
                if title != last_title:
                    last_title = title
                    active_window_title = title
                    window_switch_count += 1
                    if title not in visited_windows:
                        visited_windows[title] = now_ts()
                    # Forbidden app -> major violation
                    if any(app.lower() in title.lower() for app in forbidden_apps):
                        violation_count += 1
                        # Screenshot with timestamp and window title
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        path = os.path.join("screenshots", f"{ts}_{title.replace(' ','_')}.png")
                        ImageGrab.grab().save(path)
                        # Log violation event
                        save_violation_event(
                            frame_buffer[-1] if frame_buffer else np.zeros((480,640,3), dtype=np.uint8),
                            f"Tab Switch to: {title}",
                            60
                        )
                        play_alert()
        except Exception as e:
            # print("Window monitor error:", e)
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
    pdf.cell(0, 10, "AI PROCTORED EXAM REPORT", 0, 1, 'C', 1)
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
        if ans == test_questions[i]['answer']:
            score += 1
    if disqualified:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"STATUS: DISQUALIFIED - {disqualification_reason}", 0, 1, 'C')
    else:
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 10, f"SCORE: {score}/5 ({score*20}%)", 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Proctoring Summary", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Total Violations (major): {violation_count}", 0, 1)
    pdf.cell(0, 8, f"Tab Switches: {window_switch_count}", 0, 1)
    pdf.cell(0, 8, f"Phone Time (s): {phone_time}", 0, 1)
    pdf.cell(0, 8, f"Continuous Background Voice (>=10s): {'Yes' if voice_background_detected else 'No'}", 0, 1)
    pdf.ln(5)

    # Detailed violations
    if violation_events:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Detailed Violations", 0, 1)
        pdf.set_font("Arial", size=10)
        for i, event in enumerate(violation_events[:12]):  # limit images to keep PDF size sane
            pdf.cell(0, 8, f"#{i+1} | {event['time_str']} | Risk: {event['risk_score']}% | {event['violations']}", 0, 1)
            pdf.cell(0, 8, f"Active Window: {event['active_window']}", 0, 1)
            # Save screenshot image
            img_path = os.path.join("screenshots", f"report_violation_{i+1}.png")
            Image.fromarray(event['screenshot']).save(img_path)
            pdf.image(img_path, w=120)
            pdf.ln(6)

    # Risk graph (simple inline)
    if risk_history:
        # Save a small graph image
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4,2))
        plt.plot(risk_history, color='red')
        plt.title("Session Risk Trend")
        plt.ylim(0,100)
        graph_path = os.path.join("reports", f"{user_id}_risk_graph.png")
        plt.savefig(graph_path, dpi=120, bbox_inches='tight')
        plt.close()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Session Risk Trend", 0, 1)
        pdf.image(graph_path, w=160)

    # Heatmap
    if heatmap_accumulator.sum() > 0:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4,3))
        plt.imshow(heatmap_accumulator, cmap='hot')
        plt.title("Face/Gaze Heatmap")
        plt.axis('off')
        heatmap_path = os.path.join("heatmaps", f"{user_id}_heatmap.png")
        plt.savefig(heatmap_path, dpi=120, bbox_inches='tight')
        plt.close()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Face/Gaze Heatmap", 0, 1)
        pdf.image(heatmap_path, w=160)

    # Save
    report_path = os.path.join("reports", f"{user_id}_{current_test}_report.pdf")
    pdf.output(report_path)
    candidate_reports[user_id].append(report_path)
    return report_path

# ============================================================
# TEST DATABASE
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
        {"question": "Which is a proper noun?", "options": ["city", "london", "river", "mountain"], "answer": 1},
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

# ============================================================
# UI — START SCREEN (User vs Organizer)
# ============================================================
class StartScreen:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("900x600")
        self.root.configure(bg="#0f172a")

        tk.Label(root, text="AI Proctoring System", font=("Arial", 26, "bold"), fg="white", bg="#0f172a").pack(pady=20)
        tk.Label(root, text="Choose mode", font=("Arial", 16), fg="#bfdbfe", bg="#0f172a").pack(pady=5)

        btn_frame = tk.Frame(root, bg="#0f172a")
        btn_frame.pack(pady=30)

        tk.Button(btn_frame, text="👤 Candidate (Client)", font=("Arial", 16, "bold"),
                  bg="#10b981", fg="white", padx=30, pady=15, command=self.open_candidate).grid(row=0, column=0, padx=20)
        tk.Button(btn_frame, text="🗂 Organizer (Server)", font=("Arial", 16, "bold"),
                  bg="#3b82f6", fg="white", padx=30, pady=15, command=self.open_organizer).grid(row=0, column=1, padx=20)

        tk.Label(root, text="Integrity-first remote assessments • Real-time monitoring • Forensic reporting",
                 font=("Arial", 12), fg="#94a3b8", bg="#0f172a").pack(pady=10)

    def open_candidate(self):
        CandidateLogin(self.root)

    def open_organizer(self):
        OrganizerDashboard(self.root)

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

        tk.Button(self.window, text="Continue", font=("Arial", 14, "bold"), bg="#10b981", fg="white",
                  command=self.continue_to_tests).pack(pady=20)

    def continue_to_tests(self):
        global current_user_id
        current_user_id = self.user_entry.get().strip() or f"user_{int(time.time())}"
        TestSelectionScreen(self.window)

class TestSelectionScreen:
    def __init__(self, parent):
        self.root = tk.Toplevel(parent)
        self.root.title("Select Test")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e3a8a")

        tk.Label(self.root, text="AI PROCTORED EXAMINATION SYSTEM", font=("Arial", 24, "bold"), fg="white", bg="#1e3a8a").pack(pady=20)
        tk.Label(self.root, text="Secure • Automated • Intelligent", font=("Arial", 14), fg="#bfdbfe", bg="#1e3a8a").pack()

        card_frame = tk.Frame(self.root, bg="#1e3a8a")
        card_frame.pack(fill="both", expand=True, padx=50, pady=20)

        def start_test(name):
            global current_test, test_questions, test_start_time, user_answers, disqualified, disqualification_reason
            current_test = name
            test_questions = TEST_DATABASE[name]
            test_start_time = time.time()
            user_answers = []
            disqualified = False
            disqualification_reason = ""
            # Start threads when test begins
            threading.Thread(target=camera_proctoring_thread, daemon=True).start()
            threading.Thread(target=window_monitor_thread, daemon=True).start()
            threading.Thread(target=voice_monitor_thread, daemon=True).start()
            TestScreen(self.root)

        for idx, name in enumerate(["Math", "English", "Science"]):
            card = tk.Frame(card_frame, bg="#2563eb", bd=2, relief="raised", padx=20, pady=20)
            card.grid(row=0, column=idx, padx=20, pady=20, sticky="nsew")
            tk.Label(card, text=f"{name} TEST", font=("Arial", 18, "bold"), fg="white", bg="#2563eb").pack(pady=10)
            tk.Label(card, text="5 Questions | 10 Minutes", font=("Arial", 12), fg="white", bg="#2563eb").pack(pady=5)
            tk.Button(card, text="START TEST", font=("Arial", 12, "bold"), bg="#10b981", fg="white",
                      padx=20, pady=10, command=lambda n=name: start_test(n)).pack(pady=20)

        card_frame.columnconfigure(0, weight=1)
        card_frame.columnconfigure(1, weight=1)
        card_frame.columnconfigure(2, weight=1)

# ============================================================
# Test Screen
# ============================================================
class TestScreen:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title(f"AI Proctored Test - {current_test}")
        self.window.geometry("1200x800")
        self.window.configure(bg="#0f172a")

        header = tk.Frame(self.window, bg="#0f172a")
        header.pack(fill="x", pady=(20, 10))
        tk.Label(header, text=f"{current_test} TEST", font=("Arial", 24, "bold"), fg="white", bg="#0f172a").pack()

        status_frame = tk.Frame(self.window, bg="#1e293b", padx=20, pady=10)
        status_frame.pack(fill="x", padx=20, pady=10)
        self.timer_label = tk.Label(status_frame, text="Time Left: 10:00", font=("Arial", 14), fg="white", bg="#1e293b")
        self.timer_label.pack(side="left")
        self.violation_label = tk.Label(status_frame, text="Violations: 0", font=("Arial", 14), fg="white", bg="#1e293b")
        self.violation_label.pack(side="right")

        question_frame = tk.Frame(self.window, bg="#1e293b", padx=20, pady=20)
        question_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.question_label = tk.Label(question_frame, text="", font=("Arial", 16), fg="white", bg="#1e293b", wraplength=800, justify="left")
        self.question_label.pack(anchor="w", pady=(0, 20))

        self.option_vars = []
        self.option_buttons = []
        options_frame = tk.Frame(question_frame, bg="#1e293b")
        options_frame.pack(fill="both", expand=True)
        for i in range(4):
            var = tk.IntVar()
            btn = tk.Radiobutton(options_frame, text="", variable=var, value=i, font=("Arial", 14), fg="white", bg="#1e293b",
                                 selectcolor="#1e293b", activebackground="#1e293b", command=lambda v=i: self.select_answer(v))
            btn.pack(anchor="w", pady=10, padx=50)
            self.option_vars.append(var)
            self.option_buttons.append(btn)

        nav_frame = tk.Frame(self.window, bg="#0f172a", padx=20, pady=10)
        nav_frame.pack(fill="x", padx=20, pady=10)
        self.prev_btn = tk.Button(nav_frame, text="◀ Previous", font=("Arial", 12), state="disabled", command=self.prev_question)
        self.prev_btn.pack(side="left", padx=10)
        self.next_btn = tk.Button(nav_frame, text="Next ▶", font=("Arial", 12, "bold"), command=self.next_question)
        self.next_btn.pack(side="right", padx=10)

        self.current_question = 0
        self.load_question()
        self.update_timer()
        self.check_violations()

    def load_question(self):
        q = test_questions[self.current_question]
        self.question_label.config(text=f"Question {self.current_question+1}/5:\n{q['question']}")
        for i, btn in enumerate(self.option_buttons):
            btn.config(text=q['options'][i])
            self.option_vars[i].set(0)
        if len(user_answers) > self.current_question and user_answers[self.current_question] is not None:
            self.option_vars[user_answers[self.current_question]].set(1)
        self.prev_btn.config(state="normal" if self.current_question > 0 else "disabled")
        self.next_btn.config(text="Next ▶" if self.current_question < 4 else "Finish Test")

    def select_answer(self, option):
        if len(user_answers) <= self.current_question:
            user_answers.extend([None] * (self.current_question - len(user_answers) + 1))
        user_answers[self.current_question] = option

    def prev_question(self):
        if self.current_question > 0:
            self.current_question -= 1
            self.load_question()

    def next_question(self):
        if self.current_question < 4:
            self.current_question += 1
            self.load_question()
        else:
            self.finish_test()

    def update_timer(self):
        elapsed = time.time() - test_start_time
        remaining = max(0, test_duration - elapsed)
        mins, secs = divmod(int(remaining), 60)
        self.timer_label.config(text=f"Time Left: {mins:02d}:{secs:02d}")
        if remaining <= 0:
            self.finish_test()
        else:
            self.window.after(1000, self.update_timer)

    def check_violations(self):
        global disqualified, disqualification_reason
        # Disqualify after 3 major violations
        if violation_count >= 3:
            disqualified = True
            disqualification_reason = "Major violation limit reached (3): Phone/Tab/Multi-face/Background voice"
        self.violation_label.config(text=f"Violations: {violation_count}")
        if disqualified:
            messagebox.showerror("Disqualified", disqualification_reason)
            self.finish_test()
        else:
            self.window.after(1000, self.check_violations)

    def finish_test(self):
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
        win.geometry("800x600")
        win.configure(bg="#0f172a")

        tk.Label(win, text="TEST RESULTS", font=("Arial", 24, "bold"), fg="white", bg="#0f172a").pack(pady=20)
        card = tk.Frame(win, bg="#1e293b", padx=30, pady=20)
        card.pack(fill="both", expand=True, padx=50, pady=20)

        tk.Label(card, text=f"{current_test} Test", font=("Arial", 20), fg="white", bg="#1e293b").pack(pady=10)
        score = 0
        for i, ans in enumerate(user_answers):
            if ans == test_questions[i]['answer']:
                score += 1
        if disqualified:
            result_text = f"DISQUALIFIED\nReason: {disqualification_reason}"
            color = "#ef4444"
        else:
            result_text = f"SCORE: {score}/5 ({score*20}%)"
            color = "#10b981"
        tk.Label(card, text=result_text, font=("Arial", 24, "bold"), fg=color, bg="#1e293b").pack(pady=20)

        tk.Label(card, text=f"Report saved: {report_path}", font=("Arial", 12), fg="#bfdbfe", bg="#1e293b").pack(pady=10)
        tk.Button(card, text="Open Report", font=("Arial", 12, "bold"), bg="#3b82f6", fg="white",
                  padx=20, pady=10, command=lambda: os.startfile(report_path)).pack(pady=10)

# ============================================================
# Organizer Dashboard
# ============================================================
class OrganizerDashboard:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Organizer Dashboard")
        self.window.geometry("1000x700")
        self.window.configure(bg="#0f172a")

        tk.Label(self.window, text="Organizer Dashboard", font=("Arial", 24, "bold"), fg="white", bg="#0f172a").pack(pady=20)
        tk.Label(self.window, text="Candidate Reports", font=("Arial", 14), fg="#bfdbfe", bg="#0f172a").pack()

        self.listbox = tk.Listbox(self.window, bg="#1e293b", fg="white", font=("Consolas", 12), height=20)
        self.listbox.pack(fill="both", expand=True, padx=20, pady=20)

        btn_frame = tk.Frame(self.window, bg="#0f172a")
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Refresh", font=("Arial", 12, "bold"), bg="#64748b", fg="white",
                  command=self.refresh).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Open Selected Report", font=("Arial", 12, "bold"), bg="#3b82f6", fg="white",
                  command=self.open_selected).pack(side="left", padx=10)

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

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    StartScreen(root)
    root.mainloop()
    running = False
