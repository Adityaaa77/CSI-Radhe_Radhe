# ============================================================
# AI PROCTORING SYSTEM – FULL PREMIUM HACKATHON VERSION (UPDATED)
# ============================================================

import cv2
import time
import numpy as np
import winsound
import threading
from collections import deque
import datetime
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
# GLOBAL STATE
# ============================================================
running = True

# Replay & storage
frame_buffer = deque(maxlen=300)  # ~10 seconds rolling buffer at 30fps
violation_events = []  # {timestamp, time_str, frame, screenshot, risk_score, violations, active_window}
screenshot_folder = "screenshots"
replay_folder = "replay_data"
os.makedirs(screenshot_folder, exist_ok=True)
os.makedirs(replay_folder, exist_ok=True)

# Camera detections
phone_detected = False
book_detected = False
eyes_closed = False
head_direction = "CENTER"
camera_violation = False
risk_level = 0
eye_closed_start = None
face_count = 0

# Window monitoring
active_window_title = "Initializing..."
window_switch_count = 0
visited_windows = {}
forbidden_apps = ["Discord", "Slack", "WhatsApp", "VSCode", "Chrome", "YouTube", "Spotify"]

# Voice detection
voice_queue = queue.Queue()
voice_threshold = 0.02  # more sensitive
voice_alert = False
last_voice_time = 0

# Alert
alert_active = False
alert_start_time = 0

# Exam start
exam_start_time = time.time()

# Session risk
session_risk_score = 0
risk_history = []

# Test state
current_test = None
test_start_time = 0
test_duration = 600  # 10 minutes
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

# Violation counters
violation_count = 0  # major violations only
phone_time = 0       # seconds phone detected (for report)
tab_switches = 0     # forbidden tab switches (for report)
disqualified = False
disqualification_reason = ""

# Tab switch audit log
tab_switch_log = []  # list of dicts: {time_str, title, screenshot_path}

# ============================================================
# ALERT FUNCTION
# ============================================================
def play_alert():
    try:
        winsound.Beep(1000, 600)
    except:
        pass

# ============================================================
# SAVE VIOLATION EVENT
# ============================================================
def save_violation_event(frame, violations_text, risk_score):
    """Save violation event with camera frame and screenshot"""
    try:
        timestamp = time.time() - exam_start_time
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
        # Save to disk periodically (every 10 events)
        if len(violation_events) % 10 == 0:
            with open(os.path.join(replay_folder, f'violations_{int(timestamp)}.pkl'), 'wb') as f:
                pickle.dump(violation_events[-10:], f)
    except Exception as e:
        print(f"Error saving violation: {e}")

# ============================================================
# VOICE MONITOR THREAD (robust)
# ============================================================
def voice_monitor_thread():
    global voice_alert, running, last_voice_time
    def audio_callback(indata, frames, time_info, status):
        # RMS volume
        volume = float(np.sqrt(np.mean(np.square(indata))))
        if volume > voice_threshold:
            voice_queue.put(time.time())

    try:
        # Use mono, small blocksize for responsiveness
        with sd.InputStream(channels=1, samplerate=16000, blocksize=1024, callback=audio_callback):
            while running:
                try:
                    if not voice_queue.empty():
                        ts = voice_queue.get()
                        # Debounce: set alert true for 1.5s window
                        voice_alert = True
                        last_voice_time = ts
                    # Auto-clear after 1.5s
                    if time.time() - last_voice_time > 1.5:
                        voice_alert = False
                except:
                    pass
                time.sleep(0.05)
    except Exception as e:
        print("Voice stream error:", e)

# ============================================================
# CAMERA PROCTORING THREAD
# ============================================================
def camera_proctoring_thread():
    global phone_detected, book_detected, eyes_closed, head_direction, face_count
    global camera_violation, alert_active, alert_start_time, risk_level, running
    global eye_closed_start, session_risk_score, risk_history, voice_alert
    global violation_count, phone_time

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # Try to load YOLO, use dummy if not available
    try:
        net = cv2.dnn.readNet("yolov5n.onnx")
        yolo_available = True
    except:
        yolo_available = False
        print("YOLO model not found - object detection disabled")

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

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    last_violation_save = 0
    last_phone_time_update = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reset flags
        phone_detected = False
        book_detected = False
        eyes_closed = False
        camera_violation = False
        risk_level = 0
        face_count = 0

        # ================= OBJECT DETECTION ==================
        if yolo_available:
            try:
                blob = cv2.dnn.blobFromImage(frame, 1/255, (640,640), swapRB=True)
                net.setInput(blob)
                outputs = net.forward()[0]
                for det in outputs:
                    conf = float(det[4])
                    if conf < 0.3:
                        continue
                    class_probs = det[5:]
                    class_id = int(np.argmax(class_probs))
                    label = classes[class_id]
                    score = conf * class_probs[class_id]
                    if label == "cell phone" and score > 0.4:
                        phone_detected = True
                    if label == "book" and score > 0.4:
                        book_detected = True
            except:
                pass

        # ================= FACE/EYE/HEAD ==================
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        face_count = len(faces)
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_gray, 1.2, 5)

            if len(eyes) == 0:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start > 2:
                    eyes_closed = True
            else:
                eyes_closed = False
                eye_closed_start = None

            cx, cy = x+w//2, y+h//2
            if cx < width*0.4:
                head_direction = "LEFT"
            elif cx > width*0.6:
                head_direction = "RIGHT"
            elif cy < height*0.4:
                head_direction = "UP"
            elif cy > height*0.6:
                head_direction = "DOWN"
            else:
                head_direction = "CENTER"

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)

        # ================= VIOLATION & RISK ==================
        violation_score = 0
        violations_list = []

        # Major: Phone detected → count + phone_time
        if phone_detected:
            violation_score += 30
            violations_list.append("Phone Detected")
            if time.time() - last_phone_time_update > 1:
                phone_time += 1
                last_phone_time_update = time.time()
            # Count as major violation (once per save window)
        # Minor: Book detected → warning only
        if book_detected:
            violation_score += 20
            violations_list.append("Book Detected")
        # Minor: Eyes closed → warning only
        if eyes_closed:
            violation_score += 20
            violations_list.append("Eyes Closed")
        # Minor: Head movement → warning only
        if head_direction != "CENTER":
            violation_score += 10
            violations_list.append(f"Head {head_direction}")
        # Minor: Voice detected → warning only
        if voice_alert:
            violation_score += 15
            violations_list.append("Voice Detected")

        # Major: Multiple faces (>1) → count
        if face_count > 1:
            violation_score += 40
            violations_list.append(f"Multiple Faces ({face_count})")

        violation_score = min(100, violation_score)
        session_risk_score = violation_score
        risk_history.append(session_risk_score)

        # Save violation event if score > 30 and enough time passed
        if violation_score > 30 and time.time() - last_violation_save > 2:
            # Count major violations: phone or multiple faces or forbidden tab (tab counted elsewhere)
            if phone_detected or face_count > 1:
                violation_count += 1
            save_violation_event(frame, ", ".join(violations_list), violation_score)
            last_violation_save = time.time()

        if violation_score > 0:
            camera_violation = True
            if not alert_active:
                play_alert()
                alert_active = True
                alert_start_time = time.time()
        if alert_active and time.time()-alert_start_time>3:
            alert_active=False

        # ================= CAMERA UI ==================
        y0, dy = 30, 35
        overlay_texts = [
            f"Faces: {face_count}",
            f"Head: {head_direction}",
            f"Phone: {phone_detected}",
            f"Book: {book_detected}",
            f"Eyes Closed: {eyes_closed}",
            f"Voice Alert: {voice_alert}"
        ]
        for i, text in enumerate(overlay_texts):
            color = (0,255,0) if ("False" in text or "CENTER" in text or "Faces: 1" in text) else (0,0,255)
            cv2.putText(frame, text, (20,y0+i*dy), cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        # ================= RISK BAR ==================
        bar_width, bar_height = 250, 20
        bar_x, bar_y = width-270, 30
        fill = int(bar_width*session_risk_score/100)
        if session_risk_score<30:
            bar_color=(0,255,0)
        elif session_risk_score<60:
            bar_color=(0,255,255)
        else:
            bar_color=(0,0,255)
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+bar_width,bar_y+bar_height),(50,50,50),-1)
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+fill,bar_y+bar_height),bar_color,-1)
        cv2.putText(frame,"RISK",(bar_x,bar_y-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        # ================= FLASHING BANNER ==================
        if camera_violation:
            if int(time.time()*2)%2==0:
                cv2.rectangle(frame,(0,0),(width,50),(0,0,255),-1)
                cv2.putText(frame,"⚠ VIOLATION DETECTED!",(20,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.imshow("Camera Proctoring", frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            running=False
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# WINDOW MONITOR THREAD WITH TIMESTAMPED LOG + SCREENSHOT
# ============================================================
def window_monitor_thread():
    global active_window_title, window_switch_count, visited_windows
    global session_risk_score, alert_active, alert_start_time, running
    global tab_switches, violation_count

    last_window=None
    while running:
        try:
            active_window = gw.getActiveWindow()
            if active_window:
                title = active_window.title.strip() or "Unknown Window"
                if title != last_window:
                    last_window=title
                    active_window_title=title
                    window_switch_count += 1
                    if title not in visited_windows:
                        visited_windows[title]=time.strftime("%H:%M:%S")

                    # Timestamp + screenshot for every switch
                    ts_str = time.strftime("%H:%M:%S")
                    ts_file = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(screenshot_folder, f"{ts_file}_{title.replace(' ','_')}.png")
                    try:
                        ImageGrab.grab().save(screenshot_path)
                    except:
                        screenshot_path = None

                    tab_switch_log.append({
                        "time_str": ts_str,
                        "title": title,
                        "screenshot": screenshot_path
                    })

                    # Forbidden app → major violation
                    if any(app.lower() in title.lower() for app in forbidden_apps):
                        session_risk_score = min(100, session_risk_score+50)
                        tab_switches += 1
                        violation_count += 1
                        if not alert_active:
                            play_alert()
                            alert_active=True
                            alert_start_time=time.time()
        except:
            pass
        time.sleep(0.4)

# ============================================================
# EXAMINER REPLAY MODE WINDOW
# ============================================================
class ExaminerReplayWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("EXAMINER REPLAY MODE")
        self.window.geometry("1400x800")
        self.window.configure(bg="#0a0a0a")

        self.current_index = 0
        self.playback_speed = 1.0
        self.is_playing = False

        tk.Label(self.window, text="📹 EXAMINER REPLAY MODE",
                 font=("Arial", 20, "bold"), fg="white", bg="#0a0a0a").pack(pady=10)

        control_frame = tk.Frame(self.window, bg="#1a1a1a")
        control_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(control_frame, text="⏮ Previous", command=self.prev_violation,
                  bg="#3b82f6", fg="white", font=("Arial", 11, "bold"), padx=15).pack(side="left", padx=5)

        self.play_btn = tk.Button(control_frame, text="▶ Play", command=self.toggle_play,
                                  bg="#10b981", fg="white", font=("Arial", 11, "bold"), padx=15)
        self.play_btn.pack(side="left", padx=5)

        tk.Button(control_frame, text="Next ⏭", command=self.next_violation,
                  bg="#3b82f6", fg="white", font=("Arial", 11, "bold"), padx=15).pack(side="left", padx=5)

        self.speed_btn = tk.Button(control_frame, text="Speed: 1x", command=self.toggle_speed,
                                   bg="#8b5cf6", fg="white", font=("Arial", 11, "bold"), padx=15)
        self.speed_btn.pack(side="left", padx=5)

        self.info_label = tk.Label(control_frame, text="", fg="white", bg="#1a1a1a", font=("Arial", 11))
        self.info_label.pack(side="left", padx=20)

        content_frame = tk.Frame(self.window, bg="#0a0a0a")
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = tk.Frame(content_frame, bg="#1a1a1a", relief="raised", bd=2)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)
        tk.Label(left_frame, text="📷 CAMERA FEED", fg="white", bg="#1a1a1a",
                 font=("Arial", 12, "bold")).pack(pady=5)
        self.camera_label = tk.Label(left_frame, bg="black")
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=10)

        right_frame = tk.Frame(content_frame, bg="#1a1a1a", relief="raised", bd=2)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)
        tk.Label(right_frame, text="🖥 SCREEN CAPTURE", fg="white", bg="#1a1a1a",
                 font=("Arial", 12, "bold")).pack(pady=5)
        self.screenshot_label = tk.Label(right_frame, bg="black")
        self.screenshot_label.pack(fill="both", expand=True, padx=10, pady=10)

        detail_frame = tk.Frame(self.window, bg="#1a1a1a", relief="raised", bd=2)
        detail_frame.pack(fill="x", padx=10, pady=10)
        self.detail_text = tk.Text(detail_frame, height=6, bg="#0a0a0a", fg="lime",
                                   font=("Consolas", 11), wrap="word")
        self.detail_text.pack(fill="x", padx=10, pady=10)

        tk.Label(self.window, text="⚠ VIOLATION TIMELINE (Click to jump)",
                 fg="white", bg="#0a0a0a", font=("Arial", 12, "bold")).pack(pady=5)
        list_frame = tk.Frame(self.window, bg="#1a1a1a")
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.violation_listbox = tk.Listbox(list_frame, bg="#0a0a0a", fg="white",
                                            font=("Consolas", 10), yscrollcommand=scrollbar.set, height=8)
        self.violation_listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.violation_listbox.yview)
        self.violation_listbox.bind('<<ListboxSelect>>', self.on_violation_select)

        self.load_violations()
        if violation_events:
            self.show_violation(0)

    def load_violations(self):
        self.violation_listbox.delete(0, tk.END)
        for i, event in enumerate(violation_events):
            time_str = event['time_str']
            risk = event['risk_score']
            violations = event['violations']
            self.violation_listbox.insert(tk.END,
                f"#{i+1} | {time_str} | Risk: {risk}% | {violations}")

    def show_violation(self, index):
        if not violation_events or index < 0 or index >= len(violation_events):
            return
        self.current_index = index
        event = violation_events[index]
        self.info_label.config(
            text=f"Violation {index+1}/{len(violation_events)} | Time: {event['time_str']} | Risk: {event['risk_score']}%")

        frame = event['frame']
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((600, 450))
        photo = ImageTk.PhotoImage(img)
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

        screenshot = event['screenshot']
        screenshot_img = Image.fromarray(screenshot)
        screenshot_img.thumbnail((600, 450))
        screenshot_photo = ImageTk.PhotoImage(screenshot_img)
        self.screenshot_label.configure(image=screenshot_photo)
        self.screenshot_label.image = screenshot_photo

        self.detail_text.delete(1.0, tk.END)
        detail_info = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIOLATION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Timestamp       : {event['time_str']} ({event['timestamp']:.2f}s into exam)
Risk Score      : {event['risk_score']}%
Violations      : {event['violations']}
Active Window   : {event['active_window']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.detail_text.insert(1.0, detail_info)

        self.violation_listbox.selection_clear(0, tk.END)
        self.violation_listbox.selection_set(index)
        self.violation_listbox.see(index)

    def next_violation(self):
        if self.current_index < len(violation_events) - 1:
            self.show_violation(self.current_index + 1)

    def prev_violation(self):
        if self.current_index > 0:
            self.show_violation(self.current_index - 1)

    def toggle_speed(self):
        if self.playback_speed == 1.0:
            self.playback_speed = 2.0
            self.speed_btn.config(text="Speed: 2x")
        else:
            self.playback_speed = 1.0
            self.speed_btn.config(text="Speed: 1x")

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="⏸ Pause")
            self.auto_play()
        else:
            self.play_btn.config(text="▶ Play")

    def auto_play(self):
        if self.is_playing and self.current_index < len(violation_events) - 1:
            self.next_violation()
            delay = int(1000 / self.playback_speed)
            self.window.after(delay, self.auto_play)
        else:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")

    def on_violation_select(self, event):
        selection = self.violation_listbox.curselection()
        if selection:
            index = selection[0]
            self.show_violation(index)

# ============================================================
# TEST SELECTION SCREEN
# ============================================================
class TestSelectionScreen:
    def __init__(self, parent):
        self.root = parent
        self.root.title("AI Proctored Exam System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e3a8a")

        header_frame = tk.Frame(self.root, bg="#1e3a8a")
        header_frame.pack(fill="x", pady=(30, 20))
        tk.Label(header_frame, text="AI PROCTORED EXAMINATION SYSTEM",
                 font=("Arial", 28, "bold"), fg="white", bg="#1e3a8a").pack(pady=10)
        tk.Label(header_frame, text="Secure • Automated • Intelligent",
                 font=("Arial", 16), fg="#bfdbfe", bg="#1e3a8a").pack()

        card_frame = tk.Frame(self.root, bg="#1e3a8a")
        card_frame.pack(fill="both", expand=True, padx=50, pady=20)

        card_bg = "#2563eb"
        card_fg = "white"
        card_font = ("Arial", 16)

        math_card = tk.Frame(card_frame, bg=card_bg, bd=2, relief="raised", padx=20, pady=20)
        math_card.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        tk.Label(math_card, text="🧮 MATH TEST", font=("Arial", 20, "bold"), fg=card_fg, bg=card_bg).pack(pady=10)
        tk.Label(math_card, text="5 Questions | 10 Minutes", font=card_font, fg=card_fg, bg=card_bg).pack(pady=5)
        tk.Button(math_card, text="START TEST", font=("Arial", 14, "bold"), bg="#10b981", fg="white",
                  padx=20, pady=10, command=lambda: self.start_test("Math")).pack(pady=20)

        english_card = tk.Frame(card_frame, bg=card_bg, bd=2, relief="raised", padx=20, pady=20)
        english_card.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        tk.Label(english_card, text="📚 ENGLISH TEST", font=("Arial", 20, "bold"), fg=card_fg, bg=card_bg).pack(pady=10)
        tk.Label(english_card, text="5 Questions | 10 Minutes", font=card_font, fg=card_fg, bg=card_bg).pack(pady=5)
        tk.Button(english_card, text="START TEST", font=("Arial", 14, "bold"), bg="#10b981", fg="white",
                  padx=20, pady=10, command=lambda: self.start_test("English")).pack(pady=20)

        science_card = tk.Frame(card_frame, bg=card_bg, bd=2, relief="raised", padx=20, pady=20)
        science_card.grid(row=0, column=2, padx=20, pady=20, sticky="nsew")
        tk.Label(science_card, text="🔬 SCIENCE TEST", font=("Arial", 20, "bold"), fg=card_fg, bg=card_bg).pack(pady=10)
        tk.Label(science_card, text="5 Questions | 10 Minutes", font=card_font, fg=card_fg, bg=card_bg).pack(pady=5)
        tk.Button(science_card, text="START TEST", font=("Arial", 14, "bold"), bg="#10b981", fg="white",
                  padx=20, pady=10, command=lambda: self.start_test("Science")).pack(pady=20)

        card_frame.columnconfigure(0, weight=1)
        card_frame.columnconfigure(1, weight=1)
        card_frame.columnconfigure(2, weight=1)

        footer = tk.Label(self.root, text="© 2026 AI Proctoring System | Premium Hackathon Edition",
                          fg="#bfdbfe", bg="#1e3a8a", font=("Arial", 10))
        footer.pack(side="bottom", pady=10)

    def start_test(self, test_name):
        global current_test, test_questions, test_start_time, exam_start_time
        current_test = test_name
        test_questions = TEST_DATABASE[test_name]
        test_start_time = time.time()
        exam_start_time = time.time()

        # Hide selection screen
        self.root.withdraw()

        # Start test screen
        test_screen = tk.Toplevel()
        TestScreen(test_screen)

# ============================================================
# TEST SCREEN
# ============================================================
class TestScreen:
    def __init__(self, parent):
        self.window = parent
        self.window.title(f"AI Proctored Test - {current_test}")
        self.window.geometry("1200x800")
        self.window.configure(bg="#0f172a")

        header_frame = tk.Frame(self.window, bg="#0f172a")
        header_frame.pack(fill="x", pady=(20, 10))
        tk.Label(header_frame, text=f"{current_test} TEST", font=("Arial", 24, "bold"),
                 fg="white", bg="#0f172a").pack()

        status_frame = tk.Frame(self.window, bg="#1e293b", padx=20, pady=10)
        status_frame.pack(fill="x", padx=20, pady=10)
        self.timer_label = tk.Label(status_frame, text="Time Left: 10:00",
                                    font=("Arial", 14), fg="white", bg="#1e293b")
        self.timer_label.pack(side="left")
        self.violation_label = tk.Label(status_frame, text="Violations: 0",
                                        font=("Arial", 14), fg="white", bg="#1e293b")
        self.violation_label.pack(side="right")

        question_frame = tk.Frame(self.window, bg="#1e293b", padx=20, pady=20)
        question_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.question_label = tk.Label(question_frame, text="", font=("Arial", 16),
                                       fg="white", bg="#1e293b", wraplength=800, justify="left")
        self.question_label.pack(anchor="w", pady=(0, 20))

        self.option_vars = []
        self.option_buttons = []
        options_frame = tk.Frame(question_frame, bg="#1e293b")
        options_frame.pack(fill="both", expand=True)
        for i in range(4):
            var = tk.IntVar()
            btn = tk.Radiobutton(options_frame, text="", variable=var, value=i,
                                 font=("Arial", 14), fg="white", bg="#1e293b",
                                 selectcolor="#1e293b", activebackground="#1e293b",
                                 command=lambda v=i: self.select_answer(v))
            btn.pack(anchor="w", pady=10, padx=50)
            self.option_vars.append(var)
            self.option_buttons.append(btn)

        nav_frame = tk.Frame(self.window, bg="#0f172a", padx=20, pady=10)
        nav_frame.pack(fill="x", padx=20, pady=10)
        self.prev_btn = tk.Button(nav_frame, text="◀ Previous", font=("Arial", 12),
                                  state="disabled", command=self.prev_question)
        self.prev_btn.pack(side="left", padx=10)
        self.next_btn = tk.Button(nav_frame, text="Next ▶", font=("Arial", 12, "bold"),
                                  command=self.next_question)
        self.next_btn.pack(side="right", padx=10)

        self.current_question = 0
        self.load_question()
        self.update_timer()
        self.check_violations()

        # Start proctoring threads when test begins
        threading.Thread(target=camera_proctoring_thread, daemon=True).start()
        threading.Thread(target=window_monitor_thread, daemon=True).start()
        threading.Thread(target=voice_monitor_thread, daemon=True).start()

    def load_question(self):
        q = test_questions[self.current_question]
        self.question_label.config(text=f"Question {self.current_question+1}/5:\n{q['question']}")
        for i, btn in enumerate(self.option_buttons):
            btn.config(text=q['options'][i])
            self.option_vars[i].set(0)
        self.prev_btn.config(state="normal" if self.current_question > 0 else "disabled")
        self.next_btn.config(text="Next ▶" if self.current_question < 4 else "Finish Test")

    def select_answer(self, option):
        if not hasattr(self, 'user_answers'):
            self.user_answers = []
        if len(self.user_answers) <= self.current_question:
            self.user_answers.extend([None] * (self.current_question - len(self.user_answers) + 1))
        self.user_answers[self.current_question] = option

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
        global disqualified, disqualification_reason, violation_count
        # Auto-disqualify when major violations reach 3
        if violation_count >= 3:
            disqualified = True
            disqualification_reason = "Phone/Tab/Multiple-face violation limit reached (3)"
        self.violation_label.config(text=f"Violations: {violation_count}")
        if disqualified:
            messagebox.showerror("Disqualified", disqualification_reason)
            self.finish_test()
        else:
            self.window.after(1000, self.check_violations)

    def finish_test(self):
        self.window.destroy()
        generate_pdf_report(self.user_answers if hasattr(self, 'user_answers') else [])

# ============================================================
# RESULTS + PDF REPORT
# ============================================================
def generate_pdf_report(user_answers):
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "AI PROCTORED EXAM REPORT", 0, 1, 'C', 1)
    pdf.ln(6)

    # Test info
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"{current_test} Test Results", 0, 1, 'C')
    pdf.ln(4)

    # Score
    score = 0
    if user_answers:
        for i, ans in enumerate(user_answers):
            try:
                if ans == TEST_DATABASE[current_test][i]['answer']:
                    score += 1
            except:
                pass
    pdf.set_font("Arial", size=14)
    if disqualified:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"STATUS: DISQUALIFIED - {disqualification_reason}", 0, 1, 'C')
    else:
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 10, f"SCORE: {score}/5 ({score*20}%)", 0, 1, 'C')
    pdf.ln(6)

    # Violation summary
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Proctoring Violations Summary", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Phone Detected: {phone_time} seconds", 0, 1)
    pdf.cell(0, 8, f"Tab Switches (forbidden): {tab_switches}", 0, 1)
    pdf.cell(0, 8, f"Major Violations Count: {violation_count}", 0, 1)
    pdf.ln(6)

    # Tab switch audit log
    if tab_switch_log:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Tab/Window Switch Audit Log", 0, 1)
        pdf.set_font("Arial", size=11)
        for entry in tab_switch_log:
            pdf.cell(0, 8, f"[{entry['time_str']}] Switched to: {entry['title']}", 0, 1)
            if entry['screenshot'] and os.path.exists(entry['screenshot']):
                try:
                    pdf.image(entry['screenshot'], w=120)
                except:
                    pass
            pdf.ln(2)
        pdf.ln(4)

    # Detailed violations with screenshots
    if violation_events:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Detailed Violations", 0, 1)
        for i, event in enumerate(violation_events):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"Violation #{i+1} at {event['time_str']}", 0, 1)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 8, f"Risk Score: {event['risk_score']}%", 0, 1)
            pdf.cell(0, 8, f"Violations: {event['violations']}", 0, 1)
            pdf.cell(0, 8, f"Active Window: {event['active_window']}", 0, 1)
            # Save screenshot image to disk and embed
            try:
                img_path = os.path.join(screenshot_folder, f"violation_{i+1}.png")
                Image.fromarray(event['screenshot']).save(img_path)
                pdf.image(img_path, w=120)
            except:
                pass
            pdf.ln(6)

    pdf.output("test_report.pdf")

    # Show results window
    result_window = tk.Tk()
    result_window.title("Test Results")
    result_window.geometry("800x600")
    result_window.configure(bg="#0f172a")

    tk.Label(result_window, text="TEST RESULTS", font=("Arial", 24, "bold"),
             fg="white", bg="#0f172a").pack(pady=20)

    card = tk.Frame(result_window, bg="#1e293b", padx=30, pady=20)
    card.pack(fill="both", expand=True, padx=50, pady=20)

    tk.Label(card, text=f"{current_test} Test", font=("Arial", 20),
             fg="white", bg="#1e293b").pack(pady=10)

    if disqualified:
        result_text = f"DISQUALIFIED\nReason: {disqualification_reason}"
        color = "#ef4444"
    else:
        result_text = f"SCORE: {score}/5 ({score*20}%)"
        color = "#10b981"
    tk.Label(card, text=result_text, font=("Arial", 24, "bold"),
             fg=color, bg="#1e293b").pack(pady=20)

    tk.Label(card, text="Violation Summary:", font=("Arial", 16),
             fg="white", bg="#1e293b").pack(anchor="w", pady=(20, 5))
    violations_frame = tk.Frame(card, bg="#1e293b")
    violations_frame.pack(fill="x", pady=10)
    tk.Label(violations_frame, text=f"Phone Detected: {phone_time} seconds",
             font=("Arial", 12), fg="white", bg="#1e293b").pack(anchor="w")
    tk.Label(violations_frame, text=f"Tab Switches (forbidden): {tab_switches}",
             font=("Arial", 12), fg="white", bg="#1e293b").pack(anchor="w")
    tk.Label(violations_frame, text=f"Major Violations: {violation_count}",
             font=("Arial", 12), fg="white", bg="#1e293b").pack(anchor="w")

    btn_frame = tk.Frame(card, bg="#1e293b")
    btn_frame.pack(fill="x", pady=20)
    tk.Button(btn_frame, text="View Full Report", font=("Arial", 12, "bold"),
              bg="#3b82f6", fg="white", padx=20, pady=10,
              command=lambda: os.startfile("test_report.pdf")).pack(side="left", padx=10)
    tk.Button(btn_frame, text="Close", font=("Arial", 12),
              bg="#64748b", fg="white", padx=20, pady=10,
              command=lambda: result_window.destroy()).pack(side="right", padx=10)

    result_window.mainloop()

# ============================================================
# MAIN APPLICATION
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    TestSelectionScreen(root)
    root.mainloop()
    running = False
