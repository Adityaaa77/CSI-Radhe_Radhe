# ============================================================
# AI PROCTORING SYSTEM – FULL PREMIUM HACKATHON VERSION
# Features:
# - Camera Proctoring: Face, Eyes, Head, Phone/Book Detection
# - Window/Tab Monitoring with Screenshot Capture
# - Voice/Noise Detection
# - Session Risk Scoring
# - Live Behavior Trend Graph
# - Flashing Violation Banner & Animated Risk Bar
# - Attractive Tkinter Dashboard
# ============================================================

import cv2
import time
import numpy as np
import winsound
import threading
import tkinter as tk
import pygetwindow as gw
from PIL import ImageGrab
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sounddevice as sd
import queue

# ============================================================
# GLOBAL STATE
# ============================================================
running = True

# Camera detections
phone_detected = False
book_detected = False
eyes_closed = False
head_direction = "CENTER"
camera_violation = False
risk_level = 0  # 0=Safe,1=Mild,2=Serious
eye_closed_start = None

# Window monitoring
active_window_title = "Initializing..."
window_switch_count = 0
visited_windows = {}  # title -> first seen time
forbidden_apps = ["Discord", "Slack", "WhatsApp", "VSCode", "Chrome"]
screenshot_folder = "screenshots"
os.makedirs(screenshot_folder, exist_ok=True)

# Voice detection
voice_queue = queue.Queue()
voice_threshold = 0.03
voice_alert = False

# Alert
alert_active = False
alert_start_time = 0

# Exam start
exam_start_time = time.time()

# Session risk
session_risk_score = 0
risk_history = []

# ============================================================
# ALERT FUNCTION
# ============================================================
def play_alert():
    winsound.Beep(1000, 800)

# ============================================================
# VOICE MONITOR THREAD
# ============================================================
def voice_monitor_thread():
    global voice_alert, running
    def audio_callback(indata, frames, time_, status):
        volume_norm = np.linalg.norm(indata)/frames
        if volume_norm > voice_threshold:
            voice_queue.put(True)

    with sd.InputStream(callback=audio_callback):
        while running:
            try:
                if not voice_queue.empty():
                    voice_alert = True
                    voice_queue.get()
                else:
                    voice_alert = False
            except:
                pass
            time.sleep(0.1)

# ============================================================
# CAMERA PROCTORING THREAD
# ============================================================
def camera_proctoring_thread():
    global phone_detected, book_detected, eyes_closed, head_direction
    global camera_violation, alert_active, alert_start_time, risk_level, running
    global eye_closed_start, session_risk_score, risk_history, voice_alert

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    net = cv2.dnn.readNet("yolov5n.onnx")

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

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reset flags
        phone_detected = False
        book_detected = False
        eyes_closed = False
        camera_violation = False
        risk_level = 0

        # ================= OBJECT DETECTION ==================
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

        # ================= FACE/EYE/HEAD ==================
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 5)

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
        if phone_detected:
            violation_score += 30
        if book_detected:
            violation_score += 20
        if eyes_closed:
            violation_score += 20
        if head_direction != "CENTER":
            violation_score += 10
        if voice_alert:
            violation_score += 15

        violation_score = min(100, violation_score)
        session_risk_score = violation_score
        risk_history.append(session_risk_score)

        if violation_score > 0:
            camera_violation = True
            if not alert_active:
                play_alert()
                alert_active = True
                alert_start_time = time.time()
        if alert_active and time.time()-alert_start_time>5:
            alert_active=False

        # ================= CAMERA UI ==================
        y0, dy = 30, 35
        for i, text in enumerate([f"Head: {head_direction}",
                                  f"Phone: {phone_detected}",
                                  f"Book: {book_detected}",
                                  f"Eyes Closed: {eyes_closed}",
                                  f"Voice Alert: {voice_alert}"]):
            color = (0,255,0) if "False" in text or "CENTER" in text else (0,0,255)
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
# WINDOW MONITOR THREAD WITH SCREENSHOT
# ============================================================
def window_monitor_thread():
    global active_window_title, window_switch_count, visited_windows
    global session_risk_score, alert_active, alert_start_time, running
    last_window=None
    while running:
        try:
            active_window = gw.getActiveWindow()
            if active_window:
                title = active_window.title.strip() or "Unknown Window"
                if title != last_window:
                    last_window=title
                    active_window_title=title
                    if title not in visited_windows:
                        visited_windows[title]=time.strftime("%H:%M:%S")
                    if any(app.lower() in title.lower() for app in forbidden_apps):
                        session_risk_score = min(100, session_risk_score+50)
                        if not alert_active:
                            play_alert()
                            alert_active=True
                            alert_start_time=time.time()
                        # Screenshot
                        ts=time.strftime("%Y%m%d_%H%M%S")
                        ImageGrab.grab().save(os.path.join(screenshot_folder,f"{ts}_{title}.png"))
        except:
            pass
        time.sleep(0.5)

# ============================================================
# DASHBOARD UI WITH LIVE GRAPH
# ============================================================
root = tk.Tk()
root.title("AI PROCTORING DASHBOARD")
root.geometry("1000x650")
root.configure(bg="#0f172a")

tk.Label(root,text="AI SMART PROCTORING SYSTEM",font=("Arial",22,"bold"),fg="white",bg="#0f172a").pack(pady=10)
status_box = tk.Label(root,font=("Consolas",12),fg="white",bg="#1e293b",justify="left",padx=20,pady=20)
status_box.pack(fill="both",expand=True,padx=20,pady=10)

tk.Label(root,text="Visited Windows (first seen time)",fg="white",bg="#0f172a",font=("Arial",12,"bold")).pack()
window_list_box=tk.Text(root,height=6,bg="#1e293b",fg="lime",font=("Consolas",11))
window_list_box.pack(fill="both",padx=20,pady=10)

fig=Figure(figsize=(8,3))
ax=fig.add_subplot(111)
ax.set_title("Session Risk Trend")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Risk Score")
ax.set_ylim(0,100)
line,=ax.plot([],[],color='red',linewidth=2)
canvas=FigureCanvasTkAgg(fig,master=root)
canvas.get_tk_widget().pack(fill="both",padx=20,pady=10)

footer=tk.Label(root,text="Monitoring camera, system, and voice detection live",fg="gray",bg="#0f172a")
footer.pack(pady=5)

def update_ui():
    elapsed=int(time.time()-exam_start_time)
    status_box.config(text=
        f"EXAM TIME       : {elapsed}s\n"
        f"ACTIVE WINDOW   : {active_window_title}\n"
        f"WINDOW SWITCHES : {window_switch_count}\n"
        f"HEAD DIRECTION  : {head_direction}\n"
        f"PHONE DETECTED  : {phone_detected}\n"
        f"BOOK DETECTED   : {book_detected}\n"
        f"EYES CLOSED     : {eyes_closed}\n"
        f"VOICE ALERT     : {voice_alert}\n"
        f"SESSION RISK    : {session_risk_score}"
    )
    window_list_box.delete(1.0,tk.END)
    for w,t in visited_windows.items():
        window_list_box.insert(tk.END,f"{w} (first seen: {t})\n")
    times=list(range(len(risk_history)))
    line.set_data(times,risk_history)
    ax.set_xlim(0,max(10,len(risk_history)))
    canvas.draw()
    root.after(500,update_ui)

# ============================================================
# START THREADS
# ============================================================
threading.Thread(target=camera_proctoring_thread,daemon=True).start()
threading.Thread(target=window_monitor_thread,daemon=True).start()
threading.Thread(target=voice_monitor_thread,daemon=True).start()

update_ui()
root.mainloop()
running=False
