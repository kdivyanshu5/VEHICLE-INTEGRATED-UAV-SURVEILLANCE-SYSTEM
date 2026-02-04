import os, sys, time, math, threading, hashlib
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import serial, serial.tools.list_ports
import sqlite3
import datetime
import json
import threading


# ------------------------ App credentials ------------------------
USERNAME = "admin"
PASSWORD_HASH = "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f"  # "password123"

# ------------------------ Camera / Model -------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
USE_MJPEG = True
CAMERA_MIRROR = False

MODEL_PATH = "./best.pt"

# ------------------------ Serial / Servo -------------------------
COM_PORT = "AUTO"         # "AUTO" to discover Uno, or "COMx"
BAUD = 115200
SERVO_MIN, SERVO_MAX, SERVO_CENTER = 15.0, 165.0, 90.0
SEND_THRESH_DEG = 0.5     # only send when change >= this

# ------------------------ Detection / Filtering ------------------
CONF_THRES = 0.45
NMS_IOU = 0.5
MIN_AREA_FRAC = 0.0020
MAX_AREA_FRAC = 0.65
ASPECT_MIN, ASPECT_MAX = 0.25, 3.5
TRACK_CLASS_NAME = "drone"   # set to your class label
FORCE_CLASS_ID = None        # or force integer id

# ------------------------ Tracking / Control ---------------------
LOCK_LOST_TIMEOUT = 0.8
YOLO_REFRESH_EVERY = 12
YOLO_ROI_PAD = 0.5

# TIGHTER, SAFER GAINS (prevents fast swings)
KP = 12.0
KD = 5.0
DEADBAND = 0.04
MAX_STEP_DEG = 2.0          # was 4.2 — clamp per UI tick
LOOKAHEAD_S = 0.05
EMA_ALPHA = 0.22

# extra soft limiter to avoid big jumps (deg / frame)
SOFT_RATE_LIMIT = 1.2

NO_DET_SWEEP_AFTER = 18
SWEEP_MIN, SWEEP_MAX = 25.0, 155.0
SWEEP_SPEED_DPS = 28.0

# ------------------------ Utilities ------------------------------
def get_model_path(filename="best.pt"):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, filename)

def valid_box(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    if x2 <= x1 or y2 <= y1: return False
    w, h = x2 - x1, y2 - y1
    area = (w*h)/(W*H)
    if not (MIN_AREA_FRAC <= area <= MAX_AREA_FRAC): return False
    asp = w / max(1, h)
    if not (ASPECT_MIN <= asp <= ASPECT_MAX): return False
    if x2 < 2 or y2 < 2 or x1 > W-2 or y1 > H-2: return False
    return True

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = max(0, ax2-ax1) * max(0, ay2-ay1)
    ab = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (aa + ab - inter + 1e-6)

# ------------------------ Serial Servo (no Telemetrix) -----------
class SerialServo:
    def __init__(self, port):
        self._ser = None
        self._lock = threading.Lock()
        self._last = None
        self.angle = SERVO_CENTER
        self._connect(port)
        self.home()

    def _connect(self, port):
        ports = [port] if (port not in (None, "", "AUTO")) else [p.device for p in serial.tools.list_ports.comports()]
        last = None
        for p in ports:
            for _ in range(3):
                try:
                    ser = serial.Serial(p, baudrate=BAUD, timeout=0.2)
                    time.sleep(2.2)
                    banner = ser.readline().decode(errors="ignore").strip()
                    print(f"[Serial] {p}: {banner}")
                    print(f"[Serial] Connected: {p}")
                    self._ser = ser
                    return
                except Exception as e:
                    last = e; time.sleep(0.3)
        raise RuntimeError(f"Arduino not found. {last}")

    def _send(self, text):
        with self._lock:
            if self._ser:
                self._ser.write((text + "\n").encode())

    def home(self):
        self._last = None
        self.angle = SERVO_CENTER
        self._send("HOME")

    def set_angle(self, a):
        a = float(np.clip(a, SERVO_MIN, SERVO_MAX))
        self.angle = a
        if self._last is not None and abs(a - self._last) < SEND_THRESH_DEG:
            return
        self._send(f"{a:.1f}")
        self._last = a

    def close(self):
        try:
            if self._ser: self._ser.close()
        except: pass

# ------------------------ Camera Thread (zero-lag) ----------------
class FrameGrabber(threading.Thread):
    def __init__(self, index=0):
        super().__init__(daemon=True)
        self.index = index
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera. Try another index or close other apps.")
        if USE_MJPEG:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.stop = False

    def run(self):
        while not self.stop:
            self.cap.grab()
            ok, f = self.cap.retrieve()
            if ok:
                if CAMERA_MIRROR:
                    f = cv2.flip(f, 1)
                with self.lock:
                    self.frame = f

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self.stop = True
        time.sleep(0.05)
        try: self.cap.release()
        except: pass

# ------------------------ YOLO Inference Thread -------------------
from ultralytics import YOLO
import torch

class Inference(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[YOLO] Loading {MODEL_PATH} on {self.device}")
        self.model = YOLO(get_model_path("best.pt"))
        if self.device == "cuda":
            self.model.to("cuda")
        names = getattr(self.model, "names", None)
        if FORCE_CLASS_ID is not None:
            self.ids = {int(FORCE_CLASS_ID)}
        elif names:
            if isinstance(names, dict):
                ids = [i for i, n in names.items() if str(n).lower()==TRACK_CLASS_NAME.lower()]
            else:
                ids = [i for i, n in enumerate(names) if str(n).lower()==TRACK_CLASS_NAME.lower()]
            self.ids = set(ids) if ids else None
        else:
            self.ids = None

        self.lock = threading.Lock()
        self.frame = None
        # result: (best_box, best_conf, det_count, mean_conf)
        self.result = (None, 0.0, 0, 0.0)
        self.stop = False

    def submit(self, frame):
        with self.lock:
            self.frame = frame  # overwrite old work

    def run(self):
        while not self.stop:
            f = None
            with self.lock:
                if self.frame is not None:
                    f = self.frame.copy()
                    self.frame = None
            if f is None:
                time.sleep(0.001)
                continue

            H, W = f.shape[:2]
            r = self.model.predict(
                source=f, imgsz=640, conf=CONF_THRES, iou=NMS_IOU,
                device=0 if self.device=="cuda" else None, half=(self.device=="cuda"),
                verbose=False
            )[0]

            best, best_conf = None, 0.0
            dets, confs = 0, []

            if r.boxes is not None:
                for b in r.boxes:
                    cls = int(b.cls.item()) if b.cls is not None else None
                    if self.ids is not None and (cls not in self.ids): continue
                    conf = float(b.conf.item())
                    x1,y1,x2,y2 = b.xyxy[0].detach().float().cpu().numpy()
                    bb = [float(x1), float(y1), float(x2), float(y2)]
                    if valid_box(bb, W, H):
                        dets += 1
                        confs.append(conf)
                        if conf > best_conf:
                            best, best_conf = bb, conf

            mean_conf = float(np.mean(confs)) if confs else 0.0
            with self.lock:
                self.result = (best, best_conf, dets, mean_conf)

    def get(self):
        with self.lock:
            return self.result

    def close(self):
        self.stop = True

# ------------------------ Controller (fusion + servo) -------------
class Controller:
    def __init__(self, servo: SerialServo, grabber: FrameGrabber, infer: Inference):
        self.servo = servo
        self.cam = grabber
        self.det = infer

        self.mode = "SCAN"
        self.box = None
        self.last_seen = 0.0
        self.frame_idx = 0
        self.pan_deg = SERVO_CENTER
        self.cx_s, self.vx_s = None, 0.0
        self.err_prev = 0.0
        self.pan_sign = +1
        self.sign_probe = True
        self.worse_count = 0
        self.prev_abs_e = None
        self.sweep_dir = +1
        self.no_det = 0

        # metrics
        self.frames_total = 0
        self.dets_total = 0
        self.conf_sum = 0.0
        
        # ---- saving / DB settings ----
        # Windows path where to save images (exact path you requested)
        self.save_dir = r"C:\Users\awsmd\Desktop\Main\saved_detection"
        os.makedirs(self.save_dir, exist_ok=True)
        self.db_path = os.path.join(self.save_dir, "detections.db")

        self.save_interval = 5.0     # seconds between saved detection images
        self.last_save_time = 0.0

        # init DB
        self._init_db()


    def _sweep(self, dt):
        tgt = self.pan_deg + self.sweep_dir*SWEEP_SPEED_DPS*dt
        if tgt > SWEEP_MAX: tgt=SWEEP_MAX; self.sweep_dir=-1
        elif tgt < SWEEP_MIN: tgt=SWEEP_MIN; self.sweep_dir=+1
        step = np.clip(tgt - self.pan_deg, -SOFT_RATE_LIMIT, SOFT_RATE_LIMIT)
        self.pan_deg = float(np.clip(self.pan_deg + step, SERVO_MIN, SERVO_MAX))

    def _control(self, W, tx_pred, dt):
        ex = (tx_pred - W/2)/(W/2)
        ex_dot = (ex - self.err_prev)/max(dt,1e-3)
        self.err_prev = ex
        if abs(ex) < DEADBAND: return 0.0
        u = KP*ex + KD*ex_dot
        # two-stage clamp: tight per-frame limit + soft rate limit
        u = float(np.clip(u, -MAX_STEP_DEG, +MAX_STEP_DEG))
        u = float(np.clip(u, -SOFT_RATE_LIMIT, +SOFT_RATE_LIMIT))
        return u

    def step(self):
        t0 = time.time()
        frame = self.cam.get()
        if frame is None:
            return None

        self.frame_idx += 1
        self.frames_total += 1
        H, W = frame.shape[:2]

        # submit latest frame; read last result
        self.det.submit(frame)
        best_box, best_conf, dets, mean_conf = self.det.get()

        # accumulate metrics
        self.dets_total += dets
        self.conf_sum += mean_conf
        
        # Save annotated detection image every self.save_interval seconds
        now = time.time()
        if dets > 0 and (now - self.last_save_time) >= self.save_interval:
            # annotate a copy
            save_frame = frame.copy()   # BGR

            # timestamp / text
            ts = datetime.datetime.now()
            date_str = ts.strftime("%Y-%m-%d")
            time_str = ts.strftime("%H:%M:%S")
            text = f"Avg Conf: {mean_conf:.2f}   Date: {date_str}   Time: {time_str}"

            # draw background rectangle for readability
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(save_frame, (8, 8), (10 + tw, 12 + th), (0, 0, 0), -1)
            cv2.putText(save_frame, text, (10, 10 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # draw detection boxes (optional) — reuse current self.box if present
            if self.box is not None and valid_box(self.box, W, H):
                x1, y1, x2, y2 = map(int, self.box)
                cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(save_frame, ((x1+x2)//2, (y1+y2)//2), 4, (0,255,0), -1)

            filename = ts.strftime("%Y%m%d_%H%M%S") + f"_c{int(mean_conf*100):03d}.jpg"
            out_path = os.path.join(self.save_dir, filename)

            # write in background to avoid blocking UI
            t = threading.Thread(
                target=self._save_image_and_record,
                args=(save_frame, out_path, mean_conf, dets, ts),
                daemon=True
            )
            t.start()

            self.last_save_time = now

        if best_box is None:
            # SCAN mode: NO BOX DRAW — explicit clear
            self.mode = "SCAN"
            self.box = None
            self.no_det += 1
            if self.no_det > NO_DET_SWEEP_AFTER:
                self._sweep(1/60.0)
        else:
            self.no_det = 0
            self.mode = "TRACK"
            self.box = best_box
            self.last_seen = time.time()

            bx = 0.5*(best_box[0] + best_box[2])
            dt = 1/60.0
            if self.cx_s is None: self.cx_s = bx; self.vx_s=0.0
            else:
                dx = bx - self.cx_s
                self.cx_s = (1-EMA_ALPHA)*self.cx_s + EMA_ALPHA*bx
                self.vx_s = 0.7*self.vx_s + 0.3*(dx/dt)

            tx = self.cx_s + self.vx_s*LOOKAHEAD_S
            delta = self._control(W, tx, dt)

            # one-time auto sign correction if early movement is wrong
            if self.sign_probe:
                abs_e = abs((tx - W/2)/(W/2))
                if self.prev_abs_e is not None and abs_e > self.prev_abs_e + 0.01:
                    self.worse_count += 1
                else:
                    self.worse_count = 0
                self.prev_abs_e = abs_e
                if self.worse_count >= 4:
                    self.pan_sign *= -1
                    print(f"[PAN] auto flip -> {self.pan_sign:+d}")
                    self.sign_probe = False

            self.pan_deg = float(np.clip(self.pan_deg + self.pan_sign*delta, SERVO_MIN, SERVO_MAX))

        # send to servo (rate-limited inside class)
        self.servo.set_angle(self.pan_deg)

        # HUD — only draw in TRACK mode
        cv2.drawMarker(frame, (W//2, H//2), (255,255,255), cv2.MARKER_CROSS, 20, 1)
        if self.mode == "TRACK" and self.box is not None and valid_box(self.box, W, H):
            x1,y1,x2,y2 = map(int, self.box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 4, (0,255,0), -1)
        return frame

    def metrics(self):
        # averages since app start
        avg_dets = (self.dets_total / max(1, self.frames_total))
        avg_conf = (self.conf_sum / max(1, self.frames_total))
        return avg_dets, avg_conf
    
    #DB Helper
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                avg_conf REAL,
                det_count INTEGER,
                datetime TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _save_image_and_record(self, img_bgr, out_path, avg_conf, det_count, timestamp):
        """Writes image + records metadata (runs in background thread)."""
        try:
            # write image (BGR)
            cv2.imwrite(out_path, img_bgr)
            # record metadata
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "INSERT INTO detections (filename, avg_conf, det_count, datetime) VALUES (?, ?, ?, ?)",
                (os.path.basename(out_path), float(avg_conf), int(det_count), timestamp.isoformat())
            )
            conn.commit()
            conn.close()
            print(f"[SAVE] {out_path} ({avg_conf:.2f}, {det_count})")
        except Exception as e:
            print("[SAVE-ERR]", e)


# ------------------------ UI (your layout, with hover) -----------
class LoginWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Login - YOLO Drone Tracker")
        self.geometry("420x300"); self.configure(bg="#0D1B2A")
        self._center(); self._build()
    def _center(self):
        self.update_idletasks()
        w,h=420,300; sw,sh=self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//3}")
    def _build(self):
        style=ttk.Style(); style.theme_use("clam")
        style.configure("Rounded.TButton", font=("Segoe UI",11,"bold"),
                        background="#415A77", foreground="white", padding=8, relief="flat", borderwidth=0)
        style.map("Rounded.TButton", background=[("active","#1B263B")])
        card=tk.Frame(self, bg="white", padx=30, pady=25); card.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(card, text="Drone Tracker Login", font=("Segoe UI",15,"bold"), fg="#1B263B", bg="white").pack(pady=(0,15))
        ttk.Label(card, text="Username:").pack(anchor="w"); self.username_entry=ttk.Entry(card); self.username_entry.pack(fill="x", pady=(0,8))
        ttk.Label(card, text="Password:").pack(anchor="w"); self.password_entry=ttk.Entry(card, show="*"); self.password_entry.pack(fill="x", pady=(0,12))
        ttk.Button(card, text="Login", style="Rounded.TButton", command=self._login).pack(pady=(8,0))
        self.username_entry.focus_set()
    def _login(self):
        u=self.username_entry.get().strip(); p=self.username_entry.get().strip() if False else self.password_entry.get().strip()
        if u==USERNAME and hashlib.sha256(p.encode()).hexdigest()==PASSWORD_HASH:
            self.destroy(); MainWindow()
        else:
            messagebox.showerror("Login Failed","Invalid username or password.")

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Live Drone Tracker"); self.geometry("1400x820")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # themes
        self.themes = {
            "dark": {"bg_main":"#0D1B2A","bg_sidebar":"#1B263B","bg_top":"#415A77","bg_panel":"#E0E1DD",
                     "fg_text":"white","fg_label":"#1B263B","highlight":"#2E4372","hover":"#31486A","accent":"#5FA8D3"},
            "light":{"bg_main":"#E8F3E8","bg_sidebar":"#C7E2C7","bg_top":"#7FB77E","bg_panel":"#F5FFF5",
                     "fg_text":"#203B20","fg_label":"#2F4F2F","highlight":"#9FD89F","hover":"#B6E0B5","accent":"#68B267"}
        }
        self.current_theme="dark"
        self.active_tab=tk.StringVar(value="Live Feed")

        # hardware threads
        try:
            self.servo = SerialServo(COM_PORT)
            self.grabber = FrameGrabber(CAM_INDEX); self.grabber.start()
            self.infer = Inference(); self.infer.start()
            self.ctrl = Controller(self.servo, self.grabber, self.infer)
            import torch
            if not torch.cuda.is_available():
                messagebox.showwarning("CUDA not available",
                                       "Running on CPU — install CUDA-enabled PyTorch for best performance.")
        except Exception as e:
            messagebox.showerror("Startup Error", f"{e}"); self.destroy(); return

        self._build()
        self._after = self.after(1, self._update_frame)

    def _hoverize(self, widget, normal_bg, hover_bg):
        widget.bind("<Enter>", lambda e: widget.configure(bg=hover_bg))
        widget.bind("<Leave>", lambda e: widget.configure(bg=normal_bg))

    def _build(self):
        th=self.themes[self.current_theme]
        self.configure(bg=th["bg_main"])
        for w in self.winfo_children(): w.destroy()

        # sidebar
        sb=tk.Frame(self, width=200, bg=th["bg_sidebar"]); sb.pack(side="left", fill="y")
        tk.Label(sb, text="YOLO Tracker", fg=th["fg_text"], bg=th["bg_sidebar"], font=("Segoe UI",18,"bold")).pack(pady=25)
        for label in ["Live Feed","Settings","Help"]:
            is_active = (self.active_tab.get()==label)
            bg = th["highlight"] if is_active else th["bg_sidebar"]
            item=tk.Label(sb, text=label, fg=th["fg_text"], bg=bg, font=("Segoe UI",12),
                          padx=16, pady=12, cursor="hand2")
            item.pack(fill="x", pady=2)
            self._hoverize(item, bg, th["hover"] if not is_active else bg)
            item.bind("<Button-1>", lambda e,n=label: (self.active_tab.set(n), self._build()))

        # top bar
        top=tk.Frame(self, height=60, bg=th["bg_top"]); top.pack(side="top", fill="x")
        tk.Label(top, text="YOLO Live Drone Tracker", fg="white", bg=th["bg_top"],
                 font=("Segoe UI",14,"bold")).pack(side="left", padx=20)
        logout_btn=tk.Button(top, text="Logout", command=self._logout, fg="white", bg=th["bg_sidebar"],
                             relief="flat", font=("Segoe UI",11,"bold"), padx=14, pady=6,
                             activeforeground="white", activebackground=th["hover"], cursor="hand2")
        logout_btn.pack(side="right", padx=20, pady=10)
        self._hoverize(logout_btn, th["bg_sidebar"], th["hover"])

        # content
        self.content=tk.Frame(self, bg=th["bg_main"]); self.content.pack(fill="both", expand=True)
        if self.active_tab.get()=="Live Feed": self._tab_live(th)
        elif self.active_tab.get()=="Settings": self._tab_settings(th)
        else: self._tab_help(th)

    def _tab_live(self, th):
        self.video_panel=tk.Frame(self.content, bg=th["bg_panel"], bd=2, relief="ridge")
        self.video_panel.place(relx=0.17, rely=0.06, relwidth=0.68, relheight=0.88)
        self.video_label=tk.Label(self.video_panel, bg=th["bg_panel"]); self.video_label.pack(expand=True, fill="both")

        mp=tk.Frame(self.content, bg=th["bg_panel"], bd=2, relief="ridge")
        mp.place(relx=0.87, rely=0.06, relwidth=0.11, relheight=0.88)
        tk.Label(mp, text="Metrics", font=("Segoe UI",14,"bold"),
                 fg=th["fg_label"], bg=th["bg_panel"]).pack(pady=10)
        self.mode_lbl=tk.Label(mp, text="Mode: SCAN", font=("Segoe UI",12),
                               bg=th["bg_panel"], fg=th["fg_label"]); self.mode_lbl.pack(pady=6)
        self.fps_lbl=tk.Label(mp, text="FPS: 0.0", font=("Segoe UI",12),
                              bg=th["bg_panel"], fg=th["fg_label"]); self.fps_lbl.pack(pady=6)
        self.avgdet_lbl=tk.Label(mp, text="Avg Dets: 0.00", font=("Segoe UI",12),
                                 bg=th["bg_panel"], fg=th["fg_label"]); self.avgdet_lbl.pack(pady=6)
        self.avgconf_lbl=tk.Label(mp, text="Avg Conf.: 0.00", font=("Segoe UI",12),
                                  bg=th["bg_panel"], fg=th["fg_label"]); self.avgconf_lbl.pack(pady=6)

        test_btn=tk.Button(mp, text="Test Servo", command=self._test_servo,
                           fg=th["fg_label"], bg=th["accent"], relief="flat",
                           font=("Segoe UI",11,"bold"), padx=10, pady=6,
                           activeforeground="white", activebackground=th["hover"], cursor="hand2")
        test_btn.pack(pady=20)
        self._hoverize(test_btn, th["accent"], th["hover"])

    def _tab_settings(self, th):
        f=tk.Frame(self.content, bg=th["bg_panel"], bd=2, relief="ridge")
        f.place(relx=0.25, rely=0.15, relwidth=0.5, relheight=0.7)
        tk.Label(f, text="Settings", font=("Segoe UI",16,"bold"),
                 fg=th["fg_label"], bg=th["bg_panel"]).pack(pady=20)
        tk.Label(f, text="Choose Theme:", font=("Segoe UI",12),
                 bg=th["bg_panel"], fg=th["fg_label"]).pack(pady=(10,4))
        theme_var=tk.StringVar(value=self.current_theme)
        ttk.Radiobutton(f, text="Dark Mode", variable=theme_var, value="dark",
                        command=lambda:self._change_theme(theme_var.get())).pack(pady=4)
        ttk.Radiobutton(f, text="Light Mode", variable=theme_var, value="light",
                        command=lambda:self._change_theme(theme_var.get())).pack(pady=4)

    def _tab_help(self, th):
        f=tk.Frame(self.content, bg=th["bg_panel"], bd=2, relief="ridge")
        f.place(relx=0.25, rely=0.15, relwidth=0.5, relheight=0.7)
        tk.Label(f, text="Help & Tips", font=("Segoe UI",16,"bold"),
                 fg=th["fg_label"], bg=th["bg_panel"]).pack(pady=20)
        tk.Frame(f, bg=th["fg_label"], height=1).pack(fill="x", padx=30, pady=(0,20))
        tips=(
            "• If pan direction is wrong, set CAMERA_MIRROR=True (or auto-correct flips once).\n"
            "• Lower CONF_THRES (0.35) or MIN_AREA_FRAC (0.001) for small/far drones.\n"
            "• Install CUDA PyTorch to use RTX 3060.\n"
            "• Close other apps using the webcam."
        )
        tk.Label(f, text=tips, bg=th["bg_panel"], fg=th["fg_label"],
                 font=("Segoe UI",11), justify="left", wraplength=520).pack(padx=20, pady=10)

    def _test_servo(self):
        for a in [70, 110, 90]:
            self.ctrl.pan_deg = a
            self.servo.set_angle(a)
            self.update()
            time.sleep(0.5)

    def _change_theme(self, name):
        self.current_theme=name; self._build()

    def _fit(self, frame):
        pw=max(self.video_panel.winfo_width(), 640)
        ph=max(self.video_panel.winfo_height(), 360)
        h,w=frame.shape[:2]; s=min(pw/w, ph/h)
        return cv2.resize(frame,(int(w*s),int(h*s)), interpolation=cv2.INTER_AREA)

    def _update_frame(self):
        t0 = time.time()
        frame = self.ctrl.step()
        if frame is not None and self.video_label.winfo_exists():
            frame = self._fit(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # update metrics panel
            self.mode_lbl.configure(text=f"Mode: {self.ctrl.mode}")
            fps = 1.0 / max(1e-3, (time.time() - t0))
            self.fps_lbl.configure(text=f"FPS: {fps:.1f}")
            avg_dets, avg_conf = self.ctrl.metrics()
            self.avgdet_lbl.configure(text=f"Avg Dets: {avg_dets:.2f}")
            self.avgconf_lbl.configure(text=f"Avg Conf.: {avg_conf:.2f}")

        self._after = self.after(12, self._update_frame)  # ~80 Hz UI scheduling

    def _logout(self):
        try: self.after_cancel(self._after)
        except: pass
        try:
            self.grabber.close(); self.infer.close()
        except: pass
        try:
            self.servo.set_angle(SERVO_CENTER)
        except: pass
        self.destroy(); LoginWindow()

    def _on_close(self):
        try: self.after_cancel(self._after)
        except: pass
        try:
            self.grabber.close(); self.infer.close()
        except: pass
        try:
            self.servo.set_angle(SERVO_CENTER)
        except: pass
        self.destroy()

# ------------------------ Entrypoint ------------------------------
class LoginWindow(LoginWindow): pass

if __name__ == "__main__":
    LoginWindow().mainloop()
