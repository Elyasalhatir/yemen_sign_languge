# -*- coding: utf-8 -*-
"""
🌟 ULTIMATE Sign Language Recognition - Production Ready 🌟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ Real-time Hand Tracking with MediaPipe
🎯 Accurate Prediction (EXACT match with training)
🗣️ Working Arabic & English TTS
🎨 Beautiful Modern UI
📊 Live Confidence Visualization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import cv2
import numpy as np
import joblib
import pickle
import time
import threading
from collections import deque, Counter
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import mediapipe as mp

# ═══════════════════════════════════════════════════════════
# 🎯 CONFIGURATION
# ═══════════════════════════════════════════════════════════
MODEL_DIR = Path(r"C:\Users\elyas\Desktop\final_video_sign\FINAL_SIGNLANGUGE\mlp_dynamic106_fixed")
MODEL_PATH = MODEL_DIR / "mlp_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
INV_MAP_PATH = MODEL_DIR / "idx_to_word.pkl"

# Prediction Settings
SMOOTHING_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.50
COOLDOWN_FRAMES = 10
DYN_WINDOW = 8

# Word Database
WORD_DB = {
    'ANCLE': {'hand': 'both', 'arabic': 'عم'},
    'BROTHER': {'hand': 'left', 'arabic': 'الأخ'},
    'EASY': {'hand': 'left', 'arabic': 'سهل'},
    'ENGAGEMENT': {'hand': 'left', 'arabic': 'الخطوبة'},
    'FAMILY': {'hand': 'both', 'arabic': 'العائلة'},
    'FATHER': {'hand': 'left', 'arabic': 'الأب'},
    'HIM': {'hand': 'left', 'arabic': 'هو'},
    'HOUR': {'hand': 'both', 'arabic': 'ساعه'},
    'HOW': {'hand': 'left', 'arabic': 'كيف'},
    'MINE': {'hand': 'left', 'arabic': 'حقي'},
    'MOTHER': {'hand': 'left', 'arabic': 'الأم'},
    'MOUNTH': {'hand': 'left', 'arabic': 'الفم'},
    'NAME': {'hand': 'left', 'arabic': 'الاسم'},
    'NO': {'hand': 'left', 'arabic': 'لا'},
    'PERCENTAGE': {'hand': 'left', 'arabic': 'النسبة'},
    'RAEDY': {'hand': 'both', 'arabic': 'جاهز'},
    'WHAT': {'hand': 'left', 'arabic': 'ماذا'},
    'WHEN': {'hand': 'left', 'arabic': 'متى'},
    'WHERE': {'hand': 'left', 'arabic': 'أين'},
    'YES': {'hand': 'left', 'arabic': 'نعم'},
    'cancer': {'hand': 'both', 'arabic': 'سرطان'},
    'cold': {'hand': 'right', 'arabic': 'برد'},
    'eat': {'hand': 'right', 'arabic': 'أكل'},
    'face': {'hand': 'right', 'arabic': 'وجه'},
    'fever': {'hand': 'right', 'arabic': 'حمى'},
    'loss of hair': {'hand': 'right', 'arabic': 'تساقط الشعر'},
    'medicine': {'hand': 'right', 'arabic': 'دواء'},
    'muscle': {'hand': 'both', 'arabic': 'عضلة'}
}

# ═══════════════════════════════════════════════════════════
# 🛠️ FEATURE EXTRACTION (EXACT MATCH WITH TRAINING)
# ═══════════════════════════════════════════════════════════
def normalize_keypoints(kpts):
    if kpts is None or len(kpts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    center = np.mean(kpts, axis=0)
    centered = kpts - center
    max_dist = np.max(np.linalg.norm(centered, axis=1)) + 1e-6
    return centered / max_dist

def compute_finger_angles_from_pts(pts):
    """EXACT copy from training code"""
    tips = [4, 8, 12, 16, 20]
    dips = [3, 7, 11, 15, 19]
    mcps = [2, 5, 9, 13, 17]
    angles = []
    for idx in range(1, 5):
        try:
            a, b, c = np.array(pts[mcps[idx]]), np.array(pts[dips[idx]]), np.array(pts[tips[idx]])
            cosine = np.dot(b - a, c - b) / (np.linalg.norm(b - a) * np.linalg.norm(c - b) + 1e-6)
            angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
        except:
            angles.append(0.0)
    return np.array(angles, dtype=np.float32)

def build_static_from_pts(pts, arm, label):
    """EXACT copy from training code"""
    hand_vec = normalize_keypoints(pts).flatten()
    angs = compute_finger_angles_from_pts(pts)
    keys = ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'] if label == 'left' else ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
    arm_pts = [[arm.get(k, {'x': 0, 'y': 0, 'z': 0})[key] if key in arm.get(k, {}) else 0.0 for key in ['x', 'y', 'z']] for k in keys]
    arm_vec = normalize_keypoints(np.array(arm_pts, dtype=np.float32)).flatten()
    nose_pt = np.array([arm.get('NOSE', {'x': 0, 'y': 0, 'z': 0})['x'], arm.get('NOSE', {'x': 0, 'y': 0, 'z': 0})['y'],
                        arm.get('NOSE', {'x': 0, 'y': 0, 'z': 0})['z']], dtype=np.float32)
    wrist = pts[0]
    dist_vec = wrist - nose_pt
    static = np.concatenate([hand_vec, angs, arm_vec, dist_vec])
    
    # Pad to 82
    STATIC_LEN = 82
    if static.shape[0] != STATIC_LEN:
        if static.shape[0] < STATIC_LEN:
            static = np.concatenate([static, np.zeros(STATIC_LEN - static.shape[0], dtype=np.float32)])
        else:
            static = static[:STATIC_LEN]
    return static

def compute_dyn_stats_from_buffer(buffer_deque):
    """Compute 21-D dynamic stats"""
    if len(buffer_deque) == 0:
        return np.zeros(21, dtype=np.float32)
    
    arr = np.stack(list(buffer_deque), axis=0)
    mean_val = np.mean(arr, axis=0)
    std_val = np.std(arr, axis=0)
    min_val = np.min(arr, axis=0)
    max_val = np.max(arr, axis=0)
    range_val = max_val - min_val
    median_val = np.median(arr, axis=0)
    var_val = np.var(arr, axis=0)
    
    stats = np.concatenate([mean_val, std_val, min_val, max_val, range_val, median_val, var_val])
    return stats.astype(np.float32)

def combine_feats(left_full, right_full):
    """EXACT copy from training code"""
    return (left_full + right_full) / 2.0

# ═══════════════════════════════════════════════════════════
# 🎤 TTS WORKER (FIXED)
# ═══════════════════════════════════════════════════════════
class TTSWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.queue = deque()
        self.running = True
        self.engine = None
        self.voices = {}
        
        # Initialize in separate thread to avoid blocking
        self.init_thread = threading.Thread(target=self._init_engine, daemon=True)
        self.init_thread.start()
        
        # Processing thread
        self.process_thread = threading.Thread(target=self._process, daemon=True)
        self.process_thread.start()

    def _init_engine(self):
        """Initialize TTS engine in background"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 140)
            self.engine.setProperty('volume', 1.0)
            
            for v in self.engine.getProperty('voices'):
                name_lower = v.name.lower()
                id_lower = v.id.lower()
                if "arabic" in name_lower or "ar" in id_lower:
                    self.voices['ar'] = v.id
                if "english" in name_lower or "david" in name_lower or "zira" in name_lower:
                    self.voices['en'] = v.id
            
            print(f"[TTS] Ready. Voices: {list(self.voices.keys())}")
        except Exception as e:
            print(f"[TTS] Init failed: {e}")
            self.error.emit(str(e))

    def speak(self, text, lang='en'):
        """Add text to speech queue"""
        self.queue.append((text, lang))

    def _process(self):
        """Process speech queue"""
        while self.running:
            if self.queue and self.engine:
                text, lang = self.queue.popleft()
                try:
                    voice_id = self.voices.get(lang, self.voices.get('en'))
                    if voice_id:
                        self.engine.setProperty('voice', voice_id)
                    
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.finished.emit()
                except Exception as e:
                    print(f"[TTS] Speak error: {e}")
                    self.error.emit(str(e))
            else:
                time.sleep(0.05)

    def stop(self):
        self.running = False
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass

# ═══════════════════════════════════════════════════════════
# 🧠 SIGN PREDICTOR
# ═══════════════════════════════════════════════════════════
class SignPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.idx2word = None
        self.n_features = 0
        self.load_model()
        
        self.buffer = deque(maxlen=SMOOTHING_FRAMES)
        self.conf_buffer = deque(maxlen=SMOOTHING_FRAMES)
        self.cooldown = 0
        self.last_word = None
        
    def load_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.idx2word = pickle.load(open(INV_MAP_PATH, "rb"))
            self.n_features = self.scaler.n_features_in_
            print(f"✅ Model: {len(self.idx2word)} classes, {self.n_features} features")
        except Exception as e:
            print(f"❌ Model error: {e}")

    def predict(self, left_feats, right_feats):
        if not self.model:
            return None, 0.0, "Model not loaded"
        
        if self.cooldown > 0:
            self.cooldown -= 1
            return None, 0.0, f"Cooldown: {self.cooldown}"
        
        def process_hand(static, dynamic):
            full = np.concatenate([static, dynamic])
            
            if len(full) < self.n_features:
                full = np.concatenate([full, np.zeros(self.n_features - len(full))])
            full = full[:self.n_features].reshape(1, -1)
            
            try:
                X = self.scaler.transform(full)
                probs = self.model.predict_proba(X)[0]
                idx = np.argmax(probs)
                return self.idx2word[idx], probs[idx]
            except Exception as e:
                print(f"[PRED ERROR] {e}")
                return None, 0.0

        candidates = []
        
        if left_feats:
            static, dynamic = left_feats
            w, c = process_hand(static, dynamic)
            if w:
                candidates.append((w, c, 'left'))
            
        if right_feats:
            static, dynamic = right_feats
            w, c = process_hand(static, dynamic)
            if w:
                candidates.append((w, c, 'right'))
            
        if left_feats and right_feats:
            l_s, l_d = left_feats
            r_s, r_d = right_feats
            # EXACT match with training: combine_feats
            avg_s = combine_feats(l_s, r_s)
            avg_d = combine_feats(l_d, r_d)
            w, c = process_hand(avg_s, avg_d)
            if w:
                candidates.append((w, c, 'both'))

        if not candidates:
            return None, 0.0, "No hand"

        # Apply hand constraints (EXACT match with training)
        best_word = None
        best_conf = 0.0
        debug_info = []
        
        for word, conf, hand_used in candidates:
            required = WORD_DB.get(word, {}).get('hand', 'both')
            penalty = 1.0
            
            # EXACT match with training logic
            if required == 'left' and hand_used == 'right':
                penalty = 0.2
            elif required == 'right' and hand_used == 'left':
                penalty = 0.2
            elif required == 'both' and hand_used != 'both':
                penalty = 0.5
            
            adjusted = conf * penalty
            debug_info.append(f"{word}:{adjusted:.2f}")
            
            if adjusted > best_conf:
                best_conf = adjusted
                best_word = word

        # Smoothing
        self.buffer.append(best_word)
        self.conf_buffer.append(best_conf)
        
        debug_str = " | ".join(debug_info[:3])
        
        # Trigger
        if len(self.buffer) >= SMOOTHING_FRAMES:
            counts = Counter(self.buffer)
            most_common, count = counts.most_common(1)[0]
            avg_conf = np.mean(list(self.conf_buffer))
            
            if count >= SMOOTHING_FRAMES * 0.5 and avg_conf > CONFIDENCE_THRESHOLD:
                if most_common != self.last_word:
                    self.last_word = most_common
                    self.cooldown = COOLDOWN_FRAMES
                    self.buffer.clear()
                    self.conf_buffer.clear()
                    return most_common, avg_conf, debug_str
        
        return None, best_conf, debug_str

# ═══════════════════════════════════════════════════════════
# 📹 VIDEO THREAD
# ═══════════════════════════════════════════════════════════
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, str, float)
    debug_signal = pyqtSignal(str, float)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.predictor = SignPredictor()
        
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        prev_wrist = {'left': None, 'right': None}
        vel_buffers = {'left': deque(maxlen=DYN_WINDOW), 'right': deque(maxlen=DYN_WINDOW)}
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process
            pose_res = mp_pose.process(rgb)
            hands_res = mp_hands.process(rgb)
            
            arm = {}
            if pose_res.pose_landmarks:
                pl = pose_res.pose_landmarks.landmark
                mapping = {
                    'NOSE': 0, 'LEFT_SHOULDER': 11, 'LEFT_ELBOW': 13, 'LEFT_WRIST': 15,
                    'RIGHT_SHOULDER': 12, 'RIGHT_ELBOW': 14, 'RIGHT_WRIST': 16
                }
                for k, idx in mapping.items():
                    arm[k] = {'x': pl[idx].x, 'y': pl[idx].y, 'z': pl[idx].z}

            left_feats = None
            right_feats = None
            
            if hands_res.multi_hand_landmarks:
                for idx, landmarks in enumerate(hands_res.multi_hand_landmarks):
                    # FIX: MediaPipe returns REVERSED labels when camera is flipped!
                    # We need to swap left <-> right
                    mp_label = hands_res.multi_handedness[idx].classification[0].label.lower()
                    
                    # SWAP: MediaPipe's "left" is actually user's right hand (and vice versa)
                    label = 'right' if mp_label == 'left' else 'left'
                    
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
                    
                    # Draw
                    color = (255, 100, 100) if label == 'right' else (100, 100, 255)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=3, circle_radius=4),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Add label text
                    h, w, _ = frame.shape
                    cx = int(landmarks.landmark[0].x * w)
                    cy = int(landmarks.landmark[0].y * h)
                    cv2.putText(frame, label.upper(), (cx - 30, cy - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Velocity
                    wrist = pts[0]
                    prev = prev_wrist.get(label)
                    vel = (wrist - prev) * 30 if prev is not None else np.zeros(3, dtype=np.float32)
                    prev_wrist[label] = wrist
                    vel_buffers[label].append(vel)
                    
                    # Features (EXACT match with training)
                    static = build_static_from_pts(pts, arm, label)
                    dynamic = compute_dyn_stats_from_buffer(vel_buffers[label])
                    
                    if label == 'left':
                        left_feats = (static, dynamic)
                    else:
                        right_feats = (static, dynamic)
            
            # Predict
            word, conf, debug = self.predictor.predict(left_feats, right_feats)
            
            if word:
                arabic = WORD_DB.get(word, {}).get('arabic', word)
                self.result_signal.emit(word, arabic, conf)
            
            self.debug_signal.emit(debug, conf)
            self.frame_signal.emit(frame)
            
        cap.release()
        mp_pose.close()
        mp_hands.close()

# ═══════════════════════════════════════════════════════════
# 🎨 UI COMPONENTS
# ═══════════════════════════════════════════════════════════
class GradientButton(QPushButton):
    def __init__(self, text, color1, color2):
        super().__init__(text)
        self.color1 = QColor(color1)
        self.color2 = QColor(color2)
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 12, QFont.Bold))
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, self.color1)
        gradient.setColorAt(1, self.color2)
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 10, 10)
        
        painter.setPen(QColor("white"))
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())

class ConfidenceMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.value = 0.0
        self.setMinimumHeight(30)
        
    def setValue(self, val):
        self.value = max(0.0, min(1.0, val))
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setBrush(QColor(40, 40, 40))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)
        
        if self.value > 0:
            fill_width = int(self.width() * self.value)
            color = QColor(46, 204, 113) if self.value > 0.7 else QColor(241, 196, 15) if self.value > 0.4 else QColor(231, 76, 60)
            painter.setBrush(color)
            painter.drawRoundedRect(0, 0, fill_width, self.height(), 15, 15)
        
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, f"{int(self.value * 100)}%")

# ═══════════════════════════════════════════════════════════
# 🏠 MAIN WINDOW
# ═══════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌟 Ultimate Sign Language AI 🌟")
        self.setGeometry(100, 50, 1400, 900)
        
        self.tts = TTSWorker()
        self.thread = VideoThread()
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.result_signal.connect(self.handle_result)
        self.thread.debug_signal.connect(self.update_debug)
        
        self.sentence = []
        self.is_running = False
        
        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # LEFT: Video
        left_panel = QFrame()
        left_panel.setObjectName("videoPanel")
        left_layout = QVBoxLayout(left_panel)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 720)
        self.video_label.setStyleSheet("background: #000; border-radius: 15px;")
        left_layout.addWidget(self.video_label)
        
        self.debug_label = QLabel("Debug: Ready")
        self.debug_label.setStyleSheet("color: #888; font-size: 11px; padding: 5px;")
        left_layout.addWidget(self.debug_label)

        # RIGHT: Controls
        right_panel = QFrame()
        right_panel.setObjectName("controlPanel")
        right_panel.setFixedWidth(400)
        right_layout = QVBoxLayout(right_panel)
        
        header = QLabel("🎯 Sign Language AI")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 28px; font-weight: bold; color: white; margin: 10px;")
        right_layout.addWidget(header)
        
        # Word Card
        card = QFrame()
        card.setObjectName("wordCard")
        card_layout = QVBoxLayout(card)
        
        self.lbl_english = QLabel("Waiting...")
        self.lbl_english.setAlignment(Qt.AlignCenter)
        self.lbl_english.setStyleSheet("font-size: 32px; font-weight: bold; color: #3498db;")
        
        self.lbl_arabic = QLabel("...")
        self.lbl_arabic.setAlignment(Qt.AlignCenter)
        self.lbl_arabic.setStyleSheet("font-size: 24px; color: #e74c3c;")
        
        card_layout.addWidget(self.lbl_english)
        card_layout.addWidget(self.lbl_arabic)
        right_layout.addWidget(card)
        
        # Confidence
        conf_label = QLabel("📊 Confidence")
        conf_label.setStyleSheet("color: #aaa; font-weight: bold; margin-top: 10px;")
        right_layout.addWidget(conf_label)
        
        self.conf_meter = ConfidenceMeter()
        right_layout.addWidget(self.conf_meter)
        
        # Sentence
        sent_label = QLabel("📝 Sentence")
        sent_label.setStyleSheet("color: #aaa; font-weight: bold; margin-top: 15px;")
        right_layout.addWidget(sent_label)
        
        self.sentence_box = QTextEdit()
        self.sentence_box.setPlaceholderText("Your sentence...")
        self.sentence_box.setReadOnly(True)
        self.sentence_box.setStyleSheet("""
            QTextEdit {
                background: #2a2a2a;
                color: white;
                border: 2px solid #444;
                border-radius: 10px;
                font-size: 16px;
                padding: 10px;
            }
        """)
        self.sentence_box.setMaximumHeight(120)
        right_layout.addWidget(self.sentence_box)
        
        # Buttons
        self.btn_start = GradientButton("▶ Start", "#2ecc71", "#27ae60")
        self.btn_start.clicked.connect(self.toggle_camera)
        
        self.btn_speak = GradientButton("🔊 Speak", "#9b59b6", "#8e44ad")
        self.btn_speak.clicked.connect(self.speak_sentence)
        
        self.btn_clear = GradientButton("🗑 Clear", "#e74c3c", "#c0392b")
        self.btn_clear.clicked.connect(self.clear_all)
        
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_speak)
        right_layout.addWidget(self.btn_clear)
        
        self.chk_arabic = QCheckBox("🌐 Arabic TTS")
        self.chk_arabic.setChecked(True)
        self.chk_arabic.setStyleSheet("color: white; font-size: 14px; margin-top: 10px;")
        right_layout.addWidget(self.chk_arabic)
        
        right_layout.addStretch()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #555; font-size: 12px; font-style: italic;")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)
        
        layout.addWidget(left_panel, stretch=3)
        layout.addWidget(right_panel, stretch=1)

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0c29, stop:0.5 #302b63, stop:1 #24243e);
            }
            QFrame#videoPanel {
                background: rgba(30, 30, 30, 180);
                border-radius: 15px;
                padding: 10px;
            }
            QFrame#controlPanel {
                background: rgba(40, 40, 40, 200);
                border-radius: 15px;
                padding: 15px;
            }
            QFrame#wordCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 15px;
                padding: 20px;
                margin: 10px 0;
            }
        """)

    def toggle_camera(self):
        if not self.is_running:
            self.thread.start()
            self.is_running = True
            self.btn_start.setText("⏹ Stop")
            self.status_label.setText("🟢 Active")
        else:
            self.thread.running = False
            self.thread.quit()
            self.is_running = False
            self.btn_start.setText("▶ Start")
            self.status_label.setText("🔴 Stopped")

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_debug(self, info, conf):
        self.debug_label.setText(f"Debug: {info}")
        self.conf_meter.setValue(conf)

    def handle_result(self, word, arabic, conf):
        self.lbl_english.setText(word)
        self.lbl_arabic.setText(arabic)
        self.conf_meter.setValue(conf)
        
        self.sentence.append(word)
        self.update_sentence()
        
        # Speak immediately
        lang = 'ar' if self.chk_arabic.isChecked() else 'en'
        text = arabic if lang == 'ar' else word
        self.tts.speak(text, lang)
        print(f"[SPEAKING] {text} ({lang})")
        
        self.status_label.setText(f"✅ {word} ({int(conf*100)}%)")

    def update_sentence(self):
        if self.chk_arabic.isChecked():
            words = [WORD_DB.get(w, {}).get('arabic', w) for w in self.sentence]
            self.sentence_box.setText(" ".join(words))
        else:
            self.sentence_box.setText(" ".join(self.sentence))

    def speak_sentence(self):
        text = self.sentence_box.toPlainText()
        if text:
            lang = 'ar' if self.chk_arabic.isChecked() else 'en'
            self.tts.speak(text, lang)
            print(f"[SPEAKING SENTENCE] {text} ({lang})")

    def clear_all(self):
        self.sentence = []
        self.sentence_box.clear()
        self.lbl_english.setText("Waiting...")
        self.lbl_arabic.setText("...")
        self.conf_meter.setValue(0)
        self.status_label.setText("Cleared")

    def closeEvent(self, event):
        self.thread.running = False
        self.tts.stop()
        event.accept()

# ═══════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(20, 20, 20))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())