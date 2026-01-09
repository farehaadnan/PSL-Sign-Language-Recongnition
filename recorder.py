import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model
from datetime import datetime

print("=" * 70)
print("🎥 DEMO RECORDING MODE - Sign Language System")
print("=" * 70)
print("Auto-records video for presentation!")
print("=" * 70)
print("\nLoading models...")

# Load models
models = {}

try:
    models['alphabet'] = pickle.load(open('alphabet_model.pkl', 'rb'))
    print("✓ Alphabet")
except:
    models['alphabet'] = None

try:
    models['static_words'] = pickle.load(open('static_words_model.pkl', 'rb'))
    print("✓ Static words")
except:
    models['static_words'] = None

try:
    models['motion_1hand'] = load_model('motion_1hand_model.h5')
    models['motion_1hand_labels'] = pickle.load(open('motion_1hand_labels.pkl', 'rb'))
    print("✓ Motion (1-hand)")
except:
    models['motion_1hand'] = None

try:
    models['motion_2hand'] = load_model('motion_2hand_model.h5')
    models['motion_2hand_labels'] = pickle.load(open('motion_2hand_labels.pkl', 'rb'))
    print("✓ Motion (2-hand)")
except:
    models['motion_2hand'] = None

print("\n" + "=" * 70)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

# Video writer setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"demo_recording_{timestamp}.avi"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

print(f"\n🎥 RECORDING TO: {output_filename}")
print("=" * 70)

# Mode
current_mode = 'static'

# Motion tracking
motion_buffer = deque(maxlen=35)
MOTION_FRAMES = 30

# Predictions
prediction_history = deque(maxlen=12)
stable_pred = ""
last_added = ""
STABILITY = 8

# Sentence
sentence = []

# Recording indicator
recording = True
frame_count = 0

def normalize_coords(coords):
    coords_array = np.array(coords).reshape(21, 3)
    x_coords, y_coords, z_coords = coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]
    
    x_range = max(x_coords.max() - x_coords.min(), 0.001)
    y_range = max(y_coords.max() - y_coords.min(), 0.001)
    scale = max(x_range, y_range)
    
    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2
    x_norm = (x_coords - x_center) / scale + 0.5
    y_norm = (y_coords - y_center) / scale + 0.5
    
    z_range = max(z_coords.max() - z_coords.min(), 0.001)
    z_norm = (z_coords - z_coords.min()) / z_range
    
    normalized = []
    for i in range(21):
        normalized.extend([
            np.clip(x_norm[i], 0, 1),
            np.clip(y_norm[i], 0, 1),
            np.clip(z_norm[i], 0, 1)
        ])
    return normalized

print("\n⚡ CONTROLS:")
print("  1 - STATIC mode")
print("  2 - MOTION mode")
print("  SPACE - Add sign to sentence")
print("  R - Toggle recording ON/OFF")
print("  Q - Stop & Save video")
print("\n💡 Video is recording NOW!")
print("=" * 70)
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb)
    
    current_pred = None
    num_hands = 0
    
    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            color = (0, 255, 0) if idx == 0 else (255, 0, 0)
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        
        if current_mode == 'static':
            if num_hands == 1:
                coords = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                norm = normalize_coords(coords)
                
                best_pred = None
                best_conf = 0
                
                if models['alphabet']:
                    try:
                        proba = models['alphabet'].predict_proba([norm])[0]
                        conf = np.max(proba)
                        if conf > best_conf:
                            pred = models['alphabet'].predict([norm])[0]
                            best_pred = pred
                            best_conf = conf
                    except:
                        pass
                
                if models['static_words']:
                    try:
                        proba = models['static_words'].predict_proba([norm])[0]
                        conf = np.max(proba)
                        if conf > best_conf:
                            pred = models['static_words'].predict([norm])[0]
                            best_pred = pred
                            best_conf = conf
                    except:
                        pass
                
                if best_conf > 0.25:
                    current_pred = best_pred
        
        elif current_mode == 'motion':
            if num_hands == 1:
                coords = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                norm = normalize_coords(coords)
                motion_buffer.append(norm)
                
                if len(motion_buffer) >= MOTION_FRAMES and models['motion_1hand']:
                    try:
                        seq = list(motion_buffer)[-MOTION_FRAMES:]
                        seq_array = np.array([seq])
                        proba = models['motion_1hand'].predict(seq_array, verbose=0)[0]
                        conf = np.max(proba)
                        
                        if conf > 0.20:
                            idx = np.argmax(proba)
                            pred = models['motion_1hand_labels'].inverse_transform([idx])[0]
                            current_pred = pred
                    except:
                        pass
            
            elif num_hands == 2:
                left, right = [], []
                for lm in results.multi_hand_landmarks[0].landmark:
                    left.extend([lm.x, lm.y, lm.z])
                for lm in results.multi_hand_landmarks[1].landmark:
                    right.extend([lm.x, lm.y, lm.z])
                
                left_norm = normalize_coords(left)
                right_norm = normalize_coords(right)
                combined = left_norm + right_norm
                motion_buffer.append(combined)
                
                if len(motion_buffer) >= MOTION_FRAMES and models['motion_2hand']:
                    try:
                        seq = list(motion_buffer)[-MOTION_FRAMES:]
                        seq_array = np.array([seq])
                        proba = models['motion_2hand'].predict(seq_array, verbose=0)[0]
                        conf = np.max(proba)
                        
                        if conf > 0.20:
                            idx = np.argmax(proba)
                            pred = models['motion_2hand_labels'].inverse_transform([idx])[0]
                            current_pred = pred
                    except:
                        pass
    
    else:
        motion_buffer.clear()
        prediction_history.clear()
    
    if current_pred:
        prediction_history.append(current_pred)
        
        if len(prediction_history) >= STABILITY:
            counts = Counter(prediction_history)
            most_common = counts.most_common(1)[0]
            
            if most_common[1] >= STABILITY:
                stable_pred = most_common[0]
    
    # UI
    mode_color = (0, 255, 255) if current_mode == 'static' else (255, 128, 0)
    cv2.rectangle(frame, (0, 0), (w, 100), (30, 30, 30), -1)
    
    # Recording indicator (RED DOT)
    if recording:
        cv2.circle(frame, (w-30, 30), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w-60, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Mode
    mode_text = "STATIC" if current_mode == 'static' else "MOTION"
    cv2.rectangle(frame, (10, 10), (180, 55), mode_color, -1)
    cv2.putText(frame, mode_text, (20, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    
    # Prediction
    if stable_pred:
        cv2.putText(frame, stable_pred, (200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    else:
        hint = "Show sign..." if current_mode == 'static' else "Perform gesture..."
        cv2.putText(frame, hint, (200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    # Controls hint
    cv2.putText(frame, "1=Static | 2=Motion | SPACE=Add | Q=Save&Quit", (10, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Motion buffer
    if current_mode == 'motion' and len(motion_buffer) > 0:
        buf_progress = min(len(motion_buffer), MOTION_FRAMES)
        bar_width = int((buf_progress / MOTION_FRAMES) * 150)
        cv2.rectangle(frame, (w-170, 60), (w-170+bar_width, 75), (255, 128, 0), -1)
        cv2.putText(frame, f"{buf_progress}/{MOTION_FRAMES}", (w-165, 72),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Sentence
    cv2.rectangle(frame, (0, h-80), (w, h), (20, 20, 20), -1)
    
    sent_text = " ".join(sentence) if sentence else "Your sentence will appear here..."
    
    max_chars = int(w / 7)
    if len(sent_text) > max_chars:
        sent_text = "..." + sent_text[-(max_chars-3):]
    
    cv2.putText(frame, sent_text, (10, h-40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Pakistani Sign Language Recognition System",
               (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    
    # Write frame to video
    if recording:
        out.write(frame)
        frame_count += 1
    
    cv2.imshow('Demo Recording - Sign Language System', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('1'):
        current_mode = 'static'
        motion_buffer.clear()
        prediction_history.clear()
        stable_pred = ""
        print(f"📍 Mode: STATIC")
    
    elif key == ord('2'):
        current_mode = 'motion'
        motion_buffer.clear()
        prediction_history.clear()
        stable_pred = ""
        print(f"📍 Mode: MOTION")
    
    elif key == ord(' '):
        if stable_pred and stable_pred != last_added:
            sentence.append(stable_pred)
            last_added = stable_pred
            print(f"✓ Added: {stable_pred}")
            prediction_history.clear()
            motion_buffer.clear()
            stable_pred = ""
    
    elif key == 8:
        if sentence:
            removed = sentence.pop()
            last_added = ""
            print(f"✗ Removed: {removed}")
    
    elif key == ord('c'):
        sentence = []
        last_added = ""
    
    elif key == ord('r'):
        recording = not recording
        status = "RECORDING" if recording else "PAUSED"
        print(f"🎥 {status}")
    
    elif key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("✅ DEMO SAVED!")
print("=" * 70)
print(f"📹 Video: {output_filename}")
print(f"📊 Frames: {frame_count}")
print(f"⏱️  Duration: {frame_count/fps:.1f} seconds")

if sentence:
    print(f"📝 Final sentence: {' '.join(sentence)}")

print("\n💡 Use this video in your presentation!")
print("=" * 70)