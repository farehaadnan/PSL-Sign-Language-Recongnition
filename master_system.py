import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model

print("=" * 70)
print("🎯 FAST SIGN LANGUAGE SYSTEM - USER CONTROLLED")
print("=" * 70)
print("YOU choose mode → System detects instantly!")
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
    print(f"   Signs: {sorted(models['motion_1hand_labels'].classes_)}")
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

# Mode
current_mode = 'static'  # static, motion
modes = ['static', 'motion']

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

print("=" * 70)
print("⚡ FAST CONTROLS:")
print("=" * 70)
print("\n📍 MODE SWITCHING (Instant!):")
print("  1 - STATIC mode  (letters/words: mai, aik, hoon)")
print("  2 - MOTION mode  (gestures: larki, maa, abbu)")
print()
print("⚙️  OTHER:")
print("  SPACE - Add current sign to sentence")
print("  BACKSPACE - Remove last word")
print("  C - Clear sentence")
print("  S - Save sentence")
print("  Q - Quit")
print()
print("=" * 70)
print(f"\n✨ Starting in STATIC mode")
print("💡 TIP: Press 1 for static signs, 2 for motion signs!")
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
        
        # Draw hands
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            color = (0, 255, 0) if idx == 0 else (255, 0, 0)
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        
        # ========== STATIC MODE ==========
        if current_mode == 'static':
            if num_hands == 1:
                coords = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                norm = normalize_coords(coords)
                
                # Check both alphabet and words
                best_pred = None
                best_conf = 0
                
                # Alphabet
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
                
                # Words
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
        
        # ========== MOTION MODE ==========
        elif current_mode == 'motion':
            if num_hands == 1:
                coords = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                norm = normalize_coords(coords)
                motion_buffer.append(norm)
                
                # Check motion prediction
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
                
                # Check 2-hand motion
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
    
    # Stabilize prediction
    if current_pred:
        prediction_history.append(current_pred)
        
        if len(prediction_history) >= STABILITY:
            counts = Counter(prediction_history)
            most_common = counts.most_common(1)[0]
            
            if most_common[1] >= STABILITY:
                stable_pred = most_common[0]
    
    # UI
    # Top bar - Larger, clearer mode indicator
    mode_color = (0, 255, 255) if current_mode == 'static' else (255, 128, 0)
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    
    # Mode indicator (BIG)
    mode_text = "STATIC" if current_mode == 'static' else "MOTION"
    cv2.rectangle(frame, (10, 10), (180, 50), mode_color, -1)
    cv2.putText(frame, mode_text, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    
    # Current prediction
    if stable_pred:
        cv2.putText(frame, stable_pred, (200, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        hint = "Show sign..." if current_mode == 'static' else "Perform gesture..."
        cv2.putText(frame, hint, (200, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    # Mode hint
    cv2.putText(frame, "Press 1=Static | 2=Motion", (10, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Motion buffer indicator (in motion mode)
    if current_mode == 'motion' and len(motion_buffer) > 0:
        buf_progress = min(len(motion_buffer), MOTION_FRAMES)
        bar_width = int((buf_progress / MOTION_FRAMES) * 200)
        cv2.rectangle(frame, (w-220, 20), (w-220+bar_width, 35), (255, 128, 0), -1)
        cv2.putText(frame, f"{buf_progress}/{MOTION_FRAMES}", (w-210, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Sentence area
    cv2.rectangle(frame, (0, h-70), (w, h), (20, 20, 20), -1)
    
    sent_text = " ".join(sentence) if sentence else "Press SPACE to add signs to sentence"
    
    # Word wrap
    max_chars = int(w / 7)
    if len(sent_text) > max_chars:
        # Show last part
        sent_text = "..." + sent_text[-(max_chars-3):]
    
    cv2.putText(frame, sent_text, (10, h-35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, "SPACE=Add | BACK=Del | C=Clear | S=Save | Q=Quit",
               (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    
    cv2.imshow('Fast Sign Language System', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Mode switching
    if key == ord('1'):
        current_mode = 'static'
        motion_buffer.clear()
        prediction_history.clear()
        stable_pred = ""
        print(f"\n📍 Switched to: STATIC mode")
        print("   (For: mai, aik, hoon, letters, etc.)")
    
    elif key == ord('2'):
        current_mode = 'motion'
        motion_buffer.clear()
        prediction_history.clear()
        stable_pred = ""
        print(f"\n📍 Switched to: MOTION mode")
        print("   (For: larki, maa, abbu, hello, etc.)")
    
    # Add to sentence
    elif key == ord(' '):
        if stable_pred and stable_pred != last_added:
            sentence.append(stable_pred)
            last_added = stable_pred
            print(f"✓ Added: {stable_pred}")
            
            # Clear for next
            prediction_history.clear()
            motion_buffer.clear()
            stable_pred = ""
    
    elif key == 8:  # Backspace
        if sentence:
            removed = sentence.pop()
            last_added = ""
            print(f"✗ Removed: {removed}")
    
    elif key == ord('c'):
        sentence = []
        last_added = ""
        print("✗ Cleared sentence")
    
    elif key == ord('s'):
        if sentence:
            with open('urdu_sentences.txt', 'a', encoding='utf-8') as f:
                f.write(" ".join(sentence) + "\n")
            print(f"💾 Saved: {' '.join(sentence)}")
            sentence = []
            last_added = ""
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ System closed")
if sentence:
    print(f"\nFinal sentence: {' '.join(sentence)}")
    save = input("Save before exiting? (y/n): ")
    if save.lower() == 'y':
        with open('urdu_sentences.txt', 'a', encoding='utf-8') as f:
            f.write(" ".join(sentence) + "\n")
        print("✓ Saved!")