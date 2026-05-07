import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import json
import os
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from collections import deque, Counter
import threading

app = Flask(__name__)
CORS(app)

# ── Model loading ──────────────────────────────────────────────────────────────
models = {}

def load_models():
    try:
        models['alphabet'] = pickle.load(open('alphabet_model.pkl', 'rb'))
        print("✓ Alphabet model loaded")
    except Exception as e:
        models['alphabet'] = None
        print(f"✗ Alphabet model: {e}")

    try:
        models['static_words'] = pickle.load(open('static_words_model.pkl', 'rb'))
        print("✓ Static words model loaded")
    except Exception as e:
        models['static_words'] = None
        print(f"✗ Static words model: {e}")

    try:
        from tensorflow.keras.models import load_model
        models['motion_1hand'] = load_model('motion_1hand_model.h5')
        models['motion_1hand_labels'] = pickle.load(open('motion_1hand_labels.pkl', 'rb'))
        print("✓ Motion 1-hand model loaded")
    except Exception as e:
        models['motion_1hand'] = None
        print(f"✗ Motion 1-hand model: {e}")

    try:
        from tensorflow.keras.models import load_model
        models['motion_2hand'] = load_model('motion_2hand_model.h5')
        models['motion_2hand_labels'] = pickle.load(open('motion_2hand_labels.pkl', 'rb'))
        print("✓ Motion 2-hand model loaded")
    except Exception as e:
        models['motion_2hand'] = None
        print(f"✗ Motion 2-hand model: {e}")


# ── MediaPipe ──────────────────────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── State ──────────────────────────────────────────────────────────────────────
state = {
    'mode': 'static',
    'sentence': [],
    'stable_pred': '',
    'last_added': '',
    'motion_buffer': deque(maxlen=35),
    'prediction_history': deque(maxlen=12),
    'STABILITY': 8,
    'MOTION_FRAMES': 30,
}
state_lock = threading.Lock()


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
            float(np.clip(x_norm[i], 0, 1)),
            float(np.clip(y_norm[i], 0, 1)),
            float(np.clip(z_norm[i], 0, 1))
        ])
    return normalized


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process a base64 frame from the browser webcam."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64 image
    try:
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
    except Exception as e:
        return jsonify({'error': f'Image decode error: {str(e)}'}), 400

    # Flip (mirror) like original
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    with state_lock:
        current_mode = state['mode']
        motion_buffer = state['motion_buffer']
        prediction_history = state['prediction_history']

        current_pred = None
        num_hands = 0
        hand_landmarks_data = []

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            # Collect landmark positions for frontend drawing
            for hand_lm in results.multi_hand_landmarks:
                lm_list = []
                for lm in hand_lm.landmark:
                    lm_list.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                hand_landmarks_data.append(lm_list)

            # ── STATIC MODE ────────────────────────────────────────────────
            if current_mode == 'static' and num_hands >= 1:
                coords = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                norm = normalize_coords(coords)

                best_pred = None
                best_conf = 0

                if models.get('alphabet'):
                    try:
                        proba = models['alphabet'].predict_proba([norm])[0]
                        conf = float(np.max(proba))
                        if conf > best_conf:
                            best_pred = models['alphabet'].predict([norm])[0]
                            best_conf = conf
                    except:
                        pass

                if models.get('static_words'):
                    try:
                        proba = models['static_words'].predict_proba([norm])[0]
                        conf = float(np.max(proba))
                        if conf > best_conf:
                            best_pred = models['static_words'].predict([norm])[0]
                            best_conf = conf
                    except:
                        pass

                if best_conf > 0.25:
                    current_pred = best_pred

            # ── MOTION MODE ────────────────────────────────────────────────
            elif current_mode == 'motion':
                if num_hands == 1:
                    coords = []
                    for lm in results.multi_hand_landmarks[0].landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    norm = normalize_coords(coords)
                    motion_buffer.append(norm)

                    if len(motion_buffer) >= state['MOTION_FRAMES'] and models.get('motion_1hand'):
                        try:
                            seq = list(motion_buffer)[-state['MOTION_FRAMES']:]
                            seq_array = np.array([seq])
                            proba = models['motion_1hand'].predict(seq_array, verbose=0)[0]
                            conf = float(np.max(proba))
                            if conf > 0.20:
                                idx = int(np.argmax(proba))
                                current_pred = models['motion_1hand_labels'].inverse_transform([idx])[0]
                        except:
                            pass

                elif num_hands == 2:
                    left, right = [], []
                    for lm in results.multi_hand_landmarks[0].landmark:
                        left.extend([lm.x, lm.y, lm.z])
                    for lm in results.multi_hand_landmarks[1].landmark:
                        right.extend([lm.x, lm.y, lm.z])
                    combined = normalize_coords(left) + normalize_coords(right)
                    motion_buffer.append(combined)

                    if len(motion_buffer) >= state['MOTION_FRAMES'] and models.get('motion_2hand'):
                        try:
                            seq = list(motion_buffer)[-state['MOTION_FRAMES']:]
                            seq_array = np.array([seq])
                            proba = models['motion_2hand'].predict(seq_array, verbose=0)[0]
                            conf = float(np.max(proba))
                            if conf > 0.20:
                                idx = int(np.argmax(proba))
                                current_pred = models['motion_2hand_labels'].inverse_transform([idx])[0]
                        except:
                            pass
                else:
                    motion_buffer.clear()
                    prediction_history.clear()

        # Stabilize
        if current_pred:
            prediction_history.append(current_pred)

        stable_pred = ''
        if len(prediction_history) >= state['STABILITY']:
            counts = Counter(prediction_history)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= state['STABILITY']:
                stable_pred = most_common[0]
                state['stable_pred'] = stable_pred

        return jsonify({
            'prediction': stable_pred or state.get('stable_pred', ''),
            'raw_prediction': current_pred or '',
            'num_hands': num_hands,
            'hand_landmarks': hand_landmarks_data,
            'mode': current_mode,
            'sentence': list(state['sentence']),
            'motion_buffer_size': len(motion_buffer),
            'motion_frames_needed': state['MOTION_FRAMES'],
        })


@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json()
    mode = data.get('mode', 'static')
    if mode not in ('static', 'motion'):
        return jsonify({'error': 'Invalid mode'}), 400
    with state_lock:
        state['mode'] = mode
        state['motion_buffer'].clear()
        state['prediction_history'].clear()
        state['stable_pred'] = ''
    return jsonify({'mode': mode})


@app.route('/api/add_sign', methods=['POST'])
def add_sign():
    with state_lock:
        pred = state.get('stable_pred', '')
        if pred and pred != state.get('last_added', ''):
            state['sentence'].append(pred)
            state['last_added'] = pred
            state['prediction_history'].clear()
            state['motion_buffer'].clear()
            state['stable_pred'] = ''
            return jsonify({'sentence': list(state['sentence']), 'added': pred})
    return jsonify({'sentence': list(state['sentence']), 'added': None})


@app.route('/api/remove_last', methods=['POST'])
def remove_last():
    with state_lock:
        removed = state['sentence'].pop() if state['sentence'] else None
        state['last_added'] = ''
    return jsonify({'sentence': list(state['sentence']), 'removed': removed})


@app.route('/api/clear_sentence', methods=['POST'])
def clear_sentence():
    with state_lock:
        state['sentence'] = []
        state['last_added'] = ''
    return jsonify({'sentence': []})


@app.route('/api/save_sentence', methods=['POST'])
def save_sentence():
    with state_lock:
        sentence = list(state['sentence'])
    if sentence:
        text = ' '.join(sentence)
        with open('urdu_sentences.txt', 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        with state_lock:
            state['sentence'] = []
            state['last_added'] = ''
        return jsonify({'saved': text, 'sentence': []})
    return jsonify({'saved': None, 'sentence': sentence})


@app.route('/api/status', methods=['GET'])
def status():
    loaded = {k: (v is not None) for k, v in models.items()}
    return jsonify({
        'models': loaded,
        'mode': state['mode'],
        'sentence': list(state['sentence']),
    })


if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)