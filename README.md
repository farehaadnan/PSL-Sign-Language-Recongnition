# PSL-Sign-Language-Recongnition
# Pakistani Sign Language Recognition System

AI system that recognizes Pakistani Sign Language in real-time.

## Features
- Recognizes 62 signs (37 alphabets + 18 words + 7 gestures)
- Real-time at 30 FPS
- 85% accuracy
- Works on regular laptops

## How to Run
```bash
# Install requirements
pip install opencv-python mediapipe numpy scikit-learn tensorflow

# Run the system
python fast_system.py
```

## Controls
- Press 1 = Static mode (alphabets/words)
- Press 2 = Motion mode (gestures)
- SPACE = Add sign to sentence
- Q = Quit

## Results
- Alphabets: 90% accuracy (Random Forest)
- Static Words: 85% accuracy (Random Forest)
- Motion Gestures: 78% accuracy (LSTM)
- Overall: 85% accuracy

## Tech Stack
- Python 3.11
- TensorFlow 2.18
- MediaPipe
- OpenCV
- Scikit-learn


## License
MIT License
