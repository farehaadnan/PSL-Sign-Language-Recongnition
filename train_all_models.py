import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

print("=" * 70)
print("TRAINING ALL MODELS")
print("=" * 70)

models_trained = {}

# ========== MODEL 1: ALPHABETS ==========
print("\n" + "=" * 70)
print("📝 TRAINING ALPHABET MODEL (37 letters)")
print("=" * 70)

try:
    with open('sign_language_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X = np.array(data['data'])
    y = np.array(data['labels'])
    
    print(f"✓ Loaded: {len(X)} samples, {len(set(y))} classes")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"✓ Training accuracy: {train_acc*100:.2f}%")
    print(f"✓ Testing accuracy: {test_acc*100:.2f}%")
    
    with open('alphabet_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    models_trained['alphabets'] = {
        'file': 'alphabet_model.pkl',
        'classes': len(set(y)),
        'accuracy': test_acc,
        'type': 'static'
    }
    
    print("✓ Saved: alphabet_model.pkl")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ========== MODEL 2: STATIC WORDS ==========
print("\n" + "=" * 70)
print("📄 TRAINING STATIC WORDS MODEL")
print("=" * 70)

# Merge old and new static words
static_X = []
static_y = []

try:
    # Load old static words
    if os.path.exists('word_dataset.pkl'):
        with open('word_dataset.pkl', 'rb') as f:
            data = pickle.load(f)
        if 'data' in data:
            static_X.extend(data['data'])
            static_y.extend(data['labels'])
            print(f"✓ Loaded old static words: {len(data['data'])} samples")
    
    # Load new static words
    if os.path.exists('complete_word_dataset.pkl'):
        with open('complete_word_dataset.pkl', 'rb') as f:
            data = pickle.load(f)
        if 'data' in data:
            static_X.extend(data['data'])
            static_y.extend(data['labels'])
            print(f"✓ Loaded new static words: {len(data['data'])} samples")
    
    # Also check word_collection_progress.pkl
    if os.path.exists('word_collection_progress.pkl'):
        with open('word_collection_progress.pkl', 'rb') as f:
            data = pickle.load(f)
        if 'data' in data and len(data['data']) > 0:
            static_X.extend(data['data'])
            static_y.extend(data['labels'])
            print(f"✓ Loaded progress file: {len(data['data'])} samples")
    
    if len(static_X) > 0:
        X = np.array(static_X)
        y = np.array(static_y)
        
        print(f"\n✓ Total: {len(X)} samples, {len(set(y))} classes")
        print(f"  Classes: {sorted(set(y))}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        print(f"\n✓ Training accuracy: {train_acc*100:.2f}%")
        print(f"✓ Testing accuracy: {test_acc*100:.2f}%")
        
        with open('static_words_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        models_trained['static_words'] = {
            'file': 'static_words_model.pkl',
            'classes': len(set(y)),
            'accuracy': test_acc,
            'type': 'static'
        }
        
        print("✓ Saved: static_words_model.pkl")
    else:
        print("⚠️  No static word data found")
        
except Exception as e:
    print(f"❌ Error: {e}")

# ========== MODEL 3: ONE-HANDED MOTION ==========
print("\n" + "=" * 70)
print("🎬 TRAINING ONE-HANDED MOTION MODEL (LSTM)")
print("=" * 70)

try:
    with open('motion_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X = data['sequences']
    y = data['labels']
    
    print(f"✓ Loaded: {len(X)} sequences, {len(set(y))} classes")
    print(f"  Classes: {sorted(set(y))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n🎯 Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n✓ Testing accuracy: {test_acc*100:.2f}%")
    
    model.save('motion_1hand_model.h5')
    
    with open('motion_1hand_labels.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    models_trained['motion_1hand'] = {
        'file': 'motion_1hand_model.h5',
        'labels': 'motion_1hand_labels.pkl',
        'classes': len(label_encoder.classes_),
        'accuracy': test_acc,
        'type': 'motion'
    }
    
    print("✓ Saved: motion_1hand_model.h5")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ========== MODEL 4: TWO-HANDED MOTION ==========
print("\n" + "=" * 70)
print("🤝 TRAINING TWO-HANDED MOTION MODEL (LSTM)")
print("=" * 70)

try:
    with open('two_handed_motion_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X = data['sequences']
    y = data['labels']
    
    print(f"✓ Loaded: {len(X)} sequences, {len(set(y))} classes")
    print(f"  Classes: {sorted(set(y))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n🎯 Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n✓ Testing accuracy: {test_acc*100:.2f}%")
    
    model.save('motion_2hand_model.h5')
    
    with open('motion_2hand_labels.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    models_trained['motion_2hand'] = {
        'file': 'motion_2hand_model.h5',
        'labels': 'motion_2hand_labels.pkl',
        'classes': len(label_encoder.classes_),
        'accuracy': test_acc,
        'type': 'motion'
    }
    
    print("✓ Saved: motion_2hand_model.h5")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ========== SUMMARY ==========
print("\n" + "=" * 70)
print("📊 TRAINING SUMMARY")
print("=" * 70)

for name, info in models_trained.items():
    print(f"\n{name}:")
    print(f"  Model: {info['file']}")
    print(f"  Classes: {info['classes']}")
    print(f"  Accuracy: {info['accuracy']*100:.2f}%")
    print(f"  Type: {info['type']}")

print("\n" + "=" * 70)
print("✅ ALL MODELS TRAINED!")
print("=" * 70)
print("\nNext: python create_master_system.py")
print("=" * 70)

# Save model info
with open('models_info.pkl', 'wb') as f:
    pickle.dump(models_trained, f)