import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# Load model once
try:
    siren_model = load_model('siren_model/best_model.keras')
    print("✅ Siren detection model loaded!")
except Exception as e:
    print(f"❌ Error loading siren model: {e}")
    siren_model = None

def extract_features(audio_file, max_pad_len=862):
    try:
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
            return None
        audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        features = np.mean(mfccs, axis=1).reshape(1, 1, 80)
        return features
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

def detect_siren(audio_file):
    if siren_model is None:
        return False
    features = extract_features(audio_file)
    if features is None:
        return False
    prediction = siren_model.predict(features)
    return prediction[0][0] > 0.5
