import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score

# Configuration
SAMPLE_RATE = 16000
DURATION = 0.5        # Seconds (0.5s * 16000 = 8000 samples)
N_MFCC = 10           # Reduced for Pi Zero's capability
N_FFT = 512           # Adjusted to match 0.032s window (512/16000)
HOP_LENGTH = 256      # 50% overlap for better time resolution
DATA_PATH = "training_data/"
MODEL_PATH = "wooden_hit_model.pkl"

def extract_features(file_path):
    try:
        # Load with enforced duration and padding
        audio, sr = librosa.load(file_path, 
                                sr=SAMPLE_RATE,
                                duration=DURATION,
                                res_type='kaiser_fast')
        
        # Explicit padding to ensure correct length
        if len(audio) < SAMPLE_RATE * DURATION:
            pad_length = int(SAMPLE_RATE * DURATION) - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        
        # Extract MFCCs with safe parameters
        mfccs = librosa.feature.mfcc(y=audio,
                                    sr=sr,
                                    n_mfcc=N_MFCC,
                                    n_fft=N_FFT,
                                    hop_length=HOP_LENGTH)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_data():
    features = []
    labels = []
    
    # Load wooden hits
    wooden_dir = os.path.join(DATA_PATH, "wooden_hit")
    for f in os.listdir(wooden_dir):
        feat = extract_features(os.path.join(wooden_dir, f))
        if feat is not None:
            features.append(feat)
            labels.append(1)
    
    # Load noise
    noise_dir = os.path.join(DATA_PATH, "noise/cut")
    for f in os.listdir(noise_dir):
        feat = extract_features(os.path.join(noise_dir, f))
        if feat is not None:
            features.append(feat)
            labels.append(0)
    
    return np.array(features), np.array(labels)

def train():
    X, y = load_data()
    
    # Check for class balance
    print(f"Class distribution: Wooden hits {sum(y)} | Noise {len(y)-sum(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Use linear SVM for efficiency
    model = SVC(kernel='linear', probability=True, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Sample predictions:", y_pred[:10])
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


    # for optimized c++ based inference in lowend rpi zero
    with open("model_weights.txt", "w") as f:
        # Save SVM coefficients (for linear kernel)
        f.write(f"{model.intercept_[0]}\n")
        for coef in model.coef_[0]:
            f.write(f"{coef}\n")
    
    # Save feature standardization parameters
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    with open("feature_scale.txt", "w") as f:
        for mean, std in zip(means, stds):
            f.write(f"{mean}\n{std}\n")

if __name__ == "__main__":
    train()