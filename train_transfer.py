import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import joblib
from tqdm import tqdm

# Configuration
YAMNET_URL = 'https://tfhub.dev/google/yamnet/1'
DATA_PATH = "training_data/"
MODEL_PATH = "transfer_model.pkl"
SAMPLE_RATE = 16000
DURATION = 0.975  # Must be exactly 975ms
SAMPLES_NEEDED = int(SAMPLE_RATE * DURATION)  # 15600

# Load YAMNet with custom batch handling
yamnet = hub.load(YAMNET_URL)

# Create batched version of YAMNet
@tf.function(input_signature=[tf.TensorSpec(shape=[None, SAMPLES_NEEDED], dtype=tf.float32)])
def yamnet_batched(audio_batch):
    return tf.map_fn(
        lambda x: yamnet(x)[1],  # Extract embeddings
        audio_batch,
        fn_output_signature=tf.TensorSpec(shape=[None, 1024], dtype=tf.float32)
    )

def process_file(file_path):
    """Process single file with exact requirements"""
    audio = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)[0]
    if len(audio) < SAMPLES_NEEDED:
        audio = np.pad(audio, (0, SAMPLES_NEEDED - len(audio)))
    return audio.astype(np.float32)

def extract_features(files):
    """Process files in batches with proper tensor shaping"""
    features = []
    batch_size = 16  # Reduced for stability
    
    for i in tqdm(range(0, len(files), batch_size)):
        batch_files = files[i:i+batch_size]
        
        # Load and pad audio
        audio_batch = np.array([process_file(f) for f in batch_files])
        
        # Convert to tensor and process
        audio_tensor = tf.constant(audio_batch, dtype=tf.float32)
        embeddings = yamnet_batched(audio_tensor).numpy()
        features.extend([emb.mean(axis=0) for emb in embeddings])
    
    return np.array(features)

def main():
    # Get file paths
    wooden_files = [os.path.join(DATA_PATH, "wooden_hit", f) 
                   for f in os.listdir(os.path.join(DATA_PATH, "wooden_hit"))]
    noise_files = [os.path.join(DATA_PATH, "noise/cut", f) 
                  for f in os.listdir(os.path.join(DATA_PATH, "noise/cut"))]
    
    # Extract features
    print("Processing wooden hits...")
    wooden_features = extract_features(wooden_files)
    print("Processing noise...")
    noise_features = extract_features(noise_files)
    
    # Create dataset
    X = np.vstack([wooden_features, noise_features])
    y = np.array([1]*len(wooden_files) + [0]*len(noise_files))
    
    # Train classifier
    from sklearn.svm import SVC
    model = SVC(kernel='linear', class_weight='balanced', probability=True)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    # Suppress TF logging and CUDA warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    main()