import pyaudio
import numpy as np
import tensorflow_hub as hub
import joblib
import time
import os
import librosa

# Configuration
MODEL_PATH = "transfer_model.pkl"
YAMNET_URL = 'https://tfhub.dev/google/yamnet/1'
TARGET_SAMPLE_RATE = 16000
CHUNK = 1024
THRESHOLD = 0.85

# YAMNet requirements
YAMNET_DURATION = 0.975  # 975ms required by YAMNet
SAMPLES_NEEDED = int(TARGET_SAMPLE_RATE * YAMNET_DURATION)  # 15600

def get_h6_device(p):
    """Find Zoom H6 device index with exact matching"""
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if 'H6' in dev['name'].upper() and dev['maxInputChannels'] > 0:
            print(f"Found H6 at index {i}: {dev['name']}")
            return i
    raise RuntimeError("H6 microphone not found")

def process_audio(audio, original_rate, yamnet):
    """Process audio through YAMNet with H6-specific resampling"""
    # Convert to float32 and normalize
    audio = audio.astype(np.float32) / 32768.0
    
    # Resample if H6 uses different rate
    if original_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=original_rate,
            target_sr=TARGET_SAMPLE_RATE,
            res_type='kaiser_fast'
        )
    
    # Ensure exact length for YAMNet
    if len(audio) < SAMPLES_NEEDED:
        audio = np.pad(audio, (0, SAMPLES_NEEDED - len(audio)))
    else:
        audio = audio[:SAMPLES_NEEDED]
    
    # Get YAMNet embeddings
    _, embeddings, _ = yamnet(audio)
    return np.mean(embeddings.numpy(), axis=0)

def main():
    # Suppress ALSA warnings
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Load models
    yamnet = hub.load(YAMNET_URL)
    model = joblib.load(MODEL_PATH)
    
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        # Find and configure H6
        device_index = get_h6_device(p)
        device_info = p.get_device_info_by_index(device_index)
        hw_rate = int(device_info['defaultSampleRate'])
        
        buffer = np.array([], dtype=np.int16)
        
        print(f"Using H6 at {hw_rate}Hz")
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=hw_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
            start=True
        )
        
        print("Listening for wooden hits...")
        while True:
            # Read audio chunk from H6
            data = stream.read(CHUNK, exception_on_overflow=False)
            new_samples = np.frombuffer(data, dtype=np.int16)
            buffer = np.concatenate((buffer, new_samples))
            
            # Process when enough samples accumulated (after resampling)
            required_samples = int(SAMPLES_NEEDED * hw_rate / TARGET_SAMPLE_RATE)
            if len(buffer) >= required_samples:
                audio_window = buffer[:required_samples]
                buffer = buffer[required_samples:]
                
                features = process_audio(audio_window, hw_rate, yamnet)
                prob = model.predict_proba([features])[0][1]
                
                if prob > THRESHOLD:
                    print(f"Wooden hit detected! ({prob:.2%})")
                    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    main()