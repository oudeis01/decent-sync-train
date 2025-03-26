import pyaudio
import numpy as np
import librosa
import joblib
import time
import os
from scipy import signal
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
MODEL_PATH = "wooden_hit_model.pkl"
TARGET_SAMPLE_RATE = 16000
CHUNK = 512  # Optimal for Pi Zero
THRESHOLD = 0.7  # Adjusted threshold
BUFFER_SECONDS = 0.5  # Analysis window

# Precomputed Mel filter bank
N_FFT = 512
MEL_FILTERS = librosa.filters.mel(sr=TARGET_SAMPLE_RATE, n_fft=N_FFT, n_mels=40, fmax=4000)

def get_h6_device(p):
    """Find Zoom H6 device index"""
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if 'H6' in dev['name'] and dev['maxInputChannels'] > 0:
            print(f"Found H6 at index {i}: {dev['name']}")
            return i, int(dev['defaultSampleRate'])
    raise RuntimeError("H6 microphone not found")

def get_audio_device():
    """Find any USB audio input device"""
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            if 'umc'.lower() in dev['name'].lower():
                print(f"Found device {i}: {dev['name']}")
                return i, int(dev['defaultSampleRate'])
    p.terminate()
    raise RuntimeError("No input devices found")

def process_audio(audio, original_rate):
    """Optimized audio processing pipeline"""
    # Resample if needed
    if original_rate != TARGET_SAMPLE_RATE:
        audio = signal.resample_poly(audio, TARGET_SAMPLE_RATE, original_rate)
    
    # Normalize and trim
    audio = audio.astype(np.float32) / 32768.0
    audio = audio[:int(TARGET_SAMPLE_RATE * BUFFER_SECONDS)]
    audio = np.pad(audio, (0, max(0, int(TARGET_SAMPLE_RATE * BUFFER_SECONDS) - len(audio))))
    
    # Extract features
    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=N_FFT//2))
    mel = np.dot(MEL_FILTERS, stft)
    mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(mel), n_mfcc=10)
    return np.mean(mfccs.T, axis=0)

def main():
    # Environment setup
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Load model
    model = joblib.load(MODEL_PATH)
    
    # Audio setup
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        # check arch, if rpi, then use get_audio_device, 
        # else use get_h6_device
        if 'rpi' in os.uname().machine:
            device_index, hw_rate = get_audio_device()
        else:
            device_index, hw_rate = get_h6_device(p)
        
        # Calculate buffer
        buffer_size = int(TARGET_SAMPLE_RATE * BUFFER_SECONDS * hw_rate / TARGET_SAMPLE_RATE)
        audio_buffer = np.array([], dtype=np.int16)
        
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=hw_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
            start=True
        )
        
        print(f"Listening on device {device_index} ({hw_rate}Hz)...")
        
        while True:
            try:
                # Read audio
                data = stream.read(CHUNK, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.int16)
                audio_buffer = np.concatenate((audio_buffer, new_samples))
                
                # Process when buffer filled
                if len(audio_buffer) >= buffer_size:
                    chunk = audio_buffer[:buffer_size]
                    audio_buffer = audio_buffer[buffer_size:]
                    
                    # Extract features and predict
                    features = process_audio(chunk, hw_rate)
                    prob = model.predict_proba([features])[0][1]
                    
                    if prob > THRESHOLD:
                        print(f"🎯 Detection ({(prob*100):.1f}%)")
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                audio_buffer = np.array([], dtype=np.int16)
                
    finally:
        # Cleanup
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    main()