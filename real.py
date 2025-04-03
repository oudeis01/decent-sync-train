import pyaudio
import numpy as np
import librosa
import joblib
import time
import os
import wave
from scipy import signal
from pythonosc import udp_client
from collections import deque

# Configuration
MODEL_PATH = "wooden_hit_model.pkl"
TARGET_SAMPLE_RATE = 16000
CHUNK = 512  # Keep aligned with buffer size
N_MFCC = 10
THRESHOLD = 0.70
TARGET_RMS = 0.08  # Adjusted for electret mic
SMOOTHING_FACTOR = 0.15

# Precomputed constants
N_FFT = 512
HOP_LENGTH = 256
MEL_FILTERS = librosa.filters.mel(sr=TARGET_SAMPLE_RATE, n_fft=N_FFT, n_mels=40, fmax=4000)

# OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

class AudioNormalizer:
    def __init__(self):
        self.rms_history = deque(maxlen=10)
        self.current_gain = 1.0
        self.raw_frames = []
        self.processed_frames = []

    def normalize_chunk(self, data):
        """Normalize audio chunk while preserving raw data"""
        # Store raw audio
        self.raw_frames.append(data)
        
        # Convert to float32 for processing
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        
        # Calculate RMS and update gain
        rms = np.sqrt(np.mean(audio**2)) + 1e-6
        self.rms_history.append(rms)
        avg_rms = np.mean(self.rms_history)
        target_gain = TARGET_RMS / avg_rms
        self.current_gain = SMOOTHING_FACTOR * target_gain + (1 - SMOOTHING_FACTOR) * self.current_gain
        
        # Apply gain and prevent clipping
        processed = np.clip(audio * self.current_gain, -1.0, 1.0)
        
        # Convert back to int16 and store
        processed_bytes = (processed * 32767).astype(np.int16).tobytes()
        self.processed_frames.append(processed_bytes)
        
        return processed_bytes

def process_audio(audio, original_rate):
    """Optimized audio pipeline with normalized input"""
    # Resample if needed
    if original_rate != TARGET_SAMPLE_RATE:
        audio = signal.resample_poly(audio, TARGET_SAMPLE_RATE, original_rate)
    
    # Fixed-size processing with zero-padding
    audio = np.pad(audio, (0, N_FFT - len(audio)))[:N_FFT]
    
    # Manual MFCC computation
    stft = np.abs(librosa.stft(audio.astype(np.float32)/32768.0, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mel = np.dot(MEL_FILTERS, stft)
    mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(mel), n_mfcc=N_MFCC)
    return np.mean(mfccs.T, axis=0)

def save_recordings(normalizer, hw_rate):
    """Save both raw and processed recordings"""
    with wave.open("raw_recording.wav", 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(hw_rate)
        wf.writeframes(b''.join(normalizer.raw_frames))
    
    with wave.open("processed_recording.wav", 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(b''.join(normalizer.processed_frames))

def main():
    p = pyaudio.PyAudio()
    stream = None
    normalizer = AudioNormalizer()
    
    try:
        device_index, hw_rate = get_audio_device()
        buffer_size = int(N_FFT * hw_rate / TARGET_SAMPLE_RATE)
        
        print(f"Starting detection on {hw_rate/1000}kHz input with normalization...")
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=hw_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
            start=True
        )

        audio_buffer = np.array([], dtype=np.int16)
        last_print = time.time()
        
        while True:
            # Read and normalize audio
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            processed_data = normalizer.normalize_chunk(raw_data)
            
            # Process normalized audio
            new_samples = np.frombuffer(processed_data, dtype=np.int16)
            audio_buffer = np.concatenate((audio_buffer, new_samples))
            
            if len(audio_buffer) >= buffer_size:
                chunk = audio_buffer[:buffer_size]
                audio_buffer = audio_buffer[buffer_size:]
                
                features = process_audio(chunk, TARGET_SAMPLE_RATE)
                prob = model.predict_proba([features])[0][1]
                
                if prob > THRESHOLD:
                    print(f"Detected! ({time.time()-last_print:.2f}s since last)")
                    client.send_message("/rotate", [3000, 50, 0])
                    last_print = time.time()

    except KeyboardInterrupt:
        print("\nStopping...")
        save_recordings(normalizer, hw_rate)
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

def get_audio_device():
    """Find USB audio interface with direct ALSA access"""
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0 and 'USB' in dev['name']:
            print(f"Found device {i}: {dev['name']}")
            return i, int(dev['defaultSampleRate'])
    p.terminate()
    raise RuntimeError("No USB audio device found")

if __name__ == "__main__":
    # Load model first
    model = joblib.load(MODEL_PATH)
    
    # Performance optimizations
    os.environ['LIBROSA_CACHE_LEVEL'] = '50'
    os.sched_setaffinity(0, {0})
    
    main()