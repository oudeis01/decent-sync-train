import pyaudio
import numpy as np
import librosa
import joblib
import time
import os
from scipy import signal
from pythonosc import udp_client

# Configuration
MODEL_PATH = "wooden_hit_model.pkl"
TARGET_SAMPLE_RATE = 16000
CHUNK = 512  # Reduced for faster buffer filling
N_MFCC = 10
THRESHOLD = 0.70

# Precomputed constants
N_FFT = 512
HOP_LENGTH = 256
MEL_FILTERS = librosa.filters.mel(sr=TARGET_SAMPLE_RATE, n_fft=N_FFT, n_mels=40, fmax=4000)

# OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

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

def process_audio(audio, original_rate):
    """Optimized audio pipeline for Pi Zero"""
    # Resample using faster method
    if original_rate != TARGET_SAMPLE_RATE:
        audio = signal.resample_poly(audio, TARGET_SAMPLE_RATE, original_rate)
    
    # Fixed-size processing with zero-padding
    audio = np.pad(audio, (0, N_FFT - len(audio)))[:N_FFT]
    
    # Manual MFCC computation
    stft = np.abs(librosa.stft(audio.astype(np.float32)/32768.0, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mel = np.dot(MEL_FILTERS, stft)
    mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(mel), n_mfcc=N_MFCC)
    return np.mean(mfccs.T, axis=0)

def main():
    global client
    model = joblib.load(MODEL_PATH)
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        device_index, hw_rate = get_audio_device()
        buffer_size = int(N_FFT * hw_rate / TARGET_SAMPLE_RATE)
        
        print(f"Starting detection on {hw_rate/1000}kHz input...")
        
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
            # Read and buffer audio
            data = stream.read(CHUNK, exception_on_overflow=False)
            new_samples = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.concatenate((audio_buffer, new_samples))
            
            # Process when buffer filled
            if len(audio_buffer) >= buffer_size:
                chunk = audio_buffer[:buffer_size]
                audio_buffer = audio_buffer[buffer_size:]
                
                features = process_audio(chunk, hw_rate)
                prob = model.predict_proba([features])[0][1]
                
                if prob > THRESHOLD:
                    print(f"Detected! ({time.time()-last_print:.2f}s since last)")
                    client.send_message("/rotate", [3000, 50, 0])
                    print(prob)
                    last_print = time.time()
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    
    # Performance optimizations
    os.environ['LIBROSA_CACHE_LEVEL'] = '50'
    os.sched_setaffinity(0, {0})  # Lock to CPU core 0
    main()
