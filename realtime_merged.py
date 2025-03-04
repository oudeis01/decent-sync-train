import pyaudio
import numpy as np
import librosa
import joblib
import time
import os

# Configuration
MODEL_PATH = "wooden_hit_model.pkl"
TARGET_SAMPLE_RATE = 16000
CHUNK = 1024  # Increased chunk size
N_MFCC = 10
LED_PIN = 17
THRESHOLD = 0.85

# MFCC parameters
N_FFT = 512  # Keep this fixed for frequency resolution
HOP_LENGTH = 256  # Keep overlap for better temporal resolution

def get_audio_device():
    """Find any USB audio input device"""
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"Found device {i}: {dev['name']}")
            return i, int(dev['defaultSampleRate'])
    p.terminate()
    raise RuntimeError("No input devices found")

def get_h6_device(p):
    """Find Zoom H6 device index"""
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if 'H6' in dev['name'] and dev['maxInputChannels'] > 0:
            print(f"Found H6 at index {i}: {dev['name']}")
            return i, int(dev['defaultSampleRate'])
    raise RuntimeError("H6 microphone not found")

def process_audio(audio, original_rate):
    """Audio processing with buffer management"""
    audio = audio.astype(np.float32) / 32768.0
    
    # Resample if needed
    if original_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio, 
            orig_sr=original_rate, 
            target_sr=TARGET_SAMPLE_RATE,
            res_type='kaiser_fast'
        )
    
    # Pad audio if shorter than N_FFT
    if len(audio) < N_FFT:
        pad_size = N_FFT - len(audio)
        audio = np.pad(audio, (0, pad_size), mode='constant')
    
    # Extract MFCCs with safe parameters
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=40,
        fmax=8000,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return np.mean(mfccs.T, axis=0)

def main():
    model = joblib.load(MODEL_PATH)
    p = pyaudio.PyAudio()
    buffer_ready = False  # Track buffer status

    try:
        # check arch, if rpi, then use get_audio_device, 
        # else use get_h6_device
        if 'rpi' in os.uname().machine:
            device_index, hw_rate = get_audio_device()
        else:
            device_index, hw_rate = get_h6_device(p)
        # device_index, hw_rate = get_audio_device()
        
        print(f"Using device index {device_index} at {hw_rate}Hz")

        # Calculate required samples before resampling
        buffer_size = int(N_FFT * hw_rate / TARGET_SAMPLE_RATE)
        audio_buffer = np.array([], dtype=np.int16)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=hw_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
            start=False
        )

        stream.start_stream()
        time.sleep(2)
        print("Detection active...")

        while True:
            try:
                # Read audio chunk
                data = stream.read(CHUNK, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.int16)
                audio_buffer = np.concatenate((audio_buffer, new_samples))

                # Check buffer status
                if not buffer_ready and len(audio_buffer) >= buffer_size:
                    print("Buffer filled! Starting detection...")
                    buffer_ready = True

                # Process when we have enough samples
                while len(audio_buffer) >= buffer_size:
                    process_chunk = audio_buffer[:buffer_size]
                    audio_buffer = audio_buffer[buffer_size:]

                    features = process_audio(process_chunk, hw_rate)
                    prob = model.predict_proba([features])[0][1]
                    
                    if prob > THRESHOLD:
                        print(f"Hit detected! ({prob:.2%})")
                        
            except IOError as e:
                print(f"Audio buffer error: {str(e)}")
                time.sleep(0.1)
                
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()