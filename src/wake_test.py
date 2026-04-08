import os
os.environ["ONNXRUNTIME_DISABLE_GPU"] = "1"
import warnings
warnings.filterwarnings("ignore")

import pyaudio
import numpy as np
import wave
import subprocess
import os
import time
from openwakeword.model import Model

# Suppress ALSA/JACK warnings
import ctypes
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

# Audio config
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4  # how long to record after wake word
WAKE_THRESHOLD = 0.5
WHISPER_MODEL = os.path.expanduser("~/miles/whisper.cpp/models/ggml-base.en.bin")
WHISPER_CLI = os.path.expanduser("~/miles/whisper.cpp/build/bin/whisper-cli")
TEMP_WAV = os.path.expanduser("~/miles/build/command.wav")

# Load wake word model
print("Loading wake word model...")
model = Model(wakeword_model_paths=[os.path.expanduser("~/miles/models/hey_nova.onnx")])

# Find Razer mic
audio = pyaudio.PyAudio()
mic_index = None
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "Razer" in info["name"] or "Seiren" in info["name"]:
        mic_index = i
        print(f"Found mic: {info['name']} (index {i})")
        break

if mic_index is None:
    print("Razer mic not found!")
    exit(1)

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=mic_index,
    frames_per_buffer=CHUNK
)

def record_command():
    """Record audio after wake word is detected."""
    print("Listening for command...")
    frames = []
    num_chunks = int(RATE / CHUNK * RECORD_SECONDS)
    for _ in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    # Save to WAV file
    wf = wave.open(TEMP_WAV, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return TEMP_WAV

def transcribe(wav_path):
    """Run Whisper on the recorded audio."""
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path,
         "-bs", "1", "-bo", "1", "--no-prints", "--no-timestamps"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

print("\n=== M.I.L.E.S. v0.1 ===")
print("Listening for wake word... (Ctrl+C to stop)\n")

try:
    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        prediction = model.predict(audio_array)
        
        for wake_word, score in prediction.items():
            if score > WAKE_THRESHOLD:
                print(f"Wake word detected! (confidence: {score:.2f})")
                
                # Cooldown: flush the audio buffer so leftover
                # wake word audio doesn't re-trigger
                for _ in range(int(RATE / CHUNK * 0.5)):
                    stream.read(CHUNK, exception_on_overflow=False)
                model.reset()
                
                # Record the command
                wav_path = record_command()
                
                # Transcribe it
                start = time.time()
                text = transcribe(wav_path)
                elapsed = time.time() - start
                
                print(f"You said: {text}")
                print(f"Transcription time: {elapsed:.2f}s")
                print("Listening for wake word...\n")

except KeyboardInterrupt:
    print("\nShutting down M.I.L.E.S.")
    stream.stop_stream()
    stream.close()
    audio.terminate()