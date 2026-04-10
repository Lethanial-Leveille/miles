import os
os.environ["ONNXRUNTIME_DISABLE_GPU"] = "1"
os.environ["JACK_NO_START_SERVER"] = "1"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import pyaudio
import numpy as np
import wave
import subprocess

# Suppress ALSA warnings
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

from resemblyzer import VoiceEncoder, preprocess_wav

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1280
RECORD_SECONDS = 5
VOICEPRINT_PATH = os.path.expanduser("~/miles/models/voiceprint.npy")
TEMP_ENROLL = os.path.expanduser("~/miles/build/enroll_temp.wav")

PHRASES = [
    "Hey Nova, what's the weather today",
    "Set a timer for fifteen minutes",
    "Tell me about my schedule for tomorrow morning",
    "Lock in",
    "What do you think I should do",
]

# Find mic
audio = pyaudio.PyAudio()
mic_index = None
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "Razer" in info["name"] or "Seiren" in info["name"]:
        mic_index = i
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

def record_sample():
    frames = []
    num_chunks = int(RATE / CHUNK * RECORD_SECONDS)
    for _ in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    wf = wave.open(TEMP_ENROLL, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def playback():
    subprocess.run(["aplay", "-D", "plughw:2,0", TEMP_ENROLL], capture_output=True)

print("=== M.I.L.E.S. Voice Enrollment ===\n")
print("You will record 5 samples of your voice.")
print("Speak naturally at the distance you'll normally use.\n")

encoder = VoiceEncoder()
embeddings = []

i = 0
while i < len(PHRASES):
    print(f"Sample {i + 1}/5: \"{PHRASES[i]}\"")
    cmd = input("Press Enter to record, 's' to skip, 'q' to quit: ").strip().lower()
    
    if cmd == 'q':
        print("Enrollment cancelled.")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        exit(0)
    
    if cmd == 's':
        print("Skipped.\n")
        i += 1
        continue
    
    print("Recording 5 seconds...")
    record_sample()
    print("Done. Playing back...")
    playback()
    
    while True:
        choice = input("Keep (y), redo (n), or replay (r)? ").strip().lower()
        if choice == 'r':
            playback()
        elif choice == 'n':
            print("Redoing sample.\n")
            break
        else:
            wav = preprocess_wav(TEMP_ENROLL)
            embedding = encoder.embed_utterance(wav)
            embeddings.append(embedding)
            print(f"Sample {i + 1} captured.\n")
            i += 1
            break

if len(embeddings) < 3:
    print(f"Only {len(embeddings)} samples captured. Need at least 3 for a reliable voiceprint.")
    print("Run enrollment again.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    exit(1)

voiceprint = np.mean(embeddings, axis=0)
np.save(VOICEPRINT_PATH, voiceprint)

stream.stop_stream()
stream.close()
audio.terminate()

print(f"\nVoiceprint saved ({len(embeddings)} samples averaged)")
print("Enrollment complete.")