import subprocess
import os
import random

output_dir = os.path.expanduser("~/miles/models/training_data/hey_nova")
os.makedirs(output_dir, exist_ok=True)

# Different Piper voice models to create variety
# We'll download a few different voices
voices = {
    "lessac": "en_US-lessac-medium",
    "amy": "en_GB-amy-medium",
    "ryan": "en_US-ryan-medium",
}

# Variations of the wake phrase
phrases = [
    "hey nova",
    "hey Nova",
    "Hey nova",
    "Hey Nova",
]

count = 0
for voice_name, voice_model in voices.items():
    for phrase in phrases:
        for speed in [0.8, 0.9, 1.0, 1.1, 1.2]:
            output_file = os.path.join(output_dir, f"hey_nova_{count:04d}.wav")
            cmd = (
                f'echo "{phrase}" | '
                f'piper --model {voice_model} '
                f'--length-scale {1/speed:.2f} '
                f'--output_file {output_file}'
            )
            try:
                subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
                if os.path.exists(output_file):
                    count += 1
                    print(f"Generated {count}: {voice_name} speed={speed}")
            except:
                print(f"Failed: {voice_name} speed={speed}")

print(f"\nTotal clips generated: {count}")
