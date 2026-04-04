import os
from pydub import AudioSegment
from tqdm import tqdm
import librosa

# Paths
raw_path = "dataset/raw_audio"
processed_path = "dataset/processed_audio"

os.makedirs(processed_path, exist_ok=True)

# Get files
files = os.listdir(raw_path)

# Convert MP3 → WAV (with error handling)
for file in tqdm(files):
    if file.endswith(".mp3"):
        mp3_path = os.path.join(raw_path, file)
        wav_path = os.path.join(processed_path, file.replace(".mp3", ".wav"))

        try:
            audio = AudioSegment.from_file(mp3_path)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio.export(wav_path, format="wav")
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Optional: check one file
processed_files = os.listdir(processed_path)

if processed_files:
    file_path = os.path.join(processed_path, processed_files[0])

    audio, sr = librosa.load(file_path, sr=None)

    print("Sample rate:", sr)
    print("Duration:", len(audio) / sr)