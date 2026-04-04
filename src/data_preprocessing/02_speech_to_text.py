import os
import whisper
import pandas as pd
from tqdm import tqdm

# Paths
processed_path = "dataset/processed_audio"
transcript_path = "dataset/transcripts"

os.makedirs(transcript_path, exist_ok=True)

# Load model
model = whisper.load_model("base")

# Get files
files = sorted(os.listdir(processed_path))

print("Total files:", len(files))

# Split into batches
batch1 = files[:350]
batch2 = files[350:]

# Function to process batch
def process_batch(file_list):
    for file in tqdm(file_list):
        if file.endswith(".wav"):

            txt_file = file.replace(".wav", ".txt")
            txt_path = os.path.join(transcript_path, txt_file)

            # Skip if already processed
            if os.path.exists(txt_path):
                continue

            file_path = os.path.join(processed_path, file)

            result = model.transcribe(file_path)

            with open(txt_path, "w") as f:
                f.write(result["text"])

# Process both batches
process_batch(batch1)
process_batch(batch2)

# Create CSV
data = []

for file in os.listdir(transcript_path):
    if file.endswith(".txt"):
        with open(os.path.join(transcript_path, file)) as f:
            text = f.read()

        data.append([file, text])

df = pd.DataFrame(data, columns=["file", "transcript"])

df.to_csv("dataset/transcripts.csv", index=False)