import pandas as pd

# Load dataset
df = pd.read_csv("dataset/final_dataset.csv")

# Remove nulls
df = df.dropna()

# Clean transcript
df["transcript"] = df["transcript"].str.lower().str.strip()

# Remove empty and very short text
df = df[df["transcript"] != ""]
df = df[df["transcript"].str.len() > 5]

# Clean labels
df["label"] = df["label"].str.lower().str.strip()

# Reset index
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv("dataset/final_dataset.csv", index=False)