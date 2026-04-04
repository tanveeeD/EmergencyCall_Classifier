# Evaluate DistilBERT Model

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = "data/final_dataset.csv"
MODEL_PATH = "models/distilbert_model"


# Load Data
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df["transcript"] = df["transcript"].str.lower().str.strip()
    df["label"] = df["label"].str.lower().str.strip()
    return df


# Encode Labels
def encode_labels(df):
    label_map = {"fire": 0, "police": 1, "medical": 2}
    df["label"] = df["label"].map(label_map)
    return df


# Main Evaluation
def evaluate():

    # Load and preprocess data
    df = load_data(DATA_PATH)
    df = encode_labels(df)

    # Train-test split (same as training)
    _, test_texts, _, test_labels = train_test_split(
        df["transcript"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    # Load model + tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

    model.eval()

    predictions = []

    # Predict
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(pred)

    # Metrics
    print("Accuracy:", accuracy_score(test_labels, predictions))
    print("\nClassification Report:\n")
    print(classification_report(test_labels, predictions))


# Run
if __name__ == "__main__":
    evaluate()
