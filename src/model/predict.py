# Predict Script for Emergency Classification

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Path to saved model
MODEL_PATH = "models/distilbert_model"

# Label mapping
label_map = {0: "fire", 1: "police", 2: "medical"}


# Load model + tokenizer
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


# Predict function
def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred]


# Main function
def main():
    tokenizer, model = load_model()

    while True:
        text = input("\nEnter emergency text (or type 'exit'): ")

        if text.lower() == "exit":
            break

        result = predict(text, tokenizer, model)
        print(f"🚨 Prediction: {result}")


if __name__ == "__main__":
    main()
