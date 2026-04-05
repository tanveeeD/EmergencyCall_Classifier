import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

DATA_PATH = "src/data_preprocessing/final_dataset.csv"
MODEL_PATH = "models/distilbert_model"

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df["transcript"] = df["transcript"].str.lower().str.strip()
    df["label"] = df["label"].str.lower().str.strip()
    return df


def encode_labels(df):
    label_map = {"fire": 0, "police": 1, "medical": 2}
    df["label"] = df["label"].map(label_map)
    return df



class EmergencyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def train():
    df = load_data(DATA_PATH)
    df = encode_labels(df)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["transcript"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True
    )

    train_dataset = EmergencyDataset(train_encodings, train_labels)
    test_dataset = EmergencyDataset(test_encodings, test_labels)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    
    trainer.train()

    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    print("✅ Training completed and model saved!")



if __name__ == "__main__":
    train()
