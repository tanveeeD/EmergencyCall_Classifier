from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import whisper
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/distilbert_model"
label_map = {0: "fire", 1: "police", 2: "medical"}

print("Loading models...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

speech_model = whisper.load_model("base")

print("Models loaded!")

@app.post("/predict-text")
def predict_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return {"prediction": label_map[pred]}

@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):

    try:
        temp_file = "temp_audio.wav"

        # Save file
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Convert audio → text
        result = speech_model.transcribe(temp_file)
        text = result["text"]

        # Predict
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()

        os.remove(temp_file)

        return {
            "transcript": text,
            "prediction": label_map[pred]
        }

    except Exception as e:
        return {"error": str(e)}