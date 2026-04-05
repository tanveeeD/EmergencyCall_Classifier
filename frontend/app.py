import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/text", methods=["POST"])
def predict_text():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # The backend expects the text as a query parameter in /predict-text?text=...
        response = requests.post(f"{BACKEND_URL}/predict-text", params={"text": text})
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/audio", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        files = {"file": (file.filename, file.stream, file.mimetype)}
        response = requests.post(f"{BACKEND_URL}/predict-audio", files=files)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
