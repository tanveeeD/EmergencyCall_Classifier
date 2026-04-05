FROM python:3.10

WORKDIR /app

COPY . .
# Install system requirements for Whisper (FFmpeg)
RUN apt-get update && apt-get install -y ffmpeg

RUN pip install --upgrade pip wheel "setuptools<70.0.0" gdown

# Fix Google Drive nested folder issue by searching and moving all downloaded files safely
RUN gdown --folder "https://drive.google.com/drive/folders/1meG2hvW5ED9slToZlQlEjfsthQDCoRlP?usp=drive_link" -O gdrive_download && \
    mkdir -p models/distilbert_model && \
    find gdrive_download -type f -exec mv {} models/distilbert_model/ \;; \
    rm -rf gdrive_download

RUN pip install --default-timeout=1000 --no-build-isolation -r requirements.txt

CMD ["uvicorn","api.app:app","--host","0.0.0.0","--port","8000"]
