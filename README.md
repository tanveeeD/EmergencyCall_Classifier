**Emergency Call Classification System**

**Project Overview :**

Emergency response systems receive thousands of calls daily. Quickly identifying the type and severity of an emergency is critical for dispatching the right resources in time.

This project processes emergency call audio or text and predicts the appropriate emergency category:

- EMS (Medical Emergency)

- Fire

- Traffic / Accident

The system uses speech recognition and machine learning to convert audio recordings into text and classify the type of emergency.

**Objectives :**

The key objectives of this project are:

- Convert 911 audio recordings into text using speech recognition

- Process extracted text using Natural Language Processing (NLP)

- Train a machine learning classifier to identify emergency categories

- Deploy the trained model using FastAPI and integrate it with a web-based user interface

- Containerize the system using Docker

- Implement CI/CD automation with GitHub Actions

**Dataset :**

This project uses the 911 Recordings Dataset available on Kaggle.

**Dataset Source:**
https://www.kaggle.com/datasets/louisteitelbaum/911-recordings

**Dataset Details**

- Contains real 911 emergency call recordings

- Each file represents a recorded emergency call

- Audio recordings are processed through speech-to-text transcription

- Generated text is used for machine learning classification
 
Due to size constraints, full dataset is not uploaded.

**Workflow :**
1. Data Collection

- 911 emergency call recordings are downloaded from the Kaggle dataset.

2. Audio Preprocessing

- Audio recordings are cleaned and standardized (e.g., noise reduction, resampling, format normalization)
- Ensures consistent input quality for speech recognition
  
3. Speech-to-Text Conversion
- Preprocessed audio files are transcribed into text using the Whisper model
  
4. Data Processing

- The extracted text is cleaned and prepared for machine learning.

5. Feature Representation

- Text is tokenized and encoded for transformer-based learning

6. Model Training

- A DistilBERT-based classification model is trained
- Performance is evaluated using standard metrics (accuracy, precision, recall, F1-score)

7. Model Deployment

- The trained `DistilBERT` and `Whisper` models are served robustly using **FastAPI**.
- The backend API is hosted on **Hugging Face Spaces** to bypass memory constraints, utilizing Git LFS to automatically pull and serve model weights.
- The interactive frontend is built using **Flask**, vanilla HTML/CSS/JS, and deployed on **Render**.

8. Interactive Web UI

- **Text Analysis**: Direct querying from dispatchers via text transcriptions.
- **Record Voice (Live Mic)**: Integration with the native `MediaRecorder` Web API allowing users to record emergency audio straight from the browser.
- **Audio Upload**: Ability to parse uploaded `.wav` or `.mp3` emergency recordings. 

9. Containerization & CI/CD

- The application is packaged using Docker for consistent, scalable environments.
- **GitHub Actions** automates code linting and dry-runs tests before integrating deployment webhooks (to automatically trigger Redeploys on Hugging Face and Render).

**Website Link:**
https://emergencycall-classifier-1.onrender.com
