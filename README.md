**Emergency Call Audio Analysis & Priority Classification System**

**Status:** 🚧 In Development

**Project Overview :**

Emergency response systems receive thousands of calls daily. Quickly identifying the type and severity of an emergency is critical for dispatching the right resources in time.

This project builds an AI-powered emergency call classification system that analyzes 911 call recordings and automatically categorizes them into emergency types such as:

- EMS (Medical Emergency)

- Fire

- Traffic / Accident

The system uses speech recognition and machine learning to convert audio recordings into text and classify the type of emergency.

The project follows an MLOps workflow, integrating model training, deployment, containerization, and automation.

**Objectives :**

The key objectives of this project are:

- Convert 911 audio recordings into text using speech recognition

- Process extracted text using Natural Language Processing (NLP)

- Train a machine learning classifier to identify emergency categories

- Deploy the trained model as an API using FastAPI

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
Sample files are included.

**Workflow :**
1. Data Collection

- 911 emergency call recordings are downloaded from the Kaggle dataset.

2. Speech-to-Text Conversion

- Audio files are transcribed into text using the Whisper speech recognition model.

3. Data Processing

- The extracted text is cleaned and prepared for machine learning.

4. Feature Engineering

- Text data is converted into numerical features using TF-IDF vectorization.

5. Model Training

- Multiple machine learning models will be experimented.
- The best-performing model will be selected based on evaluation metrics

6. Model Deployment

- The trained model is exposed through a FastAPI-based prediction API.

7. Containerization

- The application is packaged using Docker to ensure consistent deployment.

8. CI/CD Automation

- GitHub Actions automates testing and builds whenever new code is pushed to the repository.
