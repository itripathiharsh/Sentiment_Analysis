# Advanced AI-Powered Sentiment & Mood Journal
This is a full-stack web application that serves as a private, intelligent journal. It performs sophisticated sentiment analysis to track and analyze emotions from user inputs, logs data to a persistent cloud database, and provides dynamic, AI-generated content to support the user's well-being.

## 🔴 Live Demo
You can interact with the live application here: [Sentiment Analysis App](https://thesentimentanalysisapp.streamlit.app/)

## 📸 Screenshots
### Main Interface & Multi-Language Support
<img width="1849" height="832" alt="Screenshot 2025-08-01 003719" src="https://github.com/user-attachments/assets/7e4ec1b7-2617-4388-bd43-ae1533bfdf71" />

### Daily Journal Section
<img width="1815" height="904" alt="Screenshot 2025-08-01 003202" src="https://github.com/user-attachments/assets/c7bf8ddc-c843-46ca-bcd6-3fc3ed2db3bf" />
<img width="1485" height="849" alt="Screenshot 2025-08-01 003218" src="https://github.com/user-attachments/assets/e3ace297-4abb-46a5-b6da-bb29c5195630" />


### AI-Generated Story with Audio & Activity Suggestions
<img width="1543" alt="AI-generated story with audio player" src="https://github.com/user-attachments/assets/5c2e47b2-019a-4f0b-a12d-b2a12763a256" />
<img width="1228" alt="AI-generated activity suggestions" src="https://github.com/user-attachments/assets/7f29a262-0e61-4888-bbd3-ffdd4466a06b" />

### Analysis and Trends
<img width="1446" height="665" alt="Screenshot 2025-08-01 003235" src="https://github.com/user-attachments/assets/bf088735-d038-4752-a40a-d7f46670e02b" />
<img width="1440" height="695" alt="Screenshot 2025-08-01 003248" src="https://github.com/user-attachments/assets/92fe3bb7-1707-4f49-81ad-8c0099c4437e" />
<img width="1459" height="566" alt="Screenshot 2025-08-01 003301" src="https://github.com/user-attachments/assets/4f6a39ea-3855-4199-a7ff-54a8fbb63363" />


## ✨ Features
**Secure Multi-User Authentication:** A complete login/signup system ensures that all user data, including journal and mood entries, is kept private.

**Persistent Cloud Storage:** All user data is saved securely to a Google Firestore database.

**Multi-Page Interface:** A clean, navigable UI with separate sections for the daily questionnaire, a private journal, today's results, and historical trends.

**Dual-Model Sentiment Analysis:** Utilizes two different transformer models for nuanced insights:

**GoEmotions** for detecting a wide range of 27 emotions.

**MentalBERT** for assessing text for signs of depression.

**Gamified Streaks:** Motivates users by tracking and displaying their consecutive daily streaks for both reflections and journal entries.

**Advanced Trend Analysis:** The "Trends" page features multiple visualizations:

A **pie chart of dominant emotions** over the last 30 days.

A **stacked area** chart comparing **positive vs. negative emotion** trends.

**AI-Generated Stories with Audio:** Uses the Google Gemini API to find and summarize a real, public domain short story from a classic author that matches the user's dominant mood. Includes a text-to-speech feature to listen to the story.

**AI-Generated Activities:** Leverages the Google Gemini API to generate personalized activities based on the user's mood, broken down into short-term, long-term, and psychological techniques.

**Multi-Language Support:** Users can select their preferred language, and the app uses the Google Gemini API to translate the interface, questions, and all generated content.

## 🛠️ Technology Stack
**Frontend:** Streamlit

**Backend & Database:** Google Firestore

**Sentiment Analysis:** Hugging Face Transformers (PyTorch)

**Generative AI:** Google Gemini API

**Deployment:** Streamlit Cloud

**📂 Project Structure**
.
├── .gitignore               # Specifies files for Git to ignore
├── App.py                   # The main Streamlit web application
├── Sentiment_analysis.py    # Standalone script for local, terminal-based analysis
├── packages.txt             # System-level dependencies for Streamlit Cloud
└── requirements.txt         # Python dependencies for the project

Note: Local data files like CSVs are only generated when running Sentiment_analysis.py. The deployed app uses Firestore exclusively.

## 🚀 Getting Started (Local Development)
Follow these instructions to run the application on your local machine.

Prerequisites
Python 3.8+

Git

Installation
Clone the repository:

git clone [https://github.com/itripathiharsh/Sentiment_Analysis.git](https://github.com/itripathiharsh/Sentiment_Analysis.git)
cd Sentiment_Analysis

Create and activate a virtual environment:

## For Windows
python -m venv venv
.\venv\Scripts\activate

## For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

Set up your local secrets:
Create a folder named .streamlit in the project root, and inside it, create a file named secrets.toml. This file will hold all your secret API keys. The .gitignore file is already configured to prevent this folder from being uploaded to GitHub.

Open secrets.toml and add your keys in the following format:

## Secret key for hashing passwords (can be any random string)
SECRET_KEY = "your_strong_secret_key_here"

## Hugging Face API Key
HUGGING_FACE_API_KEY = "your_hf_token_here"

## Numbered Google Gemini API Keys
GEMINI_API_KEY_1 = "your_gemini_key_1"
GEMINI_API_KEY_2 = "your_gemini_key_2"

## Firebase Service Account Credentials
[firebase]
type = "service_account"
project_id = "your-project-id"
### ... (copy all other fields from your Firebase JSON key file) ...

Usage
To run the web application locally, execute the following command:

streamlit run App.py

Open your web browser and navigate to the local address provided (usually http://localhost:8501).
