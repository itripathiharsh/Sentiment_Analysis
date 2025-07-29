# Advanced Sentiment Analysis & Mood Tracker
This is a full-stack web application that performs sophisticated sentiment analysis to track and analyze emotions from user inputs. It uses multiple AI models to understand emotional tone and mental state, logs the data to a persistent cloud database, and provides dynamic, AI-generated content to support the user's well-being.

🔴 Live Demo
You can interact with the live application here: https://thesentimentanalysisapp.streamlit.app/

# 📸 Screenshots
Here’s a sneak peek of the application in action.

## Main Interface & Questions:
<img width="1908" height="842" alt="Screenshot 2025-07-29 100613" src="https://github.com/user-attachments/assets/6c5db608-d324-4278-8321-8e2b647dbf5a" />

## Story and Activities
<img width="1543" height="881" alt="Screenshot 2025-07-29 100715" src="https://github.com/user-attachments/assets/5c2e47b2-019a-4f0b-a12d-b2a12763a256" />
<img width="1228" height="765" alt="Screenshot 2025-07-29 100731" src="https://github.com/user-attachments/assets/7f29a262-0e61-4888-bbd3-ffdd4466a06b" />


## Analysis Results:
<img width="1386" height="394" alt="Screenshot 2025-07-29 100650" src="https://github.com/user-attachments/assets/f2e88b93-1c7c-477c-849d-79d3286fda9e" />


## Historical Data Trends:
<img width="1520" height="588" alt="Screenshot 2025-07-29 100832" src="https://github.com/user-attachments/assets/dd12b76a-d85d-45d5-ac4d-928eac51478f" />
<img width="630" height="392" alt="Screenshot 2025-07-29 100811" src="https://github.com/user-attachments/assets/15e6c789-0b31-49e2-aa68-2a07031232b5" />

![Up<img width="1498" height="551" alt="Screenshot 2025-07-29 100845" src="https://github.com/user-attachments/assets/6ff36c38-2716-485c-bc98-ba0b7988cac0" />
loading Screenshot 2025-07-29 100832.png…]()


## ✨ Features
**Multi-Page Interface:** A clean, navigable UI with separate sections for daily questions, today's results, and historical trends.

**Dual-Model Sentiment Analysis:** Utilizes two different transformer models for nuanced insights:

**GoEmotions** for detecting a wide range of 27 emotions.

**MentalBERT** for assessing text for signs of depression.

**Persistent Cloud Storage:** User entries are saved securely to a Google Firestore database, ensuring data is never lost and can be tracked over time.

**Dynamic Daily Questions:** Connects to Firestore to log and rotate daily reflection questions, ensuring users don't get repeats for at least four days.

**AI-Generated Stories:** Uses the Google Gemini API to find and summarize a real, public domain short story from a classic author that matches the user's dominant mood.

**AI-Generated Activities:** Leverages the Google Gemini API to generate personalized activities based on the user's mood, broken down into:

**Short-Term activities** for immediate relief.

**Long-Term activities** for sustained well-being.

**Psychological technique**s with explanations.

**Secure Deployment:** Deployed on Streamlit Cloud, using secrets management for Hugging Face, Google Cloud (Firebase), and Gemini API credentials.

## 📂 Project Structure
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

# Create and activate a virtual environment:

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

# Hugging Face API Key
HUGGING_FACE_API_KEY = "your_hf_token_here"

# Numbered Google Gemini API Keys
GEMINI_API_KEY_1 = "your_gemini_key_1"
GEMINI_API_KEY_2 = "your_gemini_key_2"

# Firebase
