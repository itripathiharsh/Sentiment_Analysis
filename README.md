# Advanced Sentiment Analysis & Mood Tracker
This is a full-stack web application that performs sophisticated sentiment analysis to track and analyze emotions from user inputs. It uses multiple AI models to understand emotional tone and mental state, logs the data to a persistent cloud database, and provides dynamic, AI-generated content to support the user's well-being.

🔴 Live Demo
You can interact with the live application here: https://thesentimentanalysisapp.streamlit.app/

📸 Screenshots
Here’s a sneak peek of the application in action.

Main Interface & Questions:
<img width="1230" alt="Main application interface" src="https://github.com/user-attachments/assets/6278b8cc-8a06-4122-a9ae-c104d993648c" />

Analysis Results:
<img width="1087" alt="Analysis results for emotions and mental state" src="https://github.com/user-attachments/assets/75b4ea8c-ce53-4243-8f11-f939cebd424a" />

Historical Data Trends:
<img width="1217" alt="Historical data trends and charts" src="https://github.com/user-attachments/assets/52329a36-1498-4997-8740-60bf40a0180f" />

✨ Features
Multi-Page Interface: A clean, navigable UI with separate sections for daily questions, today's results, and historical trends.

Dual-Model Sentiment Analysis: Utilizes two different transformer models for nuanced insights:

GoEmotions for detecting a wide range of 27 emotions.

MentalBERT for assessing text for signs of depression.

Persistent Cloud Storage: User entries are saved securely to a Google Firestore database, ensuring data is never lost and can be tracked over time.

Dynamic Daily Questions: Connects to Firestore to log and rotate daily reflection questions, ensuring users don't get repeats for at least four days.

AI-Generated Stories: Uses the Google Gemini API to find and summarize a real, public domain short story from a classic author that matches the user's dominant mood.

AI-Generated Activities: Leverages the Google Gemini API to generate personalized activities based on the user's mood, broken down into:

Short-Term activities for immediate relief.

Long-Term activities for sustained well-being.

Psychological techniques with explanations.

Secure Deployment: Deployed on Streamlit Cloud, using secrets management for Hugging Face, Google Cloud (Firebase), and Gemini API credentials.

📂 Project Structure
.
├── .gitignore               # Specifies files for Git to ignore
├── App.py                   # The main Streamlit web application
├── Sentiment_analysis.py    # Standalone script for local, terminal-based analysis
├── packages.txt             # System-level dependencies for Streamlit Cloud
└── requirements.txt         # Python dependencies for the project

Note: Local data files like CSVs are only generated when running Sentiment_analysis.py. The deployed app uses Firestore exclusively.

🚀 Getting Started (Local Development)
Follow these instructions to run the application on your local machine.

Prerequisites
Python 3.8+

Git

Installation
Clone the repository:

git clone [https://github.com/itripathiharsh/Sentiment_Analysis.git](https://github.com/itripathiharsh/Sentiment_Analysis.git)
cd Sentiment_Analysis

Create and activate a virtual environment:

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
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
