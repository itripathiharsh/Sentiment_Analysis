# Sentiment Analysis & Mood Tracker
This project is a sophisticated sentiment analysis application designed to track and analyze emotions from text inputs over time. It uses natural language processing to understand the underlying sentiment and logs the data for detailed mood tracking and visualization.

## 🔴 Live Demo
You can interact with the live application here: Sentiment Analysis App

Note: The live demo runs in a temporary environment. Data entered there will not be saved permanently.

### 📸 Screenshots
Here’s a sneak peek of the application in action.

### Main Interface & Questions:
<img width="1230" alt="Main application interface" src="https://github.com/user-attachments/assets/6278b8cc-8a06-4122-a9ae-c104d993648c" />

### Analysis Results:
<img width="1087" alt="Analysis results for emotions and mental state" src="https://github.com/user-attachments/assets/75b4ea8c-ce53-4243-8f11-f939cebd424a" />

### Historical Data Trends:
<img width="1217" alt="Historical data trends and charts" src="https://github.com/user-attachments/assets/52329a36-1498-4997-8740-60bf40a0180f" />

## ✨ Features
**Dual-Model Analysis:** Utilizes two different transformer models for nuanced insights:

**GoEmotions** for detecting a wide range of emotions.

**MentalBERT** for assessing text for signs of depression.

**Local Data Logging:** Saves questions, responses, and analysis results to CSV and JSON files when run on your local machine.

**Historical Trend Analysis:** Visualizes mood patterns from locally saved data.

**Secure Deployment:** Deployed on Streamlit Cloud, using secrets management for the Hugging Face API token.

**Interactive Web Interface:** A clean and user-friendly UI built with Streamlit.

## 📂 Project Structure
.
├── .gitignore               # Specifies files for Git to ignore
├── App.py                   # The main Streamlit web application
├── Sentiment_analysis.py    # Standalone script for local, terminal-based analysis
├── packages.txt             # System-level dependencies for Streamlit Cloud
└── requirements.txt         # Python dependencies for the project

## 🚀 Getting Started (Local Development)
Follow these instructions to run the application on your local machine.

**Prerequisites**
Python 3.8+

Git

**Installation**
Clone the repository:

**git clone** [https://github.com/itripathiharsh/Sentiment_Analysis.git](https://github.com/itripathiharsh/Sentiment_Analysis.git)
cd Sentiment_Analysis

### Create and activate a virtual environment:

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

**Install the required dependencies:**

pip install -r requirements.txt

Set up your environment variables:
Create a file named .env in the project root. This file will hold your secret keys. The .gitignore file is already configured
