Advanced Sentiment Analysis & Mood Tracker
This is a full-stack web application that performs sophisticated sentiment analysis to track and analyze emotions from user inputs. It uses natural language processing to understand emotional tone and mental state, logging the data to a persistent cloud database for detailed mood tracking and visualization.

🔴 Live Demo
You can interact with the live application here: Sentiment Analysis App

📸 Screenshots
Here’s a sneak peek of the application in action.

Main Interface & Questions:
<img width="1230" alt="Main application interface" src="https://github.com/user-attachments/assets/6278b8cc-8a06-4122-a9ae-c104d993648c" />

Analysis Results:
<img width="1087" alt="Analysis results for emotions and mental state" src="https://github.com/user-attachments/assets/75b4ea8c-ce53-4243-8f11-f939cebd424a" />

Historical Data Trends:
<img width="1217" alt="Historical data trends and charts" src="https://github.com/user-attachments/assets/52329a36-1498-4997-8740-60bf40a0180f" />

✨ Features
Dual-Model Analysis: Utilizes two different transformer models for nuanced insights:

GoEmotions for detecting a wide range of emotions.

MentalBERT for assessing text for signs of depression.

Persistent Data Storage: User entries are saved securely to a Google Firestore cloud database, ensuring data is never lost.

Historical Trend Analysis: Visualizes mood patterns and mental state scores over time with interactive charts.

Secure Deployment: Deployed on Streamlit Cloud, using secrets management for both Hugging Face and Google Cloud credentials.

Interactive Web Interface: A clean and user-friendly UI built with Streamlit.

📂 Project Structure
.
├── .gitignore               # Specifies files for Git to ignore
├── App.py                   # The main Streamlit web application
├── Sentiment_analysis.py    # Standalone script for local, terminal-based analysis
├── packages.txt             # System-level dependencies for Streamlit Cloud
└── requirements.txt         # Python dependencies for the project

Note: CSV and JSON log files are only generated when running Sentiment_analysis.py locally. The deployed app uses Firestore.

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

Set up your environment variables:
Create a file named .env in the project root. This file will hold your secret keys. The .gitignore file is already configured to prevent this file from being uploaded to GitHub.

Open the .env file and add your Hugging Face API token:

HUGGING_FACE_API_KEY="your_actual_hf_token_here"

(Note: To run the app locally with the Firestore database, you would also need to set up a Google Cloud service account key. See the App.py file for details on the required credentials.)

Usage
To run the web application locally, execute the following command:

streamlit run App.py

Open your web browser and navigate to the local address provided (usually http://localhost:8501).
