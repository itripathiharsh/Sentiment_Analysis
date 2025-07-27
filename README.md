Sentiment Analysis & Mood Tracker
This project is a sophisticated sentiment analysis application designed to track and analyze emotions from text inputs over time. It uses natural language processing to understand the underlying sentiment and logs the data for detailed mood tracking and visualization.

🔴 Live Demo
You can interact with the live application here: https://thesentimentanalysisapp.streamlit.app/

✨ Features
Real-time Sentiment Analysis: Analyzes text input to determine the emotional tone.

Mood Tracking: Logs emotions from daily inputs to track mood patterns.

Data Logging: Saves questions, responses, and analysis results to CSV and JSON files for persistence.

Web Interface: Includes a simple web interface (App.py) to interact with the analysis engine.

📂 Project Structure
.
├── .gitignore               # Specifies files for Git to ignore
├── App.py                   # The main web application (e.g., Flask, Streamlit)
├── Sentiment_analysis.py    # Core logic for the sentiment analysis engine
├── daily_emotions.csv       # Log of detected emotions per day
├── mood_tracker_detailed.csv# Detailed log for mood tracking
├── question_emotions_over_time.csv # Tracks emotions related to specific questions
├── question_log.json        # Logs the questions asked
├── requirements.txt         # Lists all Python dependencies for the project
└── responses_2025-07-26.json # Example log of responses

🚀 Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites
Python 3.8 or higher

Git

Installation
Clone the repository:

git clone [https://github.com/itripathiharsh/Sentiment_Analysis.git](https://github.com/itripathiharsh/Sentiment_Analysis.git)
cd Sentiment_Analysis

Create and activate a virtual environment:
This keeps your project dependencies isolated.

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required dependencies:
This command reads the requirements.txt file and installs all necessary Python packages.

pip install -r requirements.txt

Set up your environment variables:
Create a file named .env in the root of your project folder. This file will hold your secret API keys. The .gitignore file is already configured to prevent this file from being uploaded to GitHub.

Open the .env file and add your Hugging Face API token like this:

HUGGING_FACE_API_KEY="your_new_api_key_here"

Replace "your_new_api_key_here" with your actual, new Hugging Face token.

Usage
To run the application, execute the App.py script from your terminal:

python App.py

This will start the local web server. Open your web browser and navigate to the address provided in the terminal (usually http://127.0.0.1:5000 or similar) to use the application.

📸 Screenshots
Here’s a sneak peek of the application in action.

Main Interface: <img width="1230" height="814" alt="Screenshot 2025-07-27 165136" src="https://github.com/user-attachments/assets/6278b8cc-8a06-4122-a9ae-c104d993648c" />

Analysis Results: <img width="1087" height="587" alt="Screenshot 2025-07-27 165204" src="https://github.com/user-attachments/assets/75b4ea8c-ce53-4243-8f11-f939cebd424a" />

<img width="1217" height="524" alt="Screenshot 2025-07-27 165217" src="https://github.com/user-attachments/assets/52329a36-1498-4997-8740-60bf40a0180f" />


