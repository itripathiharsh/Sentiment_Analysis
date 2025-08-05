# Green Minds 🌱
**An AI-Powered Mental Wellness Journal & Toolkit**



**[Green Minds](https://thesentimentanalysisapp.streamlit.app/)**



**Green Minds** is a comprehensive, AI-enhanced web application designed to be a companion for mental and spiritual well-being. It combines daily journaling with advanced sentiment analysis, a suite of cognitive and mindfulness exercises, and daily wisdom from ancient texts to provide a holistic approach to mental health.

# ✨ Key Features
## 🧠 Core AI & Journaling
**AI-Powered Daily Analysis:** Answer daily rotating questions and receive a detailed analysis of your emotional state (powered by GoEmotions) and mental state (powered by a fine-tuned bertweet sentiment model).

**Secure Daily Journal:** A private, secure space to write down your thoughts and feelings, with entries saved to Firestore.

**Anonymous Community Sharing:** Optionally share your journal entries anonymously with the community to give and receive support.

**AI-Generated Stories & Activities:** Based on your mood analysis, the app provides personalized, AI-generated short stories and wellness activities using a robust fallback system (Groq & Gemini).

## 🧘 Wellness Toolkit
**Cognitive Games:** A suite of games to improve focus, memory, and processing speed, including:

**Number Tap Challenge:** Test your speed and attention.

**Memory Match:** A classic card-pairing game.

**AI-Powered Brain Teasers:** Get a new, unique riddle every time.

**AI-Powered Word Scramble:** Unscramble new words generated on the fly.

**Simon Says:** Test your memory with this classic color-sequence game.

**Mindfulness & Vision Exercises:** Guided routines to reduce stress and eye strain, including a Breathing Pacer, Guided Body Scan, and Focus Shift Drills.

**Emotional & Reflective Activities:** Tools like a Gratitude Wheel and a Compliment Generator to foster positive thinking.

## 📖 Spiritual & Memory Corner
**Geeta Gyaan (Wisdom of the Gita):** Receive a "Shlok of the Day" from the Bhagavad Gita, complete with Sanskrit text, translation, and detailed meaning, fetched daily from a live API.

**My People:** A digital memory aid to help you remember important people in your life. Store names, relationships, photos (local upload or URL), key memories, and important details.

## 📊 Tracking & Community
**Historical Trends:** Visualize your emotional trends and wellness activity engagement over time with interactive charts and graphs.

**Daily Scorecard:** Track your performance and completion of wellness exercises each day.

**Community Reflections:** Read and react to anonymous journal entries from other users in a safe and supportive environment.

## 🛠️ Tech Stack
**Frontend:** Streamlit

**Database:** Google Firestore

**Machine Learning & NLP:**

PyTorch

Hugging Face Transformers for sentiment and emotion models.

**Generative AI APIs:**

Groq API (Llama 3)

Google Gemini API

**External APIs:**

Bhagavad Gita API

**Other Key Libraries:** Pandas, Plotly, gTTS.

## 🚀 Getting Started
Follow these steps to set up and run the project locally.

### 1. Prerequisites
Python 3.9+

A Firebase project with Firestore enabled.

API keys for Hugging Face, RapidAPI (for the Gita API), and at least one generative AI provider (Groq or Gemini).

### 2. Clone the Repository
git clone [https://github.com/your-username/green-minds-app.git](https://github.com/your-username/green-minds-app.git)
cd green-minds-app

### 3. Set Up a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

### 4. Install Dependencies
Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

### 5. Configure Your Secrets
The application uses Streamlit's secrets management. Create a file at .streamlit/secrets.toml and add your credentials.

Create the folder:

mkdir .streamlit

Create the secrets.toml file inside it and add the following content:

# .streamlit/secrets.toml

# Hugging Face API Key (must have "Read" permissions)
HUGGING_FACE_API_KEY = "hf_YourKeyHere"

# RapidAPI Key for Bhagavad Gita API
[api_keys]
rapidapi = "YourRapidAPIKeyHere"

# Generative AI Keys (add as many as you have)
GROQ_API_KEY_1 = "gsk_YourGroqKeyHere"
GEMINI_API_KEY_1 = "AIzaSy...YourGeminiKeyHere"

# Firebase Service Account Credentials
[firebase]
type = "service_account"
project_id = "your-firebase-project-id"
private_key_id = "your-firebase-private-key-id"
private_key = """-----BEGIN PRIVATE KEY-----\nYourPrivateKeyHere\n-----END PRIVATE KEY-----\n"""
client_email = "your-firebase-client-email"
client_id = "your-firebase-client-id"
auth_uri = "[https://accounts.google.com/o/oauth2/auth](https://accounts.google.com/o/oauth2/auth)"
token_uri = "[https://oauth2.googleapis.com/token](https://oauth2.googleapis.com/token)"
auth_provider_x509_cert_url = "[https://www.googleapis.com/oauth2/v1/certs](https://www.googleapis.com/oauth2/v1/certs)"
client_x509_cert_url = "your-firebase-cert-url"

### 6. Run the Application
Once your secrets are configured, you can run the app with a single command:

streamlit run App.py

The application should now be running in your web browser!

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
Distributed under the MIT License. See LICENSE for more information.
