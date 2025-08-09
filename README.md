# Green Minds 🌱
Your AI-Powered Mental Wellness Journal & Toolkit
A personal and community space for mind, emotions, and spirit — powered by cutting-edge AI.

### 🎥 Demo Video: [Watch here](https://vimeo.com/1107494438?share=copy#t=0)
### 🌐 Live App: [thesentimentanalysisapp.streamlit.app](https://vimeo.com/1107494438?share=copy#t=0)

## 💡 Why Green Minds?
The modern world takes a toll on our mental and spiritual health. People want:

- A safe space to reflect.

- Tools to understand and manage emotions.

- Simple daily habits that actually help.

- **Green Minds** is your AI-powered companion for journaling, mood tracking, mental exercises, and daily wisdom — all in one place.

## ✨ Key Features
### 🧠 Core AI & Journaling
- **Daily Emotional Analysis:** AI detects your mood from journal answers using GoEmotions + fine-tuned BERTweet sentiment model.

- **Secure Private Journal:** Entries stored in Firestore — yours, always private unless you choose to share.

- **Optional Anonymous Sharing:** Join a supportive, anonymous community by sharing your reflections.

- **AI-Generated Stories & Activities:** Personalized wellness stories & activities via Groq & Gemini.

### 🧘 Wellness Toolkit
Cognitive Games:

- Number Tap Challenge

- Memory Match

- AI Brain Teasers & Word Scramble

- Simon Says

Mindfulness Exercises:
- Breathing Pacer 
- Guided Body Scan 
- Focus Shift Drills

Positive Thinking Tools:
- Gratitude Wheel
- Compliment Generator

### 📖 Spiritual & Memory Corner
- **Geeta Gyaan:** Daily Bhagavad Gita shlok with translation & meaning (via live API).

- **My People:** A digital memory aid for important people & memories.

### 📊 Tracking & Community
- **Historical Trends:** Interactive charts of mood & wellness activity engagement.

- **Daily Scorecard:** Track completed exercises & progress.

- **Community Reflections:** Read & react to anonymous entries.

## 🚀 Try It Now
### Clone the Repo

- git clone https://github.com/your-username/green-minds-app.git

- cd green-minds-app

### Set Up Environment

- python -m venv venv

- source venv/bin/activate  # macOS/Linux

- venv\Scripts\activate     # Windows

- pip install -r requirements.txt

### Configure Secrets (API keys & Firebase) in .streamlit/secrets.toml.

### Run App

- streamlit run App.py

## 🛠 Tech Stack

- Frontend: Streamlit

- Database: Google Firestore

- ML/NLP: PyTorch, Hugging Face Transformers (GoEmotions, fine-tuned BERTweet)

- Generative AI: Groq API (Llama 3), Google Gemini API

- External APIs: Bhagavad Gita API

-Libraries: Pandas, Plotly, gTTS

## 🤝 Contributing
We welcome ideas, bug reports, and feature requests!

Fork → Create Branch → Commit → Push → PR.

## 📜 License
MIT License — free to use, modify, and share.

