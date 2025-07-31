import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from datetime import datetime, timedelta
import requests
import json
import os
import random
import re
from gtts import gTTS
from io import BytesIO
import plotly.graph_objects as go
import hashlib

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account

# --- Page Configuration ---
st.set_page_config(
    page_title="Mood & Mental Health Tracker",
    page_icon="😊",
    layout="wide"
)

# --- Firebase Initialization ---
@st.cache_resource
def init_connection():
    try:
        if not firebase_admin._apps:
            creds_json = {
                "type": st.secrets["firebase"]["type"],
                "project_id": st.secrets["firebase"]["project_id"],
                "private_key_id": st.secrets["firebase"]["private_key_id"],
                "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
                "client_email": st.secrets["firebase"]["client_email"],
                "client_id": st.secrets["firebase"]["client_id"],
                "auth_uri": st.secrets["firebase"]["auth_uri"],
                "token_uri": st.secrets["firebase"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
            }
            creds = credentials.Certificate(creds_json)
            firebase_admin.initialize_app(creds)
        return firestore.client()
    except Exception as e:
        st.error(f"Failed to connect to Firestore: {e}. Make sure your Firebase secrets are set correctly.")
        st.stop()

db = init_connection()

# --- Gamification & Trend Analysis Functions ---
def calculate_streak(user_id, collection_name):
    try:
        entries_ref = db.collection("users").document(user_id).collection(collection_name)
        entries_stream = entries_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        unique_dates = {entry.to_dict()["timestamp"].date() for entry in entries_stream if "timestamp" in entry.to_dict()}
        if not unique_dates:
            return 0
        sorted_dates = sorted(list(unique_dates), reverse=True)
        streak = 0
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        if sorted_dates[0] == today or sorted_dates[0] == yesterday:
            streak = 1
            for i in range(len(sorted_dates) - 1):
                if (sorted_dates[i] - sorted_dates[i+1]).days == 1:
                    streak += 1
                else:
                    break
        return streak
    except Exception:
        return 0

EMOTION_CATEGORIES = {
    'positive': ['admiration', 'amusement', 'approval', 'caring', 'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'],
    'negative': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'],
    'neutral': ['curiosity', 'neutral', 'realization', 'surprise', 'confusion']
}
EMOTION_TO_CATEGORY = {emotion: category for category, emotions in EMOTION_CATEGORIES.items() for emotion in emotions}

# --- Authentication Functions ---
def hash_password(password):
    secret_key = st.secrets.get("SECRET_KEY", "default_secret")
    return hashlib.sha256((password + secret_key).encode()).hexdigest()

def signup_user(username, password):
    users_ref = db.collection("app_users")
    if users_ref.document(username).get().exists:
        return False, "Username already exists."
    hashed_password = hash_password(password)
    users_ref.document(username).set({"password": hashed_password})
    return True, "Signup successful! Please log in."

def login_user(username, password):
    users_ref = db.collection("app_users")
    doc = users_ref.document(username).get()
    if not doc.exists:
        return False
    stored_password = doc.to_dict().get("password")
    hashed_password = hash_password(password)
    if stored_password == hashed_password:
        st.session_state.logged_in = True
        st.session_state.username = username
        return True
    return False

# --- Text-to-Speech Function ---
def text_to_audio(text, lang='en'):
    try:
        clean_text = text.replace("**", "").replace("*", "")
        tts = gTTS(text=clean_text, lang=lang)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Could not generate audio for the story. Error: {e}")
        return None

# --- Gemini API Functions ---
def get_gemini_keys():
    keys = [st.secrets.get(f"GEMINI_API_KEY_{i}") for i in range(1, 5) if st.secrets.get(f"GEMINI_API_KEY_{i}")]
    return keys

def translate_text(text_list, target_language):
    if target_language.lower() == 'english':
        return text_list
    gemini_keys = get_gemini_keys()
    if not gemini_keys:
        return text_list
    combined_text = "|||".join(text_list)
    prompt = f"Translate the following text to {target_language}. Keep the '|||' separators between each item:\n{combined_text}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    for key in gemini_keys:
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=20)
            response.raise_for_status()
            result = response.json()
            if result.get("candidates"):
                translated_combined_text = result["candidates"][0]["content"]["parts"][0]["text"]
                return translated_combined_text.split('|||')
        except requests.exceptions.RequestException:
            continue
    return text_list

def generate_story(mood, lang='en'):
    gemini_keys = get_gemini_keys()
    if not gemini_keys:
        return "Story generation is currently unavailable."
    prompt = f"Find a public domain short story that would be comforting and uplifting for someone feeling {mood}. Provide a summary of the story (around 400-500 words) in {lang}. At the end, include the title of the story and the author's name on separate lines, like this:\n**Title:** [Story Title]\n**Author:** [Author Name]"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    for key in gemini_keys:
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=25)
            response.raise_for_status()
            result = response.json()
            if result.get("candidates"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException:
            continue
    st.error("All Gemini API keys failed.")
    return "Could not retrieve a story at this time."

def get_activities(mood, lang='en'):
    gemini_keys = get_gemini_keys()
    if not gemini_keys:
        return None
    prompt = f"For someone feeling {mood}, suggest 3 short-term activities, 2 long-term activities, and 2 psychological techniques. Provide the response in {lang}. Format the response as a simple markdown list under the headings '### Short-Term Relief', '### Long-Term Well-being', and '### Psychological Techniques:'."
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    for key in gemini_keys:
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=20)
            response.raise_for_status()
            result = response.json()
            if result.get("candidates"):
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                return content
        except Exception:
            continue
    st.warning("Could not generate personalized activities. Showing default suggestions.")
    return "### Short-Term Relief\n- Take a 5-minute break to stretch."

# --- Model Loading & Analysis Functions ---
@st.cache_resource
def load_models():
    goemotions_model_name = "monologg/bert-base-cased-goemotions-original"
    mental_model_name = "mental/mental-bert-base-uncased"
    hf_token = st.secrets.get("HUGGING_FACE_API_KEY")
    if not hf_token:
        st.error("Hugging Face API Key not found in secrets.")
        st.stop()
    goemotions_tokenizer = AutoTokenizer.from_pretrained(goemotions_model_name)
    goemotions_model = AutoModelForSequenceClassification.from_pretrained(goemotions_model_name)
    mental_tokenizer = AutoTokenizer.from_pretrained(mental_model_name, token=hf_token)
    mental_model = AutoModelForSequenceClassification.from_pretrained(mental_model_name, token=hf_token)
    emotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
    emotions = requests.get(emotions_url).text.strip().split('\n')
    return goemotions_tokenizer, goemotions_model, mental_tokenizer, mental_model, emotions

goemotions_tokenizer, goemotions_model, mental_tokenizer, mental_model, emotions = load_models()

def detect_emotion(text):
    inputs = goemotions_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = goemotions_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    top_probs, top_ids = torch.topk(probs, 3)
    return [(emotions[i], float(top_probs[0][idx])) for idx, i in enumerate(top_ids[0])]

def detect_mental_state(text):
    inputs = mental_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = mental_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs).item()
    labels = ["non-depressed", "depressed"]
    return labels[label_id], float(probs[0][label_id])

# --- Question Rotation Logic ---
QUESTION_POOL = [
    "How did you feel when you woke up today?", "How was your day at work or school?",
    "Did you talk to friends or family today? How did that feel?", "What was the most stressful part of your day?",
    "How are you feeling right now?", "Did anything make you laugh today?",
    "What was something you were proud of today?", "How would you describe your energy level today?",
    "Did you feel connected to others today?", "Was there a moment of peace or joy today?",
    "Did you feel lonely at any point?", "What frustrated you the most today?",
    "Did you feel anxious or nervous? What triggered it?", "Was today better or worse than yesterday? Why?",
    "Did you do something just for yourself today?", "What's one thing you're looking forward to tomorrow?",
    "Describe a small victory you had today.", "What's something that has been on your mind lately?"
]

@st.cache_data(ttl=3600)
def get_daily_questions(user_id):
    today = datetime.now().date()
    log_ref = db.collection("users").document(user_id).collection("question_logs").document("log")
    try:
        log_doc = log_ref.get()
        if log_doc.exists:
            log_data = log_doc.to_dict().get("history", [])
            four_days_ago = today - timedelta(days=4)
            recent_questions = {q for entry in log_data if datetime.strptime(entry["date"], "%Y-%m-%d").date() >= four_days_ago for q in entry["questions"]}
        else:
            log_data, recent_questions = [], set()
    except Exception:
        log_data, recent_questions = [], set()

    available_questions = [q for q in QUESTION_POOL if q not in recent_questions]
    if len(available_questions) < 5:
        available_questions = QUESTION_POOL
    selected_questions = random.sample(available_questions, 5)

    new_log_entry = {"date": today.strftime("%Y-%m-%d"), "questions": selected_questions}
    log_data.append(new_log_entry)
    log_data = log_data[-10:]
    log_ref.set({"history": log_data})
    return selected_questions

# --- State Management ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'journal_text' not in st.session_state:
    st.session_state.journal_text = ""

def set_page(page_name):
    st.session_state.page = page_name

# --- MAIN APP ---
if not st.session_state.logged_in:
    st.title("Welcome to the Mood & Mental Health Tracker")
    choice = st.selectbox("Login or Signup?", ["Login", "Signup"])
    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(username, password):
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    else:
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        if st.button("Signup"):
            success, message = signup_user(username, password)
            if success:
                st.success(message)
            else:
                st.error(message)
else:
    # --- AUTHENTICATED APP ---
    st.sidebar.title(f"Welcome, {st.session_state.username}!")
    st.sidebar.button("Home", on_click=set_page, args=('Home',), use_container_width=True)
    st.sidebar.button("Journal", on_click=set_page, args=('Journal',), use_container_width=True)
    st.sidebar.button("Today's Results", on_click=set_page, args=('Results',), use_container_width=True)
    st.sidebar.button("Trends", on_click=set_page, args=('Trends',), use_container_width=True)
    st.sidebar.divider()

    LANGUAGE_CODES = {"English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de", "Japanese": "ja"}
    st.session_state.language = st.sidebar.selectbox("Select Language", list(LANGUAGE_CODES.keys()))

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    # --- Page Rendering ---
    if st.session_state.page == 'Home':
        st.title("😊 Daily Reflection")
        reflection_streak = calculate_streak(st.session_state.username, "mood_entries")
        st.metric(label="Reflection Streak", value=f"{reflection_streak} Days 🔥")

        st.markdown("Answer today's questions to get an analysis of your emotional state.")
        questions_en = get_daily_questions(st.session_state.username)
        questions_translated = translate_text(questions_en, st.session_state.language)

        for i, q_translated in enumerate(questions_translated):
            q_english = questions_en[i]
            st.session_state.responses[q_english] = st.text_area(
                q_translated,
                value=st.session_state.responses.get(q_english, ""),
                key=f"q_{i}"
            )

        if st.button("✅ Analyze My Day", use_container_width=True):
            answered_responses = [r for r in st.session_state.responses.values() if r and r.strip()]
            if not answered_responses:
                st.warning("Please answer at least one question before analyzing.")
            else:
                all_text = " ".join(answered_responses)
                with st.spinner("Analyzing your responses..."):
                    top_emotions = detect_emotion(all_text)
                    mental_state, mental_score = detect_mental_state(all_text)
                    st.session_state.analysis_results = {
                        "top_emotions": top_emotions,
                        "mental_state": mental_state,
                        "mental_score": mental_score
                    }
                    entry = {
                        "timestamp": datetime.now(),
                        "responses": json.dumps(st.session_state.responses),
                        "emotion_1": top_emotions[0][0],
                        "emotion_score_1": top_emotions[0][1],
                        "emotion_2": top_emotions[1][0],
                        "emotion_score_2": top_emotions[1][1],
                        "emotion_3": top_emotions[2][0],
                        "emotion_score_3": top_emotions[2][1],
                        "mental_state": mental_state,
                        "mental_score": mental_score
                    }
                    try:
                        user_entries_ref = db.collection("users").document(st.session_state.username).collection("mood_entries")
                        user_entries_ref.add(entry)
                        st.success("✅ Your entry has been saved!")
                        st.session_state.page = 'Results'
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not save data to Firestore. Error: {e}")

    elif st.session_state.page == 'Journal':
        st.title("✍️ My Daily Journal")
        journal_streak = calculate_streak(st.session_state.username, "journal_entries")
        st.metric(label="Journaling Streak", value=f"{journal_streak} Days 🔥")
        
        st.subheader("Write or Edit Your Entry")

        today_str = datetime.now().strftime("%Y-%m-%d")
        journal_ref = db.collection("users").document(st.session_state.username).collection("journal_entries").document(today_str)

        try:
            doc = journal_ref.get()
            if not st.session_state.journal_text and doc.exists:
                st.session_state.journal_text = doc.to_dict().get("text", "")
        except Exception as e:
            st.error(f"Could not load journal entry: {e}")

        st.session_state.journal_text = st.text_area(
            "How was your day? What's on your mind?",
            value=st.session_state.journal_text,
            height=300,
            key="journal_text_area"
        )

        if st.button("💾 Save Journal Entry", use_container_width=True):
            if st.session_state.journal_text.strip():
                try:
                    journal_ref.set({
                        "text": st.session_state.journal_text,
                        "timestamp": datetime.now()
                    })
                    st.success("Your journal entry has been saved!")
                except Exception as e:
                    st.error(f"Could not save journal entry. Error: {e}")
            else:
                st.warning("Please write something before saving.")

        st.divider()
        st.subheader("Past Entries")
        try:
            entries_stream = db.collection("users").document(st.session_state.username).collection("journal_entries").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(7).stream()
            for entry in entries_stream:
                entry_data = entry.to_dict()
                with st.expander(f"**{entry.id}**"):
                    st.write(entry_data.get("text"))
        except Exception as e:
            st.write("Could not load past entries.")

    elif st.session_state.page == 'Results':
        st.title("📊 Today's Analysis")
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            dominant_emotion = results['top_emotions'][0][0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("😊 Top Emotions Detected")
                for emo, score in results['top_emotions']:
                    st.write(f"**{emo.capitalize()}**: {score:.2f}")
            with col2:
                st.subheader("🧠 Mental State Analysis")
                st.write(f"Detected State: **{results['mental_state'].upper()}**")
                st.write(f"Confidence: **{results['mental_score']:.2f}**")

            st.divider()
            st.subheader("📖 A Story For You")
            with st.spinner("Finding and narrating a story for you..."):
                story_text = generate_story(dominant_emotion, st.session_state.language)
                st.markdown(f"<div style='border-left: 5px solid #ccc; padding-left: 20px; font-style: italic;'>{story_text}</div>", unsafe_allow_html=True)
                lang_code = LANGUAGE_CODES.get(st.session_state.language, 'en')
                audio_bytes = text_to_audio(story_text, lang=lang_code)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')

            st.divider()
            st.subheader("🎯 Activity Suggestions")
            with st.spinner("Finding some helpful activities..."):
                activities_markdown = get_activities(dominant_emotion, st.session_state.language)
                if activities_markdown:
                    st.markdown(activities_markdown)
                else:
                    st.error("Could not retrieve activities at this time.")
        else:
            st.info("Please complete the questionnaire on the 'Home' page to see your results.")

    elif st.session_state.page == 'Trends':
        st.title("📈 Historical Trends")
        with st.spinner("Loading historical data..."):
            try:
                user_entries_ref = db.collection("users").document(st.session_state.username).collection("mood_entries")
                entries_stream = user_entries_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(90).stream()
                entries_list = [doc.to_dict() for doc in entries_stream]

                if not entries_list:
                    st.info("No historical data found. Submit an analysis to get started.")
                else:
                    df = pd.DataFrame(entries_list)
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date

                    st.subheader("Dominant Emotions (Last 30 Days)")
                    thirty_days_ago = datetime.now().date() - timedelta(days=30)
                    df_30_days = df[df['timestamp'] >= thirty_days_ago]
                    if not df_30_days.empty:
                        emotion_counts = df_30_days['emotion_1'].value_counts()
                        fig_pie = go.Figure(data=[go.Pie(labels=emotion_counts.index, values=emotion_counts.values, hole=.3)])
                        fig_pie.update_layout(title_text='Emotion Breakdown')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.write("Not enough data in the last 30 days for a pie chart.")

                    st.divider()
                    st.subheader("Positive vs. Negative Emotion Trends (Last 30 Days)")
                    df_30_days['emotion_category'] = df_30_days['emotion_1'].map(EMOTION_TO_CATEGORY)
                    category_counts = df_30_days.groupby(['timestamp', 'emotion_category']).size().unstack(fill_value=0)
                    fig_area = go.Figure()
                    for category in ['positive', 'negative', 'neutral']:
                        if category in category_counts.columns:
                            fig_area.add_trace(go.Scatter(
                                x=category_counts.index,
                                y=category_counts[category],
                                mode='lines',
                                stackgroup='one',
                                name=category.capitalize()
                            ))
                    fig_area.update_layout(title_text='Daily Emotion Category Count')
                    st.plotly_chart(fig_area, use_container_width=True)

                    st.divider()
                    st.subheader("Emotion & Mental State Scores Over Time")
                    st.line_chart(df.set_index('timestamp')[['emotion_score_1', 'mental_score']])
            except Exception as e:
                st.error(f"Could not load data from Firestore. Error: {e}")
