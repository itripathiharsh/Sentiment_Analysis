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

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account

# --- Page Configuration ---
st.set_page_config(
    page_title="Mood & Mental Health Tracker",
    page_icon="🧠",
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
    prompt = f"Translate the following text to {target_language}. Keep the '|||' separators between each item:\n\n{combined_text}"
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
        return "Story generation is currently unavailable as no API keys were found."

    prompt = f"Find a public domain short story that would be comforting and uplifting for someone feeling {mood}. Provide a summary of the story (around 400-500 words) in {lang}. At the end, include the title of the story and the author's name on separate lines, like this:\n\n**Title:** [Story Title]\n**Author:** [Author Name]"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    for i, key in enumerate(gemini_keys):
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=25)
            response.raise_for_status()
            result = response.json()
            if result.get("candidates"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException:
            continue
    
    st.error("All Gemini API keys failed. Please check your keys and API quotas.")
    return "Could not retrieve a story at this time. Please try again later."

def get_activities(mood, lang='en'):
    gemini_keys = get_gemini_keys()
    if not gemini_keys:
        return None

    prompt = f"For someone feeling {mood}, suggest 3 short-term activities for immediate relief, 2 long-term activities for sustained well-being, and 2 psychological techniques or mindset shifts. Provide the response in {lang}. Format the response as a simple markdown list under the headings '### Short-Term Relief', '### Long-Term Well-being', and '### Psychological Techniques:'."
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for i, key in enumerate(gemini_keys):
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
    return """
### Short-Term Relief
- Take a 5-minute break to stretch.
- Listen to a calming song.
- Step outside for fresh air.

### Long-Term Well-being
- Establish a consistent daily routine.
- Incorporate 15-20 minutes of light exercise.

### Psychological Techniques
- Practice the 3-3-3 rule to ground yourself.
- Reframe a negative thought by finding a more balanced perspective.
"""

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

# --- Page Navigation & State Management ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = "default_user" 
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'responses' not in st.session_state:
    st.session_state.responses = {}

def set_page(page_name):
    st.session_state.page = page_name

st.sidebar.title("Navigation")
st.sidebar.button("Home", on_click=set_page, args=('Home',), use_container_width=True)
st.sidebar.button("Journal", on_click=set_page, args=('Journal',), use_container_width=True)
st.sidebar.button("Today's Results", on_click=set_page, args=('Results',), use_container_width=True)
st.sidebar.button("Trends", on_click=set_page, args=('Trends',), use_container_width=True)

st.sidebar.divider()

# --- NEW: Language code mapping for gTTS ---
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja"
}

st.session_state.language = st.sidebar.selectbox(
    "Select Language",
    LANGUAGE_CODES.keys()
)

# --- Page Rendering ---

# HOME PAGE
if st.session_state.page == 'Home':
    st.title("🧠 Daily Reflection")
    st.markdown("Answer today's questions to get an analysis of your emotional state.")
    
    questions_en = get_daily_questions(st.session_state.user_id)
    questions_translated = translate_text(questions_en, st.session_state.language)
    
    for i, q_translated in enumerate(questions_translated):
        q_english = questions_en[i]
        st.session_state.responses[q_english] = st.text_area(
            q_translated, 
            value=st.session_state.responses.get(q_english, ""), 
            key=f"q_{i}"
        )

    if st.button("🔍 Analyze My Day", use_container_width=True):
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
                
                entry = { "timestamp": datetime.now(), "responses": json.dumps(st.session_state.responses), "emotion_1": top_emotions[0][0], "emotion_score_1": top_emotions[0][1], "emotion_2": top_emotions[1][0], "emotion_score_2": top_emotions[1][1], "emotion_3": top_emotions[2][0], "emotion_score_3": top_emotions[2][1], "mental_state": mental_state, "mental_score": mental_score }
                try:
                    user_entries_ref = db.collection("users").document(st.session_state.user_id).collection("mood_entries")
                    user_entries_ref.add(entry)
                    st.success("✅ Your entry has been saved!")
                    st.session_state.page = 'Results'
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save data to Firestore. Error: {e}")

# JOURNAL PAGE
elif st.session_state.page == 'Journal':
    st.title("✍️ My Daily Journal")
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    journal_ref = db.collection("users").document(st.session_state.user_id).collection("journal_entries").document(today_str)
    
    try:
        doc = journal_ref.get()
        current_entry = doc.to_dict().get("text", "") if doc.exists else ""
    except Exception as e:
        st.error(f"Could not load journal entry: {e}")
        current_entry = ""

    journal_text = st.text_area("How was your day? What's on your mind?", value=current_entry, height=400, key="journal_entry")

    if st.button("💾 Save Journal Entry", use_container_width=True):
        if journal_text.strip():
            try:
                journal_ref.set({"text": journal_text, "timestamp": datetime.now()})
                st.success("Your journal entry has been saved!")
            except Exception as e:
                st.error(f"Could not save journal entry. Error: {e}")
        else:
            st.warning("Please write something before saving.")

    st.divider()
    st.subheader("Past Entries")
    try:
        entries_stream = db.collection("users").document(st.session_state.user_id).collection("journal_entries").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(7).stream()
        for entry in entries_stream:
            entry_data = entry.to_dict()
            with st.expander(f"**{entry.id}**"):
                st.write(entry_data.get("text"))
    except Exception as e:
        st.write("Could not load past entries.")


# TODAY'S RESULTS PAGE
elif st.session_state.page == 'Results':
    st.title("✨ Today's Analysis")
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        dominant_emotion = results['top_emotions'][0][0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💬 Top Emotions Detected")
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
            
            # --- FIX: Use the language code mapping ---
            lang_code = LANGUAGE_CODES.get(st.session_state.language, 'en')
            audio_bytes = text_to_audio(story_text, lang=lang_code)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')
        
        st.divider()

        st.subheader("💡 Activity Suggestions")
        with st.spinner("Finding some helpful activities..."):
            activities_markdown = get_activities(dominant_emotion, st.session_state.language)
            if activities_markdown:
                st.markdown(activities_markdown)
            else:
                st.error("Could not retrieve activities at this time.")
    else:
        st.info("Please complete the questionnaire on the 'Home' page to see your results.")

# TRENDS PAGE
elif st.session_state.page == 'Trends':
    st.title("📈 Historical Trends")
    with st.spinner("Loading historical data..."):
        try:
            user_entries_ref = db.collection("users").document(st.session_state.user_id).collection("mood_entries")
            entries_stream = user_entries_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(30).stream()
            entries_list = [doc.to_dict() for doc in entries_stream]
            if not entries_list:
                 st.info("No historical data found. Submit an analysis to get started.")
            else:
                df = pd.DataFrame(entries_list)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df.dropna(subset=['timestamp'], inplace=True)
                
                st.subheader("Most Recent Entry")
                latest = df.iloc[0]
                st.write(f"**Date:** {latest['timestamp'].strftime('%Y-%m-%d')}")
                st.write(f"**Top Emotion:** {latest['emotion_1'].capitalize()} ({latest['emotion_score_1']:.2f})")
                st.write(f"**Mental State:** {latest['mental_state'].upper()} ({latest['mental_score']:.2f})")
                st.divider()

                st.subheader("Emotion & Mental State Scores Over Time")
                st.line_chart(df.set_index('timestamp')[['emotion_score_1', 'emotion_score_2', 'emotion_score_3', 'mental_score']])
                
                if not df.empty:
                    df['week'] = df['timestamp'].dt.isocalendar().week
                    df['month'] = df['timestamp'].dt.to_period("M").astype(str)
                    st.subheader("Weekly Average Mental State Score")
                    st.bar_chart(df.groupby('week')['mental_score'].mean())
                    st.subheader("Monthly Average Mental State Score")
                    st.bar_chart(df.groupby('month')['mental_score'].mean())
        except Exception as e:
            st.error(f"Could not load data from Firestore. Error: {e}")
