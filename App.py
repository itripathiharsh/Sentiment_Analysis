import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from datetime import datetime, timedelta
import requests
import json
import random

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

# Gemini AI import
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Mood & Mental Health Tracker",
    page_icon="🧠",
    layout="wide"
)

# --- Firebase Initialization ---
@st.cache_resource
def init_connection():
    """Initializes the connection to the Firebase Firestore database."""
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

# --- Gemini API Functions ---

def generate_story(mood):
    """Generates a comforting story summary using the Gemini API."""
    prompt = f"""
You are a helpful literary assistant. Find a public domain short story that would be comforting and uplifting for someone feeling '{mood}'.

Provide a summary of the story (around 400-500 words).

At the very end, include the title and the author's name on separate lines in the following format:
**Title:** [Story Title]
**Author:** [Author Name]
"""
    for i in range(1, 5):  # Try up to 4 API keys
        key = st.secrets.get(f"GEMINI_API_KEY_{i}")
        if not key:
            continue
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content(prompt)
            if result.text:
                return result.text
            else: # Handle cases where API returns an empty response (e.g., safety blocks)
                continue
        except Exception as e:
            st.warning(f"Story generation failed with key {i}: {e}")
            continue

    st.error("All Gemini API keys failed. Could not retrieve a story.")
    return "Could not retrieve a story at this time. The service may be temporarily unavailable."

def get_activities(mood):
    """Generates activity suggestions in JSON format using the Gemini API."""
    prompt = f"""
You are a helpful mental wellness assistant. Based on the user's current mood '{mood}', generate relaxing and motivating activities in the following JSON format:

{{
  "short_term": [
    {{"activity": "...", "benefit": "..."}},
    ...
  ],
  "long_term": [
    {{"activity": "...", "benefit": "..."}},
    ...
  ],
  "psychological": [
    {{"technique": "...", "benefit": "..."}},
    ...
  ]
}}

Keep each activity concise, helpful, and applicable to someone in that mood.
"""
    for i in range(1, 5):  # Try up to 4 API keys
        key = st.secrets.get(f"GEMINI_API_KEY_{i}")
        if not key:
            continue
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content(prompt)
            # Clean the response to ensure it's valid JSON
            response_text = result.text.strip().replace("```json", "").replace("```", "")
            activities_json = json.loads(response_text)
            return activities_json
        except Exception as e:
            st.warning(f"Activity generation failed with key {i}: {e}")
            continue

    # Fallback default if all keys fail
    st.warning("Could not generate personalized activities. Showing default suggestions.")
    return {
        "short_term": [
            {"activity": "Drink water and stretch", "benefit": "Immediate physical refreshment"},
            {"activity": "Take 5 deep breaths", "benefit": "Calms your nervous system"}
        ],
        "long_term": [
            {"activity": "Start a gratitude journal", "benefit": "Improves long-term mental wellbeing"},
            {"activity": "Create a daily sleep routine", "benefit": "Enhances rest and reduces anxiety"}
        ],
        "psychological": [
            {"technique": "Cognitive reframing", "benefit": "Helps shift negative thoughts"},
            {"technique": "Progressive muscle relaxation", "benefit": "Reduces tension in the body"}
        ]
    }

# --- Model Loading & Analysis Functions ---
@st.cache_resource
def load_models():
    """Loads Hugging Face models and tokenizers for analysis."""
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
    """Detects top 3 emotions from text using the GoEmotions model."""
    inputs = goemotions_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = goemotions_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    top_probs, top_ids = torch.topk(probs, 3)
    return [(emotions[i], float(top_probs[0][idx])) for idx, i in enumerate(top_ids[0])]

def detect_mental_state(text):
    """Detects mental state (depressed/non-depressed) using the Mental-BERT model."""
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
    """Provides a new set of 5 questions daily, avoiding recent questions."""
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
    if len(available_questions) < 5: # Reset if we run out of fresh questions
        available_questions = QUESTION_POOL
    selected_questions = random.sample(available_questions, 5)

    new_log_entry = {"date": today.strftime("%Y-%m-%d"), "questions": selected_questions}
    log_data.append(new_log_entry)
    log_data = log_data[-10:] # Keep the log from getting too large
    log_ref.set({"history": log_data})
    return selected_questions

# --- Page Navigation & State Management ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = "default_user" 

def set_page(page_name):
    st.session_state.page = page_name

st.sidebar.title("Navigation")
st.sidebar.button("Home", on_click=set_page, args=('Home',), use_container_width=True)
st.sidebar.button("Today's Results", on_click=set_page, args=('Results',), use_container_width=True)
st.sidebar.button("Trends", on_click=set_page, args=('Trends',), use_container_width=True)

# --- Page Rendering ---

# HOME PAGE
if st.session_state.page == 'Home':
    st.title("🧠 Daily Reflection")
    st.markdown("Answer today's questions to get an analysis of your emotional state.")
    
    questions = get_daily_questions(st.session_state.user_id)
    responses = {}
    for i, q in enumerate(questions):
        responses[q] = st.text_area(q, key=f"q_{i}")

    if st.button("🔍 Analyze My Day", use_container_width=True, type="primary"):
        answered_responses = [r for r in responses.values() if r.strip()]
        if not answered_responses:
            st.warning("Please answer at least one question before analyzing.")
        else:
            all_text = " ".join(answered_responses)
            with st.spinner("Analyzing your responses... This may take a moment."):
                top_emotions = detect_emotion(all_text)
                mental_state, mental_score = detect_mental_state(all_text)
                
                st.session_state.analysis_results = {
                    "top_emotions": top_emotions,
                    "mental_state": mental_state,
                    "mental_score": mental_score
                }
                
                entry = {
                    "timestamp": datetime.now(), "responses": json.dumps(responses),
                    "emotion_1": top_emotions[0][0], "emotion_score_1": top_emotions[0][1],
                    "emotion_2": top_emotions[1][0], "emotion_score_2": top_emotions[1][1],
                    "emotion_3": top_emotions[2][0], "emotion_score_3": top_emotions[2][1],
                    "mental_state": mental_state, "mental_score": mental_score
                }
                try:
                    user_entries_ref = db.collection("users").document(st.session_state.user_id).collection("mood_entries")
                    user_entries_ref.add(entry)
                    st.success("✅ Your entry has been saved!")
                    set_page('Results')
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save data to Firestore. Error: {e}")

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
        
        st.subheader(f"📖 A Story For When You're Feeling {dominant_emotion.capitalize()}")
        with st.spinner("Finding a story for you..."):
            story = generate_story(dominant_emotion)
            st.markdown(f"<div style='border-left: 5px solid #ccc; padding-left: 20px; font-style: italic;'>{story}</div>", unsafe_allow_html=True)
        
        st.divider()

        st.subheader("💡 Activity Suggestions")
        with st.spinner("Finding some helpful activities..."):
            activities = get_activities(dominant_emotion)
            if activities:
                st.markdown("**For Immediate Relief:**")
                for act in activities.get("short_term", []):
                    st.markdown(f"- **{act['activity']}**: {act['benefit']}")
                
                st.markdown("\n**For Long-Term Well-being:**")
                for act in activities.get("long_term", []):
                    st.markdown(f"- **{act['activity']}**: {act['benefit']}")
                
                st.markdown("\n**Psychological Techniques:**")
                for act in activities.get("psychological", []):
                    st.markdown(f"- **{act['technique']}**: {act['benefit']}")
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
                
                df['week'] = df['timestamp'].dt.isocalendar().week.astype(str)
                df['month'] = df['timestamp'].dt.to_period("M").astype(str)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Weekly Average Mental Score")
                    st.bar_chart(df.groupby('week')['mental_score'].mean())
                with col2:
                    st.subheader("Monthly Average Mental Score")
                    st.bar_chart(df.groupby('month')['mental_score'].mean())
        except Exception as e:
            st.error(f"Could not load data from Firestore. Error: {e}")