import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from datetime import datetime
import requests
import json
import os
# Firebase imports
from firebase_admin import credentials, firestore, initialize_app, App as FirebaseApp
from google.oauth2 import service_account

# --- Firebase Initialization ---
# This function initializes the connection to your Firestore database.
# It uses secrets stored in Streamlit to keep your credentials safe.
@st.cache_resource
def init_connection():
    """
    Initialize connection to Firebase Firestore.
    Uses st.secrets for credentials.
    Returns the Firestore database client.
    """
    try:
        # Check if the app is already initialized
        if not "firebase_app" in st.session_state:
            # Get credentials from Streamlit secrets
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
            st.session_state.firebase_app = initialize_app(creds)
        
        db = firestore.client(app=st.session_state.firebase_app)
        return db
    except Exception as e:
        st.error(f"Failed to connect to Firestore: {e}. Make sure your Firebase secrets are set correctly in Streamlit Cloud.")
        st.stop()

# Initialize connection and get the database client
db = init_connection()

# --- Load Models ---
@st.cache_resource
def load_models():
    """
    Loads the tokenizers and models from Hugging Face.
    It uses the HUGGING_FACE_API_KEY from Streamlit's secrets to authenticate.
    """
    goemotions_model_name = "monologg/bert-base-cased-goemotions-original"
    mental_model_name = "mental/mental-bert-base-uncased"

    hf_token = st.secrets.get("HUGGING_FACE_API_KEY")
    if not hf_token:
        st.error("Hugging Face API Key not found in secrets. Please add it to your Streamlit Cloud app settings.")
        st.stop()

    goemotions_tokenizer = AutoTokenizer.from_pretrained(goemotions_model_name)
    goemotions_model = AutoModelForSequenceClassification.from_pretrained(goemotions_model_name)

    mental_tokenizer = AutoTokenizer.from_pretrained(mental_model_name, token=hf_token)
    mental_model = AutoModelForSequenceClassification.from_pretrained(mental_model_name, token=hf_token)

    emotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
    emotions_response = requests.get(emotions_url)
    emotions = emotions_response.text.strip().split('\n')

    return goemotions_tokenizer, goemotions_model, mental_tokenizer, mental_model, emotions

goemotions_tokenizer, goemotions_model, mental_tokenizer, mental_model, emotions = load_models()

# --- Define Functions ---
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

# --- UI ---
st.title("🧠 Mood & Mental Health Tracker")
st.markdown("Answer today's reflection questions to get an analysis of your emotional state.")

questions = [
    "How did you feel when you woke up today?",
    "How was your day at work or school?",
    "Did you talk to friends or family today? How did that feel?",
    "What was the most stressful part of your day?",
    "Did anything make you laugh today?",
]

# Use a unique user ID, for example, based on session information
# For simplicity, we'll use a generic ID, but in a real multi-user app, this should be unique per user.
if 'user_id' not in st.session_state:
    st.session_state.user_id = "default_user" 

responses = {}
for q in questions:
    responses[q] = st.text_area(q, key=q)

if st.button("🔍 Analyze My Day"):
    answered_responses = [r for r in responses.values() if r.strip()]
    
    if not answered_responses:
        st.warning("Please answer at least one question before analyzing.")
    else:
        all_text = " ".join(answered_responses)
        
        with st.spinner("Analyzing your responses..."):
            top_emotions = detect_emotion(all_text)
            mental_state, mental_score = detect_mental_state(all_text)

            st.subheader("💬 Top Emotions Detected")
            for emo, score in top_emotions:
                st.write(f"**{emo.capitalize()}**: {score:.2f}")

            st.subheader("🧠 Mental State Analysis")
            st.write(f"Detected State: **{mental_state.upper()}** (Confidence: {mental_score:.2f})")

            # --- NEW: Save Entry to Firestore ---
            entry = {
                "timestamp": datetime.now(), # Firestore handles datetime objects
                "responses": json.dumps(responses),
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
                # Create a new document in the 'mood_entries' collection for the user
                user_entries_ref = db.collection("users").document(st.session_state.user_id).collection("mood_entries")
                user_entries_ref.add(entry)
                st.success("✅ Your entry has been saved to the database!")
            except Exception as e:
                st.error(f"Could not save the data to Firestore. Error: {e}")

# --- Show Trends ---
if st.checkbox("📈 Show Historical Trends"):
    with st.spinner("Loading historical data..."):
        try:
            # --- NEW: Read data from Firestore ---
            user_entries_ref = db.collection("users").document(st.session_state.user_id).collection("mood_entries")
            entries_stream = user_entries_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
            
            entries_list = [doc.to_dict() for doc in entries_stream]

            if not entries_list:
                 st.info("No historical data found. Submit an analysis to get started.")
            else:
                df = pd.DataFrame(entries_list)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df.dropna(subset=['timestamp'], inplace=True)
                
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
