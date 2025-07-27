import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from datetime import datetime
import requests
import json
import os

# --- Load Models ---
@st.cache_resource
def load_models():
    goemotions_model_name = "monologg/bert-base-cased-goemotions-original"
    mental_model_name = "mental/mental-bert-base-uncased"

    goemotions_tokenizer = AutoTokenizer.from_pretrained(goemotions_model_name)
    goemotions_model = AutoModelForSequenceClassification.from_pretrained(goemotions_model_name)

    mental_tokenizer = AutoTokenizer.from_pretrained(mental_model_name)
    mental_model = AutoModelForSequenceClassification.from_pretrained(mental_model_name)

    emotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
    emotions = requests.get(emotions_url).text.strip().split('\n')

    return goemotions_tokenizer, goemotions_model, mental_tokenizer, mental_model, emotions

goemotions_tokenizer, goemotions_model, mental_tokenizer, mental_model, emotions = load_models()

# --- Define Functions ---
def detect_emotion(text):
    inputs = goemotions_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = goemotions_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    top_probs, top_ids = torch.topk(probs, 3)
    return [(emotions[i], float(top_probs[0][idx])) for idx, i in enumerate(top_ids[0])]

def detect_mental_state(text):
    inputs = mental_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = mental_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()
    labels = ["non-depressed", "depressed"]
    return labels[label], float(probs[0][label])

# --- UI ---
st.title("🧠 Mood & Mental Health Tracker")
st.markdown("Answer today's reflection questions:")

questions = [
    "How did you feel when you woke up today?",
    "How was your day at work or school?",
    "Did you talk to friends or family today? How did that feel?",
    "What was the most stressful part of your day?",
    "Did anything make you laugh today?",
]

responses = {}
for q in questions:
    responses[q] = st.text_input(q)

if st.button("🔍 Analyze"):
    all_text = " ".join(responses.values()).strip()
    if not all_text:
        st.warning("Please answer the questions before analyzing.")
    else:
        top_emotions = detect_emotion(all_text)
        mental_state, mental_score = detect_mental_state(all_text)

        # Display results
        st.subheader("💬 Top Emotions Detected")
        for emo, score in top_emotions:
            st.write(f"**{emo.capitalize()}**: {score:.2f}")

        st.subheader("🧠 Mental Health")
        st.write(f"**{mental_state.upper()}** (confidence: {mental_score:.2f})")

        # Save Entry
        entry = {
            "timestamp": datetime.now().isoformat(),
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

        df_entry = pd.DataFrame([entry])
        file_path = "mood_tracker_detailed.csv"
        if os.path.exists(file_path):
            df_entry.to_csv(file_path, mode='a', index=False, header=False)
        else:
            df_entry.to_csv(file_path, index=False)

        st.success("✅ Entry saved!")

# --- Show Trends ---
if st.checkbox("📈 Show Trends"):
    file_path = "mood_tracker_detailed.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.to_period("M")

        st.line_chart(df.set_index('timestamp')[['emotion_score_1', 'emotion_score_2', 'emotion_score_3', 'mental_score']])
        st.write("**Weekly Avg Depression Scores:**")
        st.write(df.groupby('week')['mental_score'].mean())

        st.write("**Monthly Avg Depression Scores:**")
        st.write(df.groupby('month')['mental_score'].mean())
    else:
        st.warning("No data yet. Submit a response first.")
