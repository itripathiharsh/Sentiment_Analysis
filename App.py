import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from datetime import datetime
import requests
import json
import os


@st.cache_resource
def load_models():
    
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
    if label_id < len(labels):
        label = labels[label_id]
        score = float(probs[0][label_id])
    else:
        label = "unknown"
        score = 0.0
    return label, score

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
            
            try:
                if os.path.exists(file_path):
                    df_entry.to_csv(file_path, mode='a', index=False, header=False)
                else:
                    df_entry.to_csv(file_path, index=False)
                
                st.success("✅ Your entry has been saved!")
            except Exception as e:
                st.error(f"Could not save the data. Error: {e}")



if st.checkbox("📈 Show Historical Trends"):
    file_path = "mood_tracker_detailed.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True) # Remove rows where timestamp could not be parsed
            
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
            st.error(f"Could not load or process the data file. Error: {e}")
    else:
        st.info("No historical data found. Submit an analysis to get started.")
