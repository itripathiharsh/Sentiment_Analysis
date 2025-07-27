# pip install transformers torch pandas matplotlib requests huggingface_hub datasets seaborn

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import random
import json
import os
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
token = os.getenv("HF_TOKEN")



# --- Load Models & Labels ---
goemotions_model_name = "monologg/bert-base-cased-goemotions-original"
goemotions_tokenizer = AutoTokenizer.from_pretrained(goemotions_model_name)
goemotions_model = AutoModelForSequenceClassification.from_pretrained(goemotions_model_name)

emotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
response = requests.get(emotions_url)
if response.status_code == 200:
    emotions = response.text.strip().split('\n')
else:
    raise Exception(f"Failed to download emotions file: status code {response.status_code}")

mental_model_name = "mental/mental-bert-base-uncased"
mental_tokenizer = AutoTokenizer.from_pretrained(mental_model_name)
mental_model = AutoModelForSequenceClassification.from_pretrained(mental_model_name)
mental_labels = ["non-depressed", "depressed"]

# --- Questions Pool ---
question_pool = [
    "How did you feel when you woke up today?",
    "How was your day at work or school?",
    "Did you talk to friends or family today? How did that feel?",
    "What was the most stressful part of your day?",
    "How are you feeling right now?",
    "Did anything make you laugh today?",
    "What was something you were proud of today?",
    "How would you describe your energy level today?",
    "Did you feel connected to others today?",
    "Was there a moment of peace or joy today?",
    "Did you feel lonely at any point?",
    "What frustrated you the most today?",
    "Did you feel anxious or nervous? What triggered it?",
    "Was today better or worse than yesterday? Why?",
    "Did you do something just for yourself today?"
]

# --- Question Log Setup ---
log_file = "question_log.json"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        question_log = json.load(f)
else:
    question_log = {}

today_str = datetime.now().strftime("%Y-%m-%d")

cutoff = datetime.now() - timedelta(days=14)
question_log = {
    date: qs for date, qs in question_log.items()
    if datetime.strptime(date, "%Y-%m-%d") >= cutoff
}

recent_questions = set(q for qs in question_log.values() for q in qs)
available_questions = [q for q in question_pool if q not in recent_questions]

if len(available_questions) < 5:
    print("⚠️ Not enough new questions left; allowing some repeats.")
    available_questions = question_pool.copy()

selected_questions = random.sample(available_questions, 5)
question_log[today_str] = selected_questions
with open(log_file, "w") as f:
    json.dump(question_log, f, indent=2)

# --- Emotion Detection ---
def detect_emotion(text):
    inputs = goemotions_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = goemotions_model(**inputs)
    logits = outputs.logits
    probs = softmax(logits, dim=1)
    k = min(3, logits.shape[1])
    top_probs, top_ids = torch.topk(probs, k)

    top_emotions = []
    for idx, i in enumerate(top_ids[0]):
        if i >= len(emotions):
            continue
        top_emotions.append((emotions[i], float(top_probs[0][idx])))

    return top_emotions

def detect_all_emotions(text):
    inputs = goemotions_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = goemotions_model(**inputs)
    logits = outputs.logits
    probs = softmax(logits, dim=1)[0].detach().cpu().numpy()
    return dict(zip(emotions, probs))

# --- Mental State Detection ---
def detect_mental_state(text):
    inputs = mental_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = mental_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs, dim=1).item()
    return mental_labels[label_id], float(probs[0][label_id])

# --- Collect Responses ---
print("\n🧠 Daily Mood & Mental Health Tracker 🧠\n")
answers = []
all_text = ""

# Also collect emotion per question for tracking
emotion_per_question = []

for q in selected_questions:
    answer = input(f"{q}\n> ").strip()
    answers.append(answer)
    all_text += " " + answer

    # Detect full emotion vector per answer for question-level tracking
    emotions_dict = detect_all_emotions(answer)
    emotion_per_question.append(emotions_dict)

# Detect emotions & mental state for whole day's text
top_emotions = detect_emotion(all_text)
mental_state, mental_score = detect_mental_state(all_text)

print("\n💬 Detected Emotions (top 3):")
for emo, score in top_emotions:
    print(f" - {emo}: {score:.2f}")

print(f"🧠 Mental State: {mental_state} ({mental_score:.2f})")

# --- Save detailed responses as JSON file ---
responses_dict = {q: a for q, a in zip(selected_questions, answers)}
with open(f"responses_{today_str}.json", "w") as f:
    json.dump(responses_dict, f, indent=2)

# --- Save daily summary to CSV ---
entry = {
    "timestamp": datetime.now(),
    "responses": json.dumps(responses_dict),
    "emotion_1": top_emotions[0][0] if len(top_emotions) > 0 else None,
    "emotion_score_1": top_emotions[0][1] if len(top_emotions) > 0 else None,
    "emotion_2": top_emotions[1][0] if len(top_emotions) > 1 else None,
    "emotion_score_2": top_emotions[1][1] if len(top_emotions) > 1 else None,
    "emotion_3": top_emotions[2][0] if len(top_emotions) > 2 else None,
    "emotion_score_3": top_emotions[2][1] if len(top_emotions) > 2 else None,
    "mental_state": mental_state,
    "mental_score": mental_score
}

df_entry = pd.DataFrame([entry])
df_entry.to_csv("mood_tracker_detailed.csv", mode='a', index=False, header=not os.path.exists("mood_tracker_detailed.csv"))

# --- Save all emotions intensities per day (average over questions) ---
avg_emotions = {emo: 0.0 for emo in emotions}
for emo_dict in emotion_per_question:
    for emo in emotions:
        avg_emotions[emo] += emo_dict.get(emo, 0.0)
for emo in avg_emotions:
    avg_emotions[emo] /= len(emotion_per_question)

avg_emotions['timestamp'] = datetime.now()
df_all_emotions = pd.DataFrame([avg_emotions])
df_all_emotions.to_csv("daily_emotions.csv", mode='a', index=False, header=not os.path.exists("daily_emotions.csv"))

# --- Save individual question emotions for future per-question trend analysis ---
per_question_data = {"timestamp": datetime.now()}
for i, q in enumerate(selected_questions):
    for emo, val in emotion_per_question[i].items():
        # key: question1_joy, question2_fear, etc.
        key = f"q{i+1}_{emo}"
        per_question_data[key] = val
df_per_question = pd.DataFrame([per_question_data])
df_per_question.to_csv("question_emotions_over_time.csv", mode='a', index=False, header=not os.path.exists("question_emotions_over_time.csv"))

# --- Load data for trend analysis and visualization ---
df = pd.read_csv("mood_tracker_detailed.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Trend Analysis: check if depression score is rising over last 4 days ---
df = df.sort_values('timestamp')
df['mental_score'] = pd.to_numeric(df['mental_score'], errors='coerce')
df['rolling_avg_3d'] = df['mental_score'].rolling(window=3).mean()

if len(df) >= 4:
    last_scores = df['rolling_avg_3d'].dropna().tail(4).values
    if len(last_scores) == 4 and all(x < y for x, y in zip(last_scores, last_scores[1:])):
        print("\n⚠️ Warning: Your depression score has been rising steadily over the last few days.")

# Weekly and monthly averages
df['week'] = df['timestamp'].dt.isocalendar().week
weekly_avg = df.groupby('week')['mental_score'].mean()
print("\nWeekly average depression scores:")
print(weekly_avg)

df['month'] = df['timestamp'].dt.to_period('M')
monthly_avg = df.groupby('month')['mental_score'].mean()
print("\nMonthly average depression scores:")
print(monthly_avg)

# --- Visualization ---

# Plot mood & mental health scores over time (top 3 emotions + mental_score)
plt.figure(figsize=(14, 7))

if 'emotion_score_1' in df.columns and df['emotion_score_1'].notna().any():
    plt.plot(df['timestamp'], df['emotion_score_1'], label=f'Emotion 1 Score ({df["emotion_1"].iloc[-1]})', marker='o')
if 'emotion_score_2' in df.columns and df['emotion_score_2'].notna().any():
    plt.plot(df['timestamp'], df['emotion_score_2'], label=f'Emotion 2 Score ({df["emotion_2"].iloc[-1]})', marker='o')
if 'emotion_score_3' in df.columns and df['emotion_score_3'].notna().any():
    plt.plot(df['timestamp'], df['emotion_score_3'], label=f'Emotion 3 Score ({df["emotion_3"].iloc[-1]})', marker='o')

if 'mental_score' in df.columns and df['mental_score'].notna().any():
    plt.plot(df['timestamp'], df['mental_score'], label='Mental Health Score (Depression Confidence)', marker='x', linestyle='--', color='black')

plt.xlabel("Date")
plt.ylabel("Score (0 to 1)")
plt.title("Mood and Mental Health Scores Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Heatmap of all emotions over time ---
df_emotions = pd.read_csv("daily_emotions.csv")
df_emotions['timestamp'] = pd.to_datetime(df_emotions['timestamp'])
df_emotions = df_emotions.set_index('timestamp')

plt.figure(figsize=(15, 8))
sns.heatmap(df_emotions.T, cmap="coolwarm", cbar_kws={'label': 'Emotion Intensity'})
plt.title("Emotion Intensities Over Time (Daily Average)")
plt.xlabel("Date")
plt.ylabel("Emotions")
plt.tight_layout()
plt.show()
