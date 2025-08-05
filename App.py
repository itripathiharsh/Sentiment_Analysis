import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from datetime import datetime, timedelta, date
import requests
import json
import os
import random
import re
from gtts import gTTS
from io import BytesIO
import plotly.graph_objects as go
import hashlib
import time
import base64
# Firebase
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
# Game helper
from streamlit_image_select import image_select

# --- Suppress TensorFlow Warnings ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# --- Page Config ---
st.set_page_config(page_title="Green Minds 🌱", page_icon="🌱", layout="wide")

# --- Firebase Init ---
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
        st.error(f"Failed to connect to Firestore: {e}")
        st.stop()

db = init_connection()

# --- GEETA GYAAN API & LOGIC ---
VERSE_COUNTS = [47, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 35, 27, 20, 24, 28, 78]
TOTAL_VERSES = sum(VERSE_COUNTS)

def get_chapter_and_verse_from_day(day_index):
    """Maps a day number (1-700) to a specific chapter and verse."""
    shlok_number = (day_index % TOTAL_VERSES)
    if shlok_number == 0: shlok_number = TOTAL_VERSES

    chapter = 1
    verse_count_sum = 0
    for count in VERSE_COUNTS:
        verse_count_sum += count
        if shlok_number <= verse_count_sum:
            verse = shlok_number - (verse_count_sum - count)
            return chapter, verse
        chapter += 1
    return 18, 78 # Fallback to the last verse

@st.cache_data(ttl=86400) # Cache the result for a full day (86400 seconds)
def fetch_daily_shlok_from_rapidapi(chapter, verse):
    """Fetches a specific shlok from the Bhagavad Gita RapidAPI."""
    rapid_api_key = st.secrets.get("api_keys", {}).get("rapidapi") or st.secrets.get("rapidapi")
    if not rapid_api_key:
        st.error("RapidAPI key not found in secrets.")
        return None
    url = f"https://bhagavad-gita3.p.rapidapi.com/v2/chapters/{chapter}/verses/{verse}/"
    headers = {
        "X-RapidAPI-Key": rapid_api_key,
        "X-RapidAPI-Host": "bhagavad-gita3.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the Geeta API. Please try again later. Error: {e}")
        return None

# --- WELLNESS EXERCISES & SCORING LOGIC ---
WELLNESS_GAMES = {
    "🧠 Cognitive Games": [
        "Number Tap Challenge", "Memory Match", "Brain Teasers",
        "Word Scramble", "Simon Says"
    ],
    "👁️ Vision & Eye Health": ["20-20-20 Reminder", "Focus Shift Drill"],
    "🌬️ Mindfulness & Breathing": ["Breathing Pacer", "Guided Body Scan"],
    "❤️ Emotional & Reflective": ["Gratitude Wheel", "Compliment Generator"]
}

def save_daily_score(username, game_name, score):
    """Saves a game score to Firestore for the current day."""
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        doc_ref = db.collection("users").document(username).collection("wellness_scores").document(today_str)
        doc_ref.set({game_name: score}, merge=True)
    except Exception as e:
        st.warning(f"Could not save score: {e}")

def display_daily_scores(username):
    """Fetches and displays today's scores from Firestore in a clean table."""
    st.subheader("🏆 Your Daily Scorecard")
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        doc_ref = db.collection("users").document(username).collection("wellness_scores").document(today_str)
        scores_doc = doc_ref.get()

        if not scores_doc.exists or not scores_doc.to_dict():
            st.info("You haven't completed any activities today. Play a game to see your score!")
            return

        scores = scores_doc.to_dict()
        score_data = []
        for category, games in WELLNESS_GAMES.items():
            for game in games:
                if game in scores:
                    score_val = scores[game]
                    display_score = "✅ Completed" if isinstance(score_val, str) else f"{score_val}"
                    score_data.append({
                        "Category": category,
                        "Activity": game,
                        "Score / Status": display_score
                    })
        
        if not score_data:
            st.info("You haven't completed any activities today. Play a game to see your score!")
            return

        with st.container(border=True):
            df = pd.DataFrame(score_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Activity": st.column_config.TextColumn("Activity", width="medium"),
                    "Score / Status": st.column_config.TextColumn("Score / Status", width="small"),
                }
            )

    except Exception as e:
        st.error(f"Could not load scores: {e}")

# --- Gamification & Trend Analysis Functions ---
def calculate_streak(user_id, collection_name):
    try:
        entries_ref = db.collection("users").document(user_id).collection(collection_name)
        entries_stream = entries_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        unique_dates = {entry.to_dict()["timestamp"].date() for entry in entries_stream if "timestamp" in entry.to_dict()}
        if not unique_dates: return 0
        sorted_dates = sorted(list(unique_dates), reverse=True)
        streak = 0
        today, yesterday = datetime.now().date(), datetime.now().date() - timedelta(days=1)
        if sorted_dates and (sorted_dates[0] == today or sorted_dates[0] == yesterday):
            streak = 1
            for i in range(len(sorted_dates) - 1):
                if (sorted_dates[i] - sorted_dates[i+1]).days == 1:
                    streak += 1
                else: break
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
    secret_key = st.secrets.get("SECRET_KEY") or st.secrets.get("api_keys", {}).get("secret_key", "default_secret")
    return hashlib.sha256((password + secret_key).encode()).hexdigest()

def signup_user(username, password):
    users_ref = db.collection("app_users")
    if users_ref.document(username).get().exists: return False, "Username already exists."
    hashed_password = hash_password(password)
    users_ref.document(username).set({"password": hashed_password})
    return True, "Signup successful! Please log in."

def login_user(username, password):
    users_ref = db.collection("app_users")
    doc = users_ref.document(username).get()
    if not doc.exists: return False
    stored_password = doc.to_dict().get("password")
    hashed_password = hash_password(password)
    if stored_password == hashed_password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.permanent_login = True
        return True
    return False

# --- Text-to-Speech Function ---
@st.cache_data(ttl=3600)
def text_to_audio(text, lang='en'):
    try:
        clean_text = re.sub(r'\*\*', '', text)
        tts = gTTS(text=clean_text, lang=lang)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Could not generate audio. Error: {e}")
        return None

# --- UNIFIED GENERATIVE AI FUNCTIONS ---
def get_gemini_keys():
    keys = []
    for i in range(1, 11):
        key_name = f"GEMINI_API_KEY_{i}"
        key = st.secrets.get(key_name) or st.secrets.get("api_keys", {}).get(key_name)
        if key: keys.append(key)
    return keys

def get_groq_keys():
    keys = []
    for i in range(1, 11):
        key_name = f"GROQ_API_KEY_{i}"
        key = st.secrets.get(key_name) or st.secrets.get("api_keys", {}).get(key_name)
        if key: keys.append(key)
    return keys

def generate_ai_response(prompt, service_preference=['groq', 'gemini']):
    """Tries a list of services in order to get an AI response."""
    for service in service_preference:
        if service == 'groq':
            keys = get_groq_keys()
            if not keys: continue
            for key in keys:
                try:
                    api_url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                    payload = {"messages": [{"role": "user", "content": prompt}], "model": "llama3-8b-8192"}
                    response = requests.post(api_url, json=payload, headers=headers, timeout=25)
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
                except Exception:
                    continue
        
        elif service == 'gemini':
            keys = get_gemini_keys()
            if not keys: continue
            for key in keys:
                try:
                    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={key}"
                    payload = {"contents": [{"parts": [{"text": prompt}]}]}
                    response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=25)
                    response.raise_for_status()
                    return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    continue
    
    st.error("All AI services failed. Please check your API keys and try again later.")
    return None

@st.cache_data(ttl=86400)
def translate_text(text_list, target_language):
    if target_language.lower() == 'english': return text_list
    combined_text = "|||".join(text_list)
    prompt = f"Translate the following text to {target_language}. Keep the '|||' separators between each item:\n{combined_text}"
    response = generate_ai_response(prompt)
    if response:
        return [item.strip() for item in response.split('|||')]
    st.warning(f"AI translation to {target_language} failed. Showing English.")
    return text_list

def generate_story(mood, lang='en'):
    prompt = (f"Find a public domain short story that would be comforting and uplifting for someone feeling {mood}. Provide a summary of the story (around 400-500 words) in {lang}. At the end, include:\n**Title:** [Story Title]\n**Author:** [Author Name]")
    response = generate_ai_response(prompt)
    return response or "Could not retrieve a story at this time."

def get_activities(mood, lang='en'):
    prompt = (f"For someone feeling {mood}, suggest 3 short-term activities, 2 long-term activities, and 2 psychological techniques. Provide the response in {lang}. Format as:\n### Short-Term Relief\n- ...\n### Long-Term Well-being\n- ...\n### Psychological Techniques:\n- ...")
    response = generate_ai_response(prompt)
    return response or "Could not retrieve activities at this time."

# --- Model Loading & Analysis Functions ---
@st.cache_resource
def load_models():
    goemotions_model_name = "monologg/bert-base-cased-goemotions-original"
    sentiment_model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    
    hf_token = st.secrets.get("HUGGING_FACE_API_KEY") or st.secrets.get("api_keys", {}).get("hugging_face_api_key")
    if not hf_token: 
        st.error("Hugging Face API Key not found in secrets.")
        st.stop()
    
    # CORRECTED: Explicitly pass the token to all from_pretrained calls
    goemotions_tokenizer = AutoTokenizer.from_pretrained(goemotions_model_name, token=hf_token)
    goemotions_model = AutoModelForSequenceClassification.from_pretrained(goemotions_model_name, token=hf_token)
    
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, token=hf_token)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, token=hf_token)
    
    emotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
    emotions = requests.get(emotions_url).text.strip().split('\n')
    
    return goemotions_tokenizer, goemotions_model, sentiment_tokenizer, sentiment_model, emotions

goemotions_tokenizer, goemotions_model, sentiment_tokenizer, sentiment_model, emotions = load_models()


def detect_emotion(text):
    inputs = goemotions_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = goemotions_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    top_probs, top_ids = torch.topk(probs, 3)
    return [(emotions[i], float(top_probs[0][idx])) for idx, i in enumerate(top_ids[0])]

def detect_mental_state(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = sentiment_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    neg_prob = probs[0][0].item()
    if neg_prob > 0.5 or torch.argmax(probs) == 0:
        return "depressed", neg_prob
    else:
        non_dep_prob = probs[0][1].item() + probs[0][2].item()
        return "non-depressed", non_dep_prob

# --- Question Rotation Logic ---
QUESTION_POOL = ["How did you feel when you woke up today?", "How was your day at work or school?", "Did you talk to friends or family today? How did that feel?", "What was the most stressful part of your day?", "How are you feeling right now?", "Did anything make you laugh today?", "What was something you were proud of today?", "How would you describe your energy level today?", "Did you feel connected to others today?", "Was there a moment of peace or joy today?", "Did you feel lonely at any point?", "What frustrated you the most today?", "Did you feel anxious or nervous? What triggered it?", "Was today better or worse than yesterday? Why?", "Did you do something just for yourself today?", "What's one thing you're looking forward to tomorrow?", "Describe a small victory you had today.", "What's something that has been on your mind lately?"]
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
    if len(available_questions) < 5: available_questions = QUESTION_POOL
    selected_questions = random.sample(available_questions, 5)
    new_log_entry = {"date": today.strftime("%Y-%m-%d"), "questions": selected_questions}
    log_data.append(new_log_entry)
    log_ref.set({"history": log_data[-10:]})
    return selected_questions

# --- State Management ---
if 'page' not in st.session_state: st.session_state.page = 'Home'
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'language' not in st.session_state: st.session_state.language = 'English'
if 'responses' not in st.session_state: st.session_state.responses = {}
if 'logged_in' not in st.session_state: st.session_state.logged_in = st.session_state.get("permanent_login", False)
if 'username' not in st.session_state: st.session_state.username = None
if 'share_anonymously' not in st.session_state: st.session_state.share_anonymously = False
def set_page(page_name): st.session_state.page = page_name

# --- MAIN APP ---
if not st.session_state.logged_in:
    st.title("Welcome to Green Minds 🌱")
    st.markdown("Your AI-powered mental wellness journal.")
    choice = st.selectbox("Login or Signup?", ["Login", "Signup"])
    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if login_user(username, password):
                st.success("Logged in successfully!")
                st.rerun()
            else: st.error("Invalid username or password.")
    else:
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        if st.button("Signup", use_container_width=True):
            success, message = signup_user(username, password)
            if success: st.success(message)
            else: st.error(message)
else:
    # --- AUTHENTICATED APP ---
    st.sidebar.title("Green Minds 🌱")
    st.sidebar.markdown(f"Welcome, **{st.session_state.username}**!")
    st.sidebar.button("Home", on_click=set_page, args=('Home',), use_container_width=True)
    st.sidebar.button("My People", on_click=set_page, args=('People',), use_container_width=True)
    st.sidebar.button("Geeta Gyaan", on_click=set_page, args=('Geeta',), use_container_width=True)
    st.sidebar.button("Journal", on_click=set_page, args=('Journal',), use_container_width=True)
    st.sidebar.button("Community", on_click=set_page, args=('Community',), use_container_width=True)
    st.sidebar.button("Wellness Exercises", on_click=set_page, args=('Exercises',), use_container_width=True)
    st.sidebar.button("Today's Results", on_click=set_page, args=('Results',), use_container_width=True)
    st.sidebar.button("Trends", on_click=set_page, args=('Trends',), use_container_width=True)
    st.sidebar.divider()

    LANGUAGE_CODES = {"English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de", "Japanese": "ja"}
    LANGUAGE_API_MAP = {"English": "english", "Hindi": "hindi", "Spanish": "spanish"}
    st.session_state.language = st.sidebar.selectbox("Select Language", list(LANGUAGE_CODES.keys()))

    if st.sidebar.button("Logout", use_container_width=True):
        for key in ["logged_in", "username", "permanent_login", "responses", "analysis_results"]:
            st.session_state.pop(key, None)
        st.rerun()

    # --- Page Rendering ---
    if st.session_state.page == 'Home':
        st.title("🧠 Daily Reflection")
        reflection_streak = calculate_streak(st.session_state.username, "mood_entries")
        st.metric(label="Reflection Streak", value=f"{reflection_streak} Days 🔥")
        st.markdown("Answer today's questions to get an analysis of your emotional state.")
        questions_en = get_daily_questions(st.session_state.username)
        questions_translated = translate_text(questions_en, st.session_state.language)
        for i, q_translated in enumerate(questions_translated):
            q_english = questions_en[i]
            st.session_state.responses[q_english] = st.text_area(q_translated, value=st.session_state.responses.get(q_english, ""), key=f"q_{i}", height=100)
        if st.button("🔍 Analyze My Day", use_container_width=True):
            answered_responses = [r for r in st.session_state.responses.values() if r and r.strip()]
            if not answered_responses:
                st.warning("Please answer at least one question before analyzing.")
            else:
                all_text = " ".join(answered_responses)
                with st.spinner("Analyzing your responses..."):
                    top_emotions = detect_emotion(all_text)
                    mental_state, mental_score = detect_mental_state(all_text)
                    st.session_state.analysis_results = {"top_emotions": top_emotions, "mental_state": mental_state, "mental_score": mental_score}
                    entry = {"timestamp": datetime.now(), "responses": json.dumps(st.session_state.responses), "emotion_1": top_emotions[0][0], "emotion_score_1": top_emotions[0][1], "emotion_2": top_emotions[1][0], "emotion_score_2": top_emotions[1][1], "emotion_3": top_emotions[2][0], "emotion_score_3": top_emotions[2][1], "mental_state": mental_state, "mental_score": mental_score}
                    try:
                        user_entries_ref = db.collection("users").document(st.session_state.username).collection("mood_entries")
                        user_entries_ref.add(entry)
                        st.success("✅ Your entry has been saved!")
                        st.session_state.page = 'Results'
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not save data to Firestore. Error: {e}")

    elif st.session_state.page == 'People':
        st.title("👥 My People")
        st.markdown("A place to remember the important people in your life, including yourself. This can be a helpful memory aid.")
        
        people_ref = db.collection("users").document(st.session_state.username).collection("people")
        
        with st.expander("➕ Add a New Person", expanded=False):
            with st.form("person_form", clear_on_submit=True):
                name = st.text_input("Name*")
                relation = st.text_input("Relation (e.g., Mother, Best Friend, Myself)")
                uploaded_photo = st.file_uploader("Upload a Photo (Optional)", type=["png", "jpg", "jpeg"])
                photo_url = st.text_input("Or enter a Photo URL (Optional)")
                memories = st.text_area("Key Memories & Stories")
                details = st.text_area("Important Details (e.g., Loves gardening, Birthday: March 15)")
                submitted = st.form_submit_button("Save Person")

                if submitted and name:
                    person_data = {"name": name, "relation": relation, "memories": memories, "details": details, "photo_url": photo_url, "photo_base64": None}
                    if uploaded_photo is not None:
                        image_bytes = uploaded_photo.getvalue()
                        base64_image = base64.b64encode(image_bytes).decode()
                        person_data["photo_base64"] = base64_image
                    people_ref.add(person_data)
                    st.success(f"Saved details for {name}!")
                    st.rerun()
        st.divider()

        try:
            people_docs = people_ref.stream()
            people_list = {doc.id: doc.to_dict() for doc in people_docs}
            if not people_list:
                st.info("You haven't added anyone yet. Click the expander above to add your first person!")
            
            if "view_person_id" in st.session_state and st.session_state.view_person_id:
                person_to_view = people_list.get(st.session_state.view_person_id)
                if person_to_view:
                    @st.dialog(f"{person_to_view['name']}'s Details")
                    def view_person_details():
                        if person_to_view.get("photo_base64"):
                            st.image(base64.b64decode(person_to_view["photo_base64"]))
                        elif person_to_view.get("photo_url"):
                            st.image(person_to_view["photo_url"])
                        st.subheader(f"Relation: {person_to_view['relation']}")
                        st.markdown("**Key Memories & Stories**")
                        st.write(person_to_view.get("memories", "N/A"))
                        st.markdown("**Important Details**")
                        st.write(person_to_view.get("details", "N/A"))
                        if st.button("Close"):
                            st.session_state.view_person_id = None
                            st.rerun()
                    view_person_details()

            cols = st.columns(3)
            for i, (doc_id, person) in enumerate(people_list.items()):
                with cols[i % 3]:
                    with st.container(border=True):
                        if person.get("photo_base64"):
                            st.image(base64.b64decode(person["photo_base64"]), caption=person["name"])
                        elif person.get("photo_url"):
                            st.image(person["photo_url"], caption=person["name"])
                        st.subheader(person["name"])
                        st.write(f"**Relation:** {person['relation']}")
                        
                        if st.button("View Full Details", key=f"view_{doc_id}", use_container_width=True):
                            st.session_state.view_person_id = doc_id
                            st.rerun()
                        
                        if st.button("Delete", key=f"del_{doc_id}", use_container_width=True):
                            people_ref.document(doc_id).delete()
                            st.rerun()
        except Exception as e:
            st.error(f"Could not load people from Firestore. Error: {e}")

    elif st.session_state.page == 'Geeta':
        st.title("📖 Geeta Gyaan (गीता ज्ञान)")
        st.markdown("<p style='text-align: center; font-style: italic;'>The Gita holds timeless answers to life's profound questions.</p>", unsafe_allow_html=True)
        st.divider()
        epoch_date = date(2024, 1, 1)
        today = date.today()
        days_since_epoch = (today - epoch_date).days
        chapter, verse = get_chapter_and_verse_from_day(days_since_epoch)
        daily_shlok_data = fetch_daily_shlok_from_rapidapi(chapter, verse)
        if daily_shlok_data:
            sanskrit_shlok = daily_shlok_data.get('text', 'Shlok not available.')
            selected_lang_api_name = LANGUAGE_API_MAP.get(st.session_state.language, "english")
            found_translation, found_meaning = None, None
            for t in daily_shlok_data.get('translations', []):
                if t.get('language') == selected_lang_api_name:
                    found_translation = t.get('description'); break
            for c in daily_shlok_data.get('commentaries', []):
                 if c.get('language') == selected_lang_api_name:
                    found_meaning = c.get('description'); break
            english_translation, english_meaning = "Translation not available.", "Meaning not available."
            for t in daily_shlok_data.get('translations', []):
                if t.get('language') == "english":
                    english_translation = t.get('description'); break
            for c in daily_shlok_data.get('commentaries', []):
                if c.get('language') == "english":
                    english_meaning = c.get('description'); break
            if not found_translation and st.session_state.language != "English":
                st.info(f"A direct translation to {st.session_state.language} is not available. Using AI to translate from English.")
                with st.spinner("Translating the divine wisdom for you..."):
                    translated_texts = translate_text([english_translation, english_meaning], st.session_state.language)
                    final_translation, final_meaning = translated_texts[0], translated_texts[1]
            else:
                final_translation = found_translation or english_translation
                final_meaning = found_meaning or english_meaning
            with st.container(border=True):
                st.subheader(f"Shlok (श्लोक) - Chapter {chapter}, Verse {verse}")
                st.markdown(f"<h3 style='text-align: center; font-family: sans-serif; line-height: 2;'>{sanskrit_shlok}</h3>", unsafe_allow_html=True)
            st.write("")
            with st.container(border=True):
                st.subheader("Translation (अनुवाद)")
                st.write(final_translation)
            st.write("")
            with st.container(border=True):
                st.subheader("Meaning (तात्पर्य)")
                st.write(final_meaning)
        else:
             st.error("Could not load the Shlok of the Day. Please try again tomorrow.")

    elif st.session_state.page == 'Journal':
        st.title("✍️ My Daily Journal")
        journal_streak = calculate_streak(st.session_state.username, "journal_entries")
        st.metric(label="Journaling Streak", value=f"{journal_streak} Days ✍️")
        today_str = datetime.now().strftime("%Y-%m-%d")
        journal_ref = db.collection("users").document(st.session_state.username).collection("journal_entries").document(today_str)
        try:
            doc = journal_ref.get()
            current_entry = doc.to_dict().get("text", "") if doc.exists else ""
        except Exception as e:
            st.error(f"Could not load journal entry: {e}"); current_entry = ""
        journal_text = st.text_area("How was your day? What's on your mind?", value=current_entry, height=400, key="journal_entry")
        st.checkbox("Share this entry anonymously with the community", key="share_anonymously")
        if st.button("💾 Save Journal Entry", use_container_width=True):
            if journal_text.strip():
                try:
                    journal_ref.set({"text": journal_text, "timestamp": datetime.now()})
                    st.success("Your journal entry has been saved!")
                    if st.session_state.share_anonymously:
                        shared_ref = db.collection("shared_journals").document()
                        shared_ref.set({"text": journal_text, "timestamp": datetime.now(), "reactions": {"❤️": 0, "🙏": 0, "🤗": 0}})
                        st.info("Your entry has been shared anonymously.")
                except Exception as e: st.error(f"Could not save journal entry. Error: {e}")
            else: st.warning("Please write something before saving.")
        st.divider()
        st.subheader("Past Entries")
        try:
            entries_stream = db.collection("users").document(st.session_state.username).collection("journal_entries").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(7).stream()
            entries_list = [entry.to_dict() for entry in entries_stream]
            if not entries_list: st.info("No past entries found.")
            else:
                for entry in entries_list:
                    with st.expander(f"**{entry['timestamp'].strftime('%Y-%m-%d')}**"):
                        st.write(entry.get("text"))
        except Exception as e: st.write("Could not load past entries.")

    elif st.session_state.page == 'Community':
        st.title("🤝 Community Reflections")
        st.markdown("Read anonymous reflections from other users. React to show your support.")
        try:
            shared_entries_ref = db.collection("shared_journals").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20)
            entries = shared_entries_ref.stream()
            entries_list = list(entries)
            if not entries_list: st.info("No community entries yet. Share your journal to start!")
            else:
                for entry in entries_list:
                    entry_data = entry.to_dict()
                    with st.container(border=True): st.markdown(entry_data['text'])
                    reactions = entry_data.get("reactions", {"❤️": 0, "🙏": 0, "🤗": 0})
                    cols = st.columns(3)
                    emojis = ["❤️", "🙏", "🤗"]
                    for i, col in enumerate(cols):
                        emoji = emojis[i]
                        if col.button(f"{emoji} {reactions.get(emoji, 0)}", key=f"{entry.id}_{emoji}"):
                            entry.reference.update({f"reactions.{emoji}": firestore.Increment(1)})
                            st.rerun()
                    st.divider()
        except Exception as e: st.error(f"Could not load community entries. Error: {e}")

    elif st.session_state.page == 'Exercises':
        st.title("🧘 Wellness Exercises")
        selected_category = st.selectbox("Choose a category:", list(WELLNESS_GAMES.keys()))
        selected_game = st.selectbox("Choose an activity:", WELLNESS_GAMES[selected_category])
        st.divider()
        with st.container(border=True):
            if selected_game == "Number Tap Challenge":
                st.subheader("🧠 Number Tap Challenge")
                st.markdown("Tap numbers 1 to N in order. Lower time is better!")
                N = st.slider("Difficulty (Number Count)", 5, 20, 9, key="nt_N")
                if st.button("🔁 Start New Game", use_container_width=True):
                    numbers = list(range(1, N + 1)); random.shuffle(numbers)
                    st.session_state.update({"number_grid": numbers, "current_number": 1, "start_time": time.time(), "game_over": False})
                    st.rerun()
                if 'game_over' in st.session_state and not st.session_state.game_over:
                    grid_size = int(N ** 0.5) + 1
                    cols = st.columns(grid_size)
                    for idx, num in enumerate(st.session_state.number_grid):
                        col = cols[idx % grid_size]
                        display_text = "✅" if num == "✅" else str(num)
                        if col.button(display_text, key=f"tap_{idx}", use_container_width=True):
                            if num == st.session_state.current_number:
                                st.session_state.number_grid[idx] = "✅"; st.session_state.current_number += 1
                                if st.session_state.current_number > N: st.session_state.update({"game_over": True, "end_time": time.time()})
                            else: st.warning("❌ Wrong number!")
                            st.rerun()
                if 'game_over' in st.session_state and st.session_state.game_over:
                    duration = st.session_state.end_time - st.session_state.start_time
                    st.success(f"🎉 You finished in **{duration:.2f} seconds**!")
                    save_daily_score(st.session_state.username, selected_game, round(duration, 2))
            elif selected_game == "Memory Match":
                st.subheader("🤔 Memory Match")
                st.markdown("Find matching pairs. Fewer attempts are better!")
                if 'memory_emojis' not in st.session_state:
                    st.session_state.memory_emojis = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼']
                num_pairs = st.slider("Number of Pairs", 2, 8, 4)
                if st.button("New Memory Game", use_container_width=True):
                    emojis = st.session_state.memory_emojis[:num_pairs] * 2; random.shuffle(emojis)
                    st.session_state.update({"memory_board": emojis, "memory_revealed": [False] * len(emojis), "memory_first_flip": None, "memory_matches": 0, "memory_attempts": 0})
                    st.rerun()
                if 'memory_board' in st.session_state:
                    if st.session_state.memory_matches == num_pairs:
                        st.success(f"🎉 You found all pairs in {st.session_state.memory_attempts} attempts!")
                        save_daily_score(st.session_state.username, selected_game, st.session_state.memory_attempts)
                    else:
                        st.info(f"Matches: {st.session_state.memory_matches}/{num_pairs} | Attempts: {st.session_state.memory_attempts}")
                        grid_size = 4
                        cols = st.columns(grid_size)
                        for i, emoji in enumerate(st.session_state.memory_board):
                            revealed = st.session_state.memory_revealed[i]
                            if cols[i % grid_size].button(emoji if revealed else '❓', key=f"mem_{i}", use_container_width=True, disabled=revealed):
                                if not revealed:
                                    st.session_state.memory_revealed[i] = True
                                    if st.session_state.memory_first_flip is None:
                                        st.session_state.memory_first_flip = i
                                    else:
                                        first_idx = st.session_state.memory_first_flip
                                        if st.session_state.memory_board[first_idx] == emoji: st.session_state.memory_matches += 1
                                        else: st.session_state.memory_revealed[first_idx] = False; st.session_state.memory_revealed[i] = False
                                        st.session_state.memory_first_flip = None
                                        st.session_state.memory_attempts += 1
                                    st.rerun()
            elif selected_game == "Brain Teasers":
                st.subheader("🧩 Brain Teasers")
                st.markdown("A new riddle is generated by AI each time!")
                if st.button("New Riddle"):
                    st.session_state.pop('current_riddle', None)
                    st.rerun()

                if 'current_riddle' not in st.session_state:
                    with st.spinner("Generating a new riddle..."):
                        prompt = "Generate a single, short, clever riddle with a one or two-word answer. Your entire response must follow this exact format, with no extra text or introductions:\nRiddle: [The riddle text]\nAnswer: [The answer text]"
                        response = generate_ai_response(prompt)
                        if response and "Riddle:" in response and "Answer:" in response:
                            parts = response.split("Answer:")
                            riddle_text = parts[0].replace("Riddle:", "").strip()
                            answer_text = parts[1].strip()
                            st.session_state.current_riddle = {"q": riddle_text, "a": answer_text, "h": None}
                        else:
                            st.error("Could not generate a riddle. Please try again.")
                            st.stop()
                
                riddle = st.session_state.current_riddle
                st.markdown(f"**{riddle['q']}**")

                if not riddle.get('h'):
                    if st.button("Show Hint"):
                        with st.spinner("Generating a hint..."):
                            hint_prompt = f"Give a simple, one-sentence hint for the following riddle. The hint should be easy to understand. Do not reveal the answer ('{riddle['a']}').\nRiddle: '{riddle['q']}'"
                            hint = generate_ai_response(hint_prompt)
                            if hint:
                                st.session_state.current_riddle['h'] = hint.strip()
                            else:
                                st.session_state.current_riddle['h'] = "Could not generate a hint right now."
                            st.rerun()
                else:
                    st.info(f"Hint: {riddle['h']}")

                with st.form(key=f"riddle_form_{riddle['q']}"):
                    user_answer = st.text_input("Your Answer:")
                    submitted = st.form_submit_button("Check Answer")
                    if submitted:
                        if user_answer.lower().strip() == riddle['a'].lower():
                            st.success("Correct! 🎉")
                            save_daily_score(st.session_state.username, selected_game, "Solved")
                            st.rerun()
                        else:
                            st.error(f"Not quite! The answer was: **{riddle['a']}**")

            elif selected_game == "Word Scramble":
                st.subheader("📝 Word Scramble")
                st.markdown("Unscramble a new AI-generated word each time.")
                if st.button("New Word"):
                    st.session_state.pop('scrambled_word', None)
                    st.rerun()

                if 'scrambled_word' not in st.session_state:
                    with st.spinner("Generating a new word..."):
                        prompt = "Give me a single, common, positive English word between 4 and 7 letters long. Only provide the word itself, with no extra text."
                        original_word = generate_ai_response(prompt)
                        if original_word and len(original_word.split()) == 1:
                            original_word = original_word.strip().lower()
                            scrambled_list = list(original_word)
                            random.shuffle(scrambled_list)
                            st.session_state.update({"original_word": original_word, "scrambled_word": "".join(scrambled_list), "scramble_solved": False})
                        else:
                            st.error("Could not generate a word. Please try again.")
                            st.stop()
                
                if not st.session_state.get('scramble_solved', False):
                    st.markdown(f"Unscramble this word: ## `{st.session_state.scrambled_word}`")
                    with st.form(key=f"scramble_form_{st.session_state.scrambled_word}"):
                        user_guess = st.text_input("Your Guess:").lower()
                        submitted = st.form_submit_button("Submit")
                        if submitted:
                            if user_guess == st.session_state.original_word:
                                st.success("You got it! ✅"); st.session_state.scramble_solved = True
                                save_daily_score(st.session_state.username, selected_game, "Solved"); st.rerun()
                            else: st.error("Try again! ❌")
                else:
                    st.success("Correct! Try another word.")

            elif selected_game == "Simon Says":
                st.subheader("🎨 Simon Says")
                st.markdown("Watch the sequence, then repeat it. Higher level is better!")
                colors = {"🔴": "red", "🟢": "green", "🔵": "blue", "🟡": "yellow"}
                if st.button("Start New Simon Game", use_container_width=True):
                    st.session_state.update({"simon_sequence": [random.choice(list(colors.keys()))], "simon_player_sequence": [], "simon_state": "display"})
                    st.rerun()
                if 'simon_state' in st.session_state:
                    st.info(f"Level: {len(st.session_state.simon_sequence)}")
                    if st.session_state.simon_state == "display":
                        st.write("Watch carefully..."); placeholder = st.empty(); time.sleep(1)
                        for color_emoji in st.session_state.simon_sequence:
                            placeholder.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{color_emoji}</h1>", unsafe_allow_html=True); time.sleep(0.8)
                            placeholder.empty(); time.sleep(0.2)
                        st.session_state.simon_state = "guess"; st.rerun()
                    elif st.session_state.simon_state == "guess":
                        st.write("Your turn!"); cols = st.columns(len(colors))
                        for i, (emoji, color) in enumerate(colors.items()):
                            if cols[i].button(emoji, key=f"simon_{color}"):
                                st.session_state.simon_player_sequence.append(emoji)
                                if st.session_state.simon_player_sequence != st.session_state.simon_sequence[:len(st.session_state.simon_player_sequence)]:
                                    st.session_state.simon_state = "game_over"
                                elif len(st.session_state.simon_player_sequence) == len(st.session_state.simon_sequence):
                                    st.session_state.simon_sequence.append(random.choice(list(colors.keys()))); st.session_state.simon_player_sequence = []; st.session_state.simon_state = "display"
                                st.rerun()
                        st.write(f"Your input: {' '.join(st.session_state.simon_player_sequence)}")
                    elif st.session_state.simon_state == "game_over":
                        final_score = len(st.session_state.simon_sequence) - 1
                        st.error(f"Game Over! You reached level {final_score}.")
                        save_daily_score(st.session_state.username, selected_game, final_score)
            elif selected_game == "20-20-20 Reminder":
                st.subheader("👁️ Eye Health Reminder")
                st.markdown("Relax your eyes with the **20-20-20 Rule** and a blinking routine.")
                st.info("**The Rule:** Every 20 minutes, look at something 20 feet away for 20 seconds.")
                if st.button("🟢 Start Blinking Routine", use_container_width=True):
                    placeholder = st.empty()
                    for i in range(10):
                        placeholder.markdown(f"<h2 style='text-align:center;'>Blink Now 👁️👁️ ({i+1}/10)</h2>", unsafe_allow_html=True); time.sleep(1.2)
                        placeholder.markdown(f"<h2 style='text-align:center;'>Relax 😌</h2>", unsafe_allow_html=True); time.sleep(1.2)
                    placeholder.success("Session complete! Great job caring for your eyes. 👏")
                    save_daily_score(st.session_state.username, selected_game, "Completed")
            elif selected_game == "Focus Shift Drill":
                st.subheader("🔭 Focus Shift Drill")
                st.markdown("""This exercise helps strengthen your eye muscles.
                1.  **Hold your thumb** about 10 inches from your face and focus on it for 15 seconds.
                2.  **Shift your gaze** to a distant object (at least 20 feet away) and focus on it for 15 seconds.
                3.  **Return your focus** to your thumb. Repeat this 5 times.""")
                if st.button("Mark as Complete"):
                    st.balloons(); save_daily_score(st.session_state.username, selected_game, "Completed"); st.success("Activity marked as complete for today!")
            elif selected_game == "Breathing Pacer":
                st.subheader("🌬️ Breathing Pacer (Box Breathing)")
                st.markdown("Follow the guide to breathe in a 4-4-4-4 pattern for 5 cycles.")
                if st.button("Start Pacer", use_container_width=True):
                    placeholder = st.empty()
                    for i in range(5):
                        placeholder.markdown(f"<h1 style='text-align:center;'>Inhale... (4s)</h1>", unsafe_allow_html=True); time.sleep(4)
                        placeholder.markdown(f"<h1 style='text-align:center;'>Hold... (4s)</h1>", unsafe_allow_html=True); time.sleep(4)
                        placeholder.markdown(f"<h1 style='text-align:center;'>Exhale... (4s)</h1>", unsafe_allow_html=True); time.sleep(4)
                        placeholder.markdown(f"<h1 style='text-align:center;'>Hold... (4s)</h1>", unsafe_allow_html=True); time.sleep(4)
                    placeholder.success("Breathing session complete. Feel the calm. 😌")
                    save_daily_score(st.session_state.username, selected_game, "Completed")
            elif selected_game == "Guided Body Scan":
                st.subheader("🧘 Guided Body Scan Meditation")
                st.markdown("Press play to begin a short, relaxing body scan meditation.")
                meditation_script = "Welcome. Find a comfortable position. Close your eyes. Bring your attention to your breath. Now, bring your awareness to the tips of your toes. Notice any sensations. Slowly, let this awareness travel up to your feet, your ankles. Continue this awareness up through your calves, your knees. Feel the weight of your legs. Now bring your attention to your thighs, your hips. Let your awareness flow into your stomach, your chest. Notice your heartbeat. Become aware of your fingertips, your hands, and up through your arms to your shoulders. Let go of any tension. Finally, bring your awareness to your neck, your jaw, your cheeks, your eyes, and the very top of your head. Rest in this full-body awareness. When you're ready, gently wiggle your fingers and toes, and slowly open your eyes."
                audio_bytes = text_to_audio(meditation_script, lang='en')
                if audio_bytes: st.audio(audio_bytes, format='audio/mp3')
                if st.button("Mark as Complete"):
                    save_daily_score(st.session_state.username, selected_game, "Completed"); st.success("Activity marked as complete for today!")
            elif selected_game == "Gratitude Wheel":
                st.subheader("🌟 Gratitude Wheel")
                st.markdown("Spin the wheel and reflect on something you're grateful for.")
                prompts = ["A person who has helped you.", "A simple pleasure you enjoyed.", "A skill you have.", "Something beautiful in nature.", "A favorite memory.", "A food that you love."]
                if st.button("Spin the Wheel!", use_container_width=True):
                    prompt = random.choice(prompts)
                    st.success(f"**Reflect on:** {prompt}")
                    save_daily_score(st.session_state.username, selected_game, "Completed")
            elif selected_game == "Compliment Generator":
                st.subheader("💖 Compliment Generator")
                st.markdown("Click the button for a small reminder of how great you are.")
                compliments = ["You are doing a great job.", "Your resilience is inspiring.", "It's okay to not be okay.", "You are worthy of peace.", "Your efforts are not going unnoticed.", "You've overcome so much.", "You are more than enough."]
                if st.button("Generate Compliment", use_container_width=True):
                    st.success(f"✨ {random.choice(compliments)} ✨")
                    save_daily_score(st.session_state.username, selected_game, "Completed")
        st.divider()
        display_daily_scores(st.session_state.username)

    elif st.session_state.page == 'Results':
        st.title("✨ Today's Analysis")
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            dominant_emotion = results['top_emotions'][0][0]
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💬 Top Emotions Detected")
                for emo, score in results['top_emotions']: st.write(f"**{emo.capitalize()}**: {score:.2f}")
            with col2:
                st.subheader("🧠 Mental State Analysis")
                st.write(f"Detected State: **{results['mental_state'].upper()}**"); st.write(f"Confidence: **{results['mental_score']:.2f}**")
            st.divider()
            st.subheader("📖 A Story For You")
            with st.spinner("Finding and narrating a story for you..."):
                story_text = generate_story(dominant_emotion, st.session_state.language)
                if story_text and "Could not retrieve" not in story_text:
                    st.info(story_text)
                    lang_code = LANGUAGE_CODES.get(st.session_state.language, 'en')
                    audio_bytes = text_to_audio(story_text, lang=lang_code)
                    if audio_bytes: st.audio(audio_bytes, format='audio/mp3')
                else: st.warning("Could not generate a story at this time. Please try again later.")
            st.divider()
            st.subheader("💡 Activity Suggestions")
            with st.spinner("Finding some helpful activities..."):
                activities_markdown = get_activities(dominant_emotion, st.session_state.language)
                if activities_markdown: st.markdown(activities_markdown)
                else: st.error("Could not retrieve activities at this time.")
        else: st.info("Please complete the questionnaire on the 'Home' page to see your results.")

    elif st.session_state.page == 'Trends':
        st.title("📈 Historical Trends")
        username = st.session_state.username
        with st.spinner("Loading historical data..."):
            st.subheader("Emotion Trends")
            try:
                user_entries_ref = db.collection("users").document(username).collection("mood_entries")
                entries_stream = user_entries_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(90).stream()
                entries_list = [doc.to_dict() for doc in entries_stream]
                if not entries_list:
                    st.info("No emotional data found. Submit an analysis to get started.")
                else:
                    df = pd.DataFrame(entries_list)
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
                    thirty_days_ago = datetime.now().date() - timedelta(days=30)
                    df_30_days = df[df['timestamp'] >= thirty_days_ago]
                    if not df_30_days.empty:
                        emotion_counts = df_30_days['emotion_1'].value_counts()
                        fig_pie = go.Figure(data=[go.Pie(labels=emotion_counts.index, values=emotion_counts.values, hole=.3)])
                        fig_pie.update_layout(title_text='Dominant Emotions (Last 30 Days)')
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        df_30_days['emotion_category'] = df_30_days['emotion_1'].map(EMOTION_TO_CATEGORY)
                        category_counts = df_30_days.groupby(['timestamp', 'emotion_category']).size().unstack(fill_value=0)
                        fig_area = go.Figure()
                        for cat in ['positive', 'negative', 'neutral']:
                            if cat in category_counts.columns:
                                fig_area.add_trace(go.Scatter(x=category_counts.index, y=category_counts[cat], mode='lines', stackgroup='one', name=cat.capitalize()))
                        fig_area.update_layout(title_text='Daily Emotion Category Count')
                        st.plotly_chart(fig_area, use_container_width=True)
                    else:
                        st.write("Not enough emotion data in the last 30 days for charts.")
            except Exception as e:
                st.error(f"Could not load emotion data from Firestore. Error: {e}")
            st.divider()
            st.subheader("Wellness Activity Trends")
            try:
                scores_ref = db.collection("users").document(username).collection("wellness_scores")
                scores_stream = scores_ref.limit(90).stream()
                scores_list = {doc.id: doc.to_dict() for doc in scores_stream}
                if not scores_list:
                    st.info("No activity data found. Complete some wellness exercises to see your trends!")
                else:
                    df_scores = pd.DataFrame.from_dict(scores_list, orient='index')
                    df_scores.index = pd.to_datetime(df_scores.index)
                    thirty_days_ago_ts = pd.Timestamp.now() - pd.Timedelta(days=30)
                    df_scores_30 = df_scores[df_scores.index >= thirty_days_ago_ts].copy()
                    
                    played_games = df_scores_30.count().sort_values(ascending=False)
                    if not played_games.empty:
                        fig_pie_games = go.Figure(data=[go.Pie(labels=played_games.index, values=played_games.values, hole=.3)])
                        fig_pie_games.update_layout(title_text='Activity Engagement (Last 30 Days)')
                        st.plotly_chart(fig_pie_games, use_container_width=True)

                    numerical_games = ['Number Tap Challenge', 'Memory Match', 'Simon Says']
                    available_numerical_games = [game for game in numerical_games if game in df_scores.columns]
                    if available_numerical_games:
                        selected_game_trend = st.selectbox("Select a game to see your score trend:", available_numerical_games)
                        help_text = "Lower is better" if selected_game_trend in ['Number Tap Challenge', 'Memory Match'] else "Higher is better"
                        st.line_chart(df_scores[selected_game_trend].dropna(), use_container_width=True)
                        st.caption(f"Trend for {selected_game_trend}. *Note: {help_text}.*")

                    completion_games_all = [game for sublist in WELLNESS_GAMES.values() for game in sublist if game not in numerical_games]
                    completion_games_in_df = [game for game in completion_games_all if game in df_scores_30.columns]
                    
                    if completion_games_in_df:
                        df_scores_30[completion_games_in_df] = df_scores_30[completion_games_in_df].apply(lambda x: x.notna().astype(int))
                        completion_counts = df_scores_30[completion_games_in_df].sum().sort_values(ascending=False)
                        completion_counts = completion_counts[completion_counts > 0] 
                        if not completion_counts.empty:
                            fig_bar_completed = go.Figure(data=[go.Bar(x=completion_counts.index, y=completion_counts.values)])
                            fig_bar_completed.update_layout(title_text='Total Completions (Last 30 Days)')
                            st.plotly_chart(fig_bar_completed, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load activity score data from Firestore. Error: {e}")
