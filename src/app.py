import streamlit as st
st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="💬", layout="wide")

import os
import sqlite3
import pickle
from datetime import datetime

# Guarded optional imports
_transformers_available = True
_speechrec_available = True
_torch_available = True

try:
    from transformers import pipeline, AutoConfig
except Exception as e:
    _transformers_available = False
    transformers_import_error = e

try:
    import speech_recognition as sr
except Exception:
    _speechrec_available = False

try:
    import torch
except Exception:
    _torch_available = False

# ----------------------------
# Paths (update if needed)
# ----------------------------
BASE_DIR = r"C:\Users\acer\Desktop\Mini proj 5th sem\project"
MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert_model")
BASELINE_PKL = os.path.join(BASE_DIR, "models", "baseline_LR.pkl")
DB_PATH = os.path.join(BASE_DIR, "data", "chat_history.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ----------------------------
# Label mapping (ensure this matches training)
# ----------------------------
id2label = {
    "LABEL_0": "neutral",
    "LABEL_1": "anxious",
    "LABEL_2": "depressed",
    "LABEL_3": "stressed",
    "LABEL_4": "suicidal"
}

# ----------------------------
# DB helpers
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_input TEXT,
                    predicted_emotion TEXT,
                    confidence REAL
                )''')
    conn.commit()
    conn.close()

def log_chat(user_input, emotion, confidence):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_logs (timestamp, user_input, predicted_emotion, confidence) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_input, emotion, float(confidence))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database logging failed: {e}")

init_db()

# ----------------------------
# Feedback dictionary
# ----------------------------
feedback_dict = {
    "neutral": "You're feeling balanced — that's great. Keep maintaining your calm energy.",
    "anxious": "Take a few deep breaths. Try journaling or a short walk to clear your mind.",
    "depressed": "Remember, small steps matter. Try talking to someone you trust.",
    "stressed": "Pause for a moment. A short break or listening to music might help.",
    "suicidal": "If you're having suicidal thoughts, please reach out to a helpline immediately: AASRA (91-9820466726) or iCall (91-9152987821). You are not alone."
}

# ----------------------------
# Model loading with fallback
# ----------------------------
@st.cache_resource
def load_distilbert_pipeline():
    if not _transformers_available:
        raise ImportError(f"transformers not available: {transformers_import_error}")
    # device selection: 0 for cuda, -1 for cpu
    device = 0 if (_torch_available and torch.cuda.is_available()) else -1
    try:
        pipe = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, device=device)
        return ("distilbert", pipe)
    except Exception as e:
        # bubble up the exception to caller
        raise RuntimeError(f"Failed to load DistilBERT pipeline: {e}")

@st.cache_resource
def load_baseline_model():
    if not os.path.exists(BASELINE_PKL):
        raise FileNotFoundError("Baseline model pickle not found at: " + BASELINE_PKL)
    with open(BASELINE_PKL, "rb") as f:
        model = pickle.load(f)
    return ("baseline", model)

@st.cache_resource
def load_model_with_fallback():
    # Try DistilBERT first (preferred)
    if _transformers_available:
        try:
            tag, model = load_distilbert_pipeline()
            return (tag, model)
        except Exception as e:
            st.warning(f"DistilBERT load failed, falling back to baseline model. Reason: {e}")
    # Try baseline
    try:
        tag, model = load_baseline_model()
        return (tag, model)
    except Exception as e:
        raise RuntimeError(f"Both DistilBERT and baseline model loading failed: {e}")

# ----------------------------
# Voice helper (optional)
# ----------------------------
def recognize_speech():
    if not _speechrec_available:
        st.warning("SpeechRecognition library not available. Install via `pip install SpeechRecognition` to enable voice input.")
        return None
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now.")
            audio = recognizer.listen(source, phrase_time_limit=6)
        text = recognizer.recognize_google(audio)
        st.success(f"🗣 You said: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please try again.")
    except sr.RequestError:
        st.error("Voice recognition service unavailable.")
    except Exception as e:
        st.error(f"Voice capture failed: {e}")
    return None

# ----------------------------
# UI and main app
# ----------------------------
st.title("💬 AI Mental Health Chatbot")

# Inform about missing packages early (avoid crash)
if not _transformers_available:
    st.error("The 'transformers' package failed to import. Reinstall with: pip install transformers==4.37.2")
if not _torch_available:
    st.info("PyTorch not available or failed to import. Model will run on CPU if possible.")

# Load model (with spinner)
model_tag = None
model_obj = None
try:
    with st.spinner("Loading model (DistilBERT preferred)..."):
        model_tag, model_obj = load_model_with_fallback()
    st.success(f"Model loaded: {model_tag}")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    model_tag = None
    model_obj = None

# Sidebar info
st.sidebar.header("Model & System Info")
if model_tag:
    st.sidebar.write(f"Using model: **{model_tag}**")
else:
    st.sidebar.write("No model available. Fix above errors.")

st.sidebar.write(f"Transformers installed: {_transformers_available}")
st.sidebar.write(f"SpeechRecognition installed: {_speechrec_available}")
st.sidebar.write(f"PyTorch installed: {_torch_available}")

# Input area
user_input = st.text_input("Type your message here...")
col1, col2 = st.columns([1,1])
with col1:
    voice_btn = st.button("🎙 Use Voice Input")
with col2:
    send_btn = st.button("Send")

if voice_btn:
    spoken_text = recognize_speech()
    if spoken_text:
        user_input = spoken_text

if send_btn:
    if not user_input or not user_input.strip():
        st.warning("Please enter a message before sending.")
    elif model_obj is None:
        st.error("No model loaded. Check errors in the sidebar and logs.")
    else:
        with st.spinner("Analyzing message..."):
            try:
                if model_tag == "distilbert":
                    res = model_obj(user_input)[0]
                    label = res.get("label") if isinstance(res, dict) else res[0].get("label")
                    score = res.get("score") if isinstance(res, dict) else res[0].get("score")
                    emotion = id2label.get(label, label)
                else:
                    # baseline model (sklearn-like) expects vectorized input; change as per your baseline pipeline
                    # Here we assume baseline model supports .predict and returns numeric labels 0..4
                    pred = model_obj.predict([user_input])[0]
                    mapping = {0: "neutral", 1: "anxious", 2: "depressed", 3: "stressed", 4: "suicidal"}
                    emotion = mapping.get(int(pred), str(pred))
                    score = None
                # Log and display
                log_chat(user_input, emotion, score or 0.0)
                st.subheader(f"Predicted Emotion: {emotion.capitalize()}")
                if score is not None:
                    st.caption(f"Confidence: {score:.2f}")
                    st.progress(float(score))
                st.success(feedback_dict.get(emotion.lower(), "I'm here to listen — tell me more."))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Sidebar recent chats
st.sidebar.header("Recent Chats")
try:
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT timestamp, user_input, predicted_emotion FROM chat_logs ORDER BY id DESC LIMIT 10")
        rows = c.fetchall()
        conn.close()
        if rows:
            for row in rows:
                st.sidebar.write(f"**{row[0]}** — *{row[2].capitalize()}*")
                st.sidebar.caption(f"> {row[1]}")
        else:
            st.sidebar.info("No chat history yet.")
    else:
        st.sidebar.info("Database not found.")
except Exception as e:
    st.sidebar.error(f"Failed to read chat history: {e}")
