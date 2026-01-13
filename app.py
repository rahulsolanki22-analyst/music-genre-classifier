import os
import time
import tempfile
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt


# -----------------------------
# Page & global style
# -----------------------------
st.set_page_config(page_title="Music Genre Classifier", layout="centered")

st.markdown("""
<style>
/* Background + base text */
.stApp {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(255,255,255,0.06), transparent 40%),
              radial-gradient(1000px 500px at 90% 20%, rgba(255,255,255,0.05), transparent 40%),
              linear-gradient(135deg, #10131a 0%, #1a2030 100%);
  color: #eef0f3;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
h1, .title-text {
  text-align:center;
  font-weight:800;
  letter-spacing:0.3px;
  margin-bottom:0.2rem;
  color:#f5f7fb;
}

/* Uploader */
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed rgba(200, 210, 230, 0.35) !important;
  background: rgba(255,255,255,0.02) !important;
  border-radius: 16px !important;
  transition: border-color .25s ease, transform .2s ease, background .25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(120, 160, 255, 0.65) !important;
  transform: translateY(-1px);
  background: rgba(120,160,255,0.06) !important;
}

/* Button */
.stButton > button {
  background: linear-gradient(135deg, #4f7cff, #6dd6ff);
  border: 0;
  color: #0b1220;
  border-radius: 12px;
  font-weight: 700;
  padding: 0.6rem 1rem;
  box-shadow: 0 8px 24px rgba(109, 214, 255, 0.2);
  transition: transform .15s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  filter: brightness(1.02);
  box-shadow: 0 10px 26px rgba(109, 214, 255, 0.28);
}

/* Card */
.pred-card {
  border-radius: 18px;
  padding: 18px 18px 8px 18px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.07);
  backdrop-filter: blur(6px);
  box-shadow: 0 10px 24px rgba(0,0,0,0.25);
}

/* Confidence ring */
.conf-ring {
  --p: 0;
  --fill: #4cd964;
  width: 120px;
  aspect-ratio: 1 / 1;
  border-radius: 50%;
  background: conic-gradient(var(--fill) calc(var(--p)*1%), #283044 0);
  display: grid;
  place-items: center;
  position: relative;
  transition: background 0.4s ease;
}
.conf-ring::before{
  content: "";
  position: absolute;
  inset: 10px;
  background: #0f1422;
  border-radius: 50%;
}
.conf-ring span{
  position: relative;
  font-weight: 800;
  font-size: 1.05rem;
  color: #e9ecf3;
}

/* Labels and chips */
.pred-label { font-size: 1.15rem; font-weight: 700; letter-spacing: .2px; }
.badge {
  display:inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 0.75rem; font-weight: 700; background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12); color: #dfe7ff;
}
.chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:6px; }
.chip { font-size:.78rem; padding:6px 10px; border-radius:999px;
        background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10); }

/* Small fade-in */
.fade-in { animation: fadein 500ms ease 1; }
@keyframes fadein { from { opacity:0; transform: translateY(6px); } to { opacity:1; transform: none; } }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Genre meta (icon + short text)
# -----------------------------
GENRE_META = {
    "blues":      {"icon": "🎹", "desc": "Guitar-driven patterns with expressive bends and call-and-response."},
    "classical":  {"icon": "🎼", "desc": "Orchestral and chamber music with structured forms and dynamics."},
    "country":    {"icon": "🤠", "desc": "Storytelling vocals, acoustic guitars, steady rhythms."},
    "disco":      {"icon": "🪩", "desc": "Four-on-the-floor beats, bass grooves, dance-focused production."},
    "hiphop":     {"icon": "🎤", "desc": "Rhythmic speech over beats, sampling, heavy drums."},
    "jazz":       {"icon": "🎷", "desc": "Improvisation, swing rhythms, extended harmonies."},
    "metal":      {"icon": "🎸", "desc": "Distorted guitars, aggressive drums, powerful vocals."},
    "pop":        {"icon": "🎧", "desc": "Catchy melodies, verse-chorus structures, polished production."},
    "reggae":     {"icon": "🥁", "desc": "Off-beat rhythms, deep bass lines, relaxed groove."},
    "rock":       {"icon": "🎸", "desc": "Electric guitars, strong backbeat, energetic vocals."},
}


# -----------------------------
# Model + preprocessing
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model = tf.keras.models.load_model('music_genre_cnn.h5', compile=False)
    scaler = joblib.load('scaler.joblib')
    genre_mapping = {
        0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
        5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
    }
    return model, scaler, genre_mapping


def load_audio_to_temp(file_like, sample_rate=22050, duration=30):
    """Return y, sr from uploaded file by writing to a temp wav first."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        data = file_like.read()
        tmp.write(data)
        path = tmp.name
    try:
        y, sr = librosa.load(path, sr=sample_rate, duration=duration)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    return y, sr


def extract_features_from_y(y, sr, n_mfcc=13, n_chroma=12):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    chroma_mean = np.mean(chroma, axis=1)

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_cent_mean = np.mean(spec_cent)

    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_roll_mean = np.mean(spec_roll)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    features = np.concatenate([
        mfcc_mean, chroma_mean,
        np.array([spec_cent_mean, spec_roll_mean, zcr_mean])
    ])
    return features


def predict_genre_from_features(features):
    model, scaler, genre_mapping = load_model_and_scaler()
    features_reshaped = features.reshape(1, -1)
    features_scaled = scaler.transform(features_reshaped)
    features_cnn = np.expand_dims(features_scaled, axis=-1)
    probs = model.predict(features_cnn, verbose=0)[0]
    idx = int(np.argmax(probs))
    genre = genre_mapping.get(idx, "Unknown")
    conf = float(probs[idx]) * 100.0
    all_probs = {genre_mapping[i]: float(p) * 100.0 for i, p in enumerate(probs)}
    return genre, conf, all_probs


# -----------------------------
# Small helpers (UI)
# -----------------------------
def animate_conf_ring(container, confidence, fill_hex):
    """Animate the CSS ring from 0 to confidence%."""
    target = int(round(confidence))
    for p in range(0, target + 1, 3):
        container.markdown(
            f"""
            <div class="conf-ring" style="--p:{p}; --fill:{fill_hex};">
              <span>{p:.0f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.01)


def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig, clear_figure=True)


def plot_prob_bars(all_probs):
    genres, vals = zip(*sorted(all_probs.items(), key=lambda x: -x[1]))
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.bar(genres, vals)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Genre probabilities", fontsize=10)
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig, clear_figure=True)


def push_history(genre, confidence):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "genre": genre,
        "confidence": round(confidence, 1)
    })
    # keep last 6
    st.session_state.history = st.session_state.history[-6:]


# -----------------------------
# App
# -----------------------------
def main():
    st.markdown("<h1 class='title-text'>Music Genre Classification</h1>", unsafe_allow_html=True)
    st.caption("Upload a .wav file. The app predicts the genre using a CNN and shows confidence.")

    uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

    if uploaded_file is None:
        st.info("Select a .wav file to start.")
        return

    # Load audio and show player
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Processing audio and classifying..."):
        # 1) audio -> y,sr
        y, sr = load_audio_to_temp(uploaded_file)
        # 2) features
        feats = extract_features_from_y(y, sr)
        # 3) predict
        genre, confidence, all_probs = predict_genre_from_features(feats)
        push_history(genre, confidence)

    # UI columns
    left, right = st.columns([0.55, 0.45], vertical_alignment="center")

    # Left: waveform + bars
    with left:
        st.markdown("#### Audio analysis")
        plot_waveform(y, sr)
        plot_prob_bars(all_probs)

        with st.expander("What the model uses"):
            st.write(
                "- MFCC, Chroma, Spectral Centroid/Rolloff, Zero-Crossing Rate\n"
                "- Features are scaled, then fed to a CNN\n"
                "- Output is a probability for each genre"
            )

        with st.expander("Recent predictions (this session)"):
            if "history" in st.session_state and st.session_state.history:
                for h in reversed(st.session_state.history):
                    st.write(f"{h['time']} — {h['genre'].capitalize()} ({h['confidence']}%)")
            else:
                st.write("No history yet.")

    # Right: result card with animated ring + description
    with right:
        if confidence >= 80:
            fill = "#4cd964"
        elif confidence >= 50:
            fill = "#ffd166"
        else:
            fill = "#ff6b6b"

        icon = GENRE_META.get(genre.lower(), {}).get("icon", "🎵")
        desc = GENRE_META.get(genre.lower(), {}).get("desc", "No description available.")

        st.markdown(
            f"""
            <div class="pred-card fade-in">
              <div class="pred-label">Predicted genre</div>
              <div style="font-size:1.9rem; font-weight:800; margin:.15rem 0 .6rem 0;">
                {icon} {genre.capitalize()}
              </div>

              <div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
                <div id="ring-slot"></div>
                <div style="flex:1; min-width:160px;">
                  <div class="badge">Confidence</div>
                  <div style="margin-top:6px; opacity:.85; font-size:0.92rem;">
                    Higher means the model is more sure about the prediction.
                  </div>
                </div>
              </div>

              <div style="margin-top:14px;">
                <div class="badge">About the genre</div>
                <div style="margin-top:8px; opacity:.92;">{desc}</div>
              </div>

              <div class="chips" style="margin-top:14px;">
                <div class="chip">CNN</div>
                <div class="chip">MFCC • Chroma • ZCR</div>
                <div class="chip">Scaled features</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # mount animated ring into the placeholder
        ring_placeholder = st.empty()
        animate_conf_ring(ring_placeholder, confidence, fill)

        # Small celebration if very sure
        if confidence >= 90:
            st.balloons()


if __name__ == "__main__":
    main()
