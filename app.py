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
def load_all_models_and_scaler():
    models = {}
    
    # Try loading each model and provide a warning if it doesn't exist
    model_paths = {
        "Random Forest (Recommended)": ("random_forest_model.joblib", "sklearn"),
        "Support Vector Machine (SVM)": ("svm_model.joblib", "sklearn"),
        "Logistic Regression": ("logistic_regression_model.joblib", "sklearn"),
        "Convolutional Neural Network (CNN)": ("music_genre_cnn.h5", "keras")
    }
    
    for name, (path, mtype) in model_paths.items():
        if os.path.exists(path):
            try:
                if mtype == "sklearn":
                    models[name] = joblib.load(path)
                elif mtype == "keras":
                    models[name] = tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                st.warning(f"Error loading model '{name}' from '{path}': {e}")
        else:
            st.warning(f"Model file '{path}' for '{name}' was not found. Please ensure it has been trained.")

    scaler = None
    if os.path.exists('scaler.joblib'):
        try:
            scaler = joblib.load('scaler.joblib')
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
    else:
        st.error("Scaler file 'scaler.joblib' was not found! Please run training first.")
        
    genre_mapping = {
        0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
        5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
    }
    return models, scaler, genre_mapping


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


def extract_features_segmented(y, sr, n_mfcc=13, n_chroma=12, n_fft=2048, hop_length=512, num_segments=10):
    """Segment audio into equal chunks and extract features from each chunk, matching training."""
    # Ensure y has at least 30 seconds worth of samples (661500 samples for sr=22050)
    expected_samples = sr * 30
    if len(y) < expected_samples:
        y = np.pad(y, (0, expected_samples - len(y)), 'constant')
    else:
        y = y[:expected_samples]
        
    samples_per_segment = expected_samples // num_segments
    features_list = []
    
    for s in range(num_segments):
        start_sample = s * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment = y[start_sample:end_sample]
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spec_cent_mean = np.mean(spec_cent)
        
        # Spectral Rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spec_roll_mean = np.mean(spec_roll)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=segment, hop_length=hop_length)
        zcr_mean = np.mean(zcr)
        
        # Combine
        features = np.concatenate([
            mfcc_mean, chroma_mean,
            np.array([spec_cent_mean, spec_roll_mean, zcr_mean])
        ])
        features_list.append(features)
        
    return np.array(features_list)


def predict_genre_from_segments(segmented_features, model_name):
    models, scaler, genre_mapping = load_all_models_and_scaler()
    
    if model_name not in models:
        raise ValueError(f"Selected model '{model_name}' is not loaded.")
    if scaler is None:
        raise ValueError("Scaler is not loaded.")
        
    model = models[model_name]
    
    # Scale each segment's features
    import pandas as pd
    features_df = pd.DataFrame(segmented_features, columns=[str(i) for i in range(28)])
    features_scaled = scaler.transform(features_df)
    
    if model_name == "Convolutional Neural Network (CNN)":
        features_cnn = np.expand_dims(features_scaled, axis=-1)
        probs_all = model.predict(features_cnn, verbose=0)
    else:
        probs_all = model.predict_proba(features_scaled)
        
    # Ensemble: Average probabilities across all 10 segments
    avg_probs = np.mean(probs_all, axis=0)
    probs_percentage = avg_probs * 100.0
    
    idx = int(np.argmax(probs_percentage))
    genre = genre_mapping.get(idx, "Unknown")
    conf = float(probs_percentage[idx])
    
    all_probs = {genre_mapping[i]: float(probs_percentage[i]) for i in range(len(genre_mapping))}
    
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
    # Set dark background style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 2.3))
    
    # Set transparent/dark backgrounds
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Waveshow with a premium glowing blue/cyan color
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#4f7cff', alpha=0.85)
    
    # Polish labels and grid
    ax.set_title("Audio Waveform", fontsize=10, fontweight='bold', color='#f5f7fb', pad=10)
    ax.set_xlabel("Time (s)", fontsize=8, color='#a0aec0')
    ax.set_ylabel("Amplitude", fontsize=8, color='#a0aec0')
    ax.tick_params(colors='#a0aec0', labelsize=8)
    
    # Hide top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#2d3748')
    ax.spines['left'].set_color('#2d3748')
    
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def plot_prob_bars(all_probs):
    plt.style.use('dark_background')
    genres, vals = zip(*sorted(all_probs.items(), key=lambda x: -x[1]))
    
    fig, ax = plt.subplots(figsize=(6, 2.7))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Create horizontal or vertical bar plot with gradient-like colors
    colors = ['#6dd6ff' if i == 0 else '#4f7cff' for i in range(len(genres))]
    
    bars = ax.bar(genres, vals, color=colors, edgecolor='none', width=0.6, alpha=0.9)
    
    # Style grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.15, color='#e2e8f0')
    ax.set_axisbelow(True)
    
    # Polish labels and axes
    ax.set_ylabel("Probability (%)", fontsize=8, color='#a0aec0')
    ax.set_ylim(0, 105)
    ax.set_title("Class Probabilities", fontsize=10, fontweight='bold', color='#f5f7fb', pad=10)
    ax.tick_params(colors='#a0aec0', labelsize=8)
    plt.xticks(rotation=35, ha='right')
    
    # Clean up spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#2d3748')
    ax.spines['left'].set_color('#2d3748')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 5:
            ax.annotate(f'{height:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, color='#eef0f3', fontweight='semibold')
            
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


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
    # Load all models and scaler
    models, scaler, genre_mapping = load_all_models_and_scaler()
    
    # Sidebar style and header
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
          <h2 style="font-weight: 800; color: #f5f7fb; font-size: 1.6rem; letter-spacing: 0.5px;">Control Panel</h2>
          <p style="color: #a0aec0; font-size: 0.85rem;">Select & compare classification models</p>
        </div>
        <hr style="border-color: rgba(255,255,255,0.08); margin-bottom: 25px;" />
        """,
        unsafe_allow_html=True
    )
    
    # Model Selection
    st.sidebar.markdown("<h3 style='font-size:1.1rem; color:#f5f7fb; font-weight:700;'>Classifier Model</h3>", unsafe_allow_html=True)
    model_options = list(models.keys())
    
    if not model_options:
        st.error("No models could be loaded. Please verify model files exist in the project directory.")
        return
        
    selected_model_name = st.sidebar.selectbox(
        "Select Active Model",
        options=model_options,
        index=0 if "Random Forest (Recommended)" in model_options else 0,
        label_visibility="collapsed"
    )
    
    # Model info card in sidebar
    model_details = {
        "Random Forest (Recommended)": {
            "Accuracy": "78%",
            "F1-Score": "78%",
            "Type": "Ensemble Forest",
            "Pros": "Superior overall performance. Highly robust across most genres, handles complex overlaps gracefully, and has zero deep learning compute overhead at inference."
        },
        "Support Vector Machine (SVM)": {
            "Accuracy": "74%",
            "F1-Score": "74%",
            "Type": "Kernel SVM (RBF)",
            "Pros": "Excellent at establishing margins in high-dimensional feature spaces. Strong at separating close harmonic genres (e.g. Blues vs. Jazz)."
        },
        "Logistic Regression": {
            "Accuracy": "59%",
            "F1-Score": "58%",
            "Type": "Linear Baseline",
            "Pros": "Extremely fast and lightweight. Serves as a great simple baseline, though lacks complexity for rich audio details."
        },
        "Convolutional Neural Network (CNN)": {
            "Accuracy": "73%",
            "F1-Score": "74%",
            "Type": "1D CNN Deep Model",
            "Pros": "Highly expressive deep neural network architecture. Strong feature extraction, but slightly overfits GTZAN segment-level representations."
        }
    }
    
    details = model_details.get(selected_model_name, {})
    if details:
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); padding: 15px; border-radius: 12px; margin-top: 15px;">
              <div style="font-size:0.75rem; font-weight:700; color:#4f7cff; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;">Model Spec</div>
              <div style="font-size:0.95rem; font-weight:700; color:#eef0f3; margin-bottom:12px;">{details['Type']}</div>
              <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                <div>
                  <div style="font-size:0.7rem; color:#a0aec0; text-transform:uppercase;">Accuracy</div>
                  <div style="font-size:1.15rem; font-weight:800; color:#6dd6ff;">{details['Accuracy']}</div>
                </div>
                <div>
                  <div style="font-size:0.7rem; color:#a0aec0; text-transform:uppercase;">F1-Score</div>
                  <div style="font-size:1.15rem; font-weight:800; color:#6dd6ff;">{details['F1-Score']}</div>
                </div>
              </div>
              <div style="font-size:0.7rem; color:#a0aec0; text-transform:uppercase; margin-bottom:4px;">Characteristics</div>
              <div style="font-size:0.82rem; color:#d1d5db; line-height:1.45;">{details['Pros']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<h1 class='title-text'>Music Genre Classification</h1>", unsafe_allow_html=True)
    st.caption("Upload a .wav file. The app segments the audio and ensembles predictions across the track.")

    uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

    if uploaded_file is None:
        st.info("Select a .wav file to start.")
        return

    # Load audio and show player
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner(f"Extracting features & ensembling predictions using {selected_model_name}..."):
        # 1) audio -> y,sr
        y, sr = load_audio_to_temp(uploaded_file)
        # 2) features (segmented into 10 chunks of 3-seconds each)
        feats_segmented = extract_features_segmented(y, sr)
        # 3) predict using selected model
        try:
            genre, confidence, all_probs = predict_genre_from_segments(feats_segmented, selected_model_name)
            push_history(genre, confidence)
        except Exception as e:
            st.error(f"Inference Error: {e}")
            return

    # UI columns
    left, right = st.columns([0.55, 0.45], vertical_alignment="center")

    # Left: waveform + bars
    with left:
        st.markdown("#### Audio analysis")
        plot_waveform(y, sr)
        plot_prob_bars(all_probs)

        with st.expander("What the model uses"):
            st.write(
                "- Features: 13 MFCCs, 12 Chroma STFTs, Spectral Centroid, Spectral Rolloff, Zero-Crossing Rate\n"
                "- Standard scaling applied before inference\n"
                "- Ensembling: Predicts on ten 3-second segments and averages predicted probabilities."
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

        model_chip_name = selected_model_name.split(" (")[0]
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
                <div class="chip">{model_chip_name}</div>
                <div class="chip">MFCC • Chroma • ZCR</div>
                <div class="chip">Segment Ensembled</div>
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
