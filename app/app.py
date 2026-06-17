import os
import joblib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

# Try to import TensorFlow (fails gracefully on CPU AVX limits)
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except Exception:
    pass

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern dark theme and layout
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .stButton>button {
        background-color: #6366f1;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
    }
    .title-text {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #9ca3af;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
    }
    .emotion-highlight {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f43f5e;
    }
</style>
""", unsafe_allow_html=True)

# Emotion display helper (labels to emoji/colors)
EMOTION_META = {
    'neutral': {'emoji': '😐', 'color': '#9ca3af'},
    'calm': {'emoji': '😌', 'color': '#10b981'},
    'happy': {'emoji': '😊', 'color': '#fbbf24'},
    'sad': {'emoji': '😢', 'color': '#3b82f6'},
    'angry': {'emoji': '😠', 'color': '#ef4444'},
    'fearful': {'emoji': '😨', 'color': '#8b5cf6'},
    'disgust': {'emoji': '🤢', 'color': '#10b981'},
    'surprised': {'emoji': '😲', 'color': '#ec4899'}
}

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model.keras")
MLP_PATH = os.path.join(MODELS_DIR, "mlp_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.joblib")

@st.cache_resource
def load_ml_assets():
    """Loads the model, scaler, and label encoder once and caches them."""
    if not (os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
        return None, None, None, "none"
    try:
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        
        # 1. Attempt loading TensorFlow CNN first if available
        if HAS_TENSORFLOW and os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            return model, scaler, encoder, "cnn"
            
        # 2. Otherwise try loading MLP fallback model
        if os.path.exists(MLP_PATH):
            model = joblib.load(MLP_PATH)
            return model, scaler, encoder, "mlp"
            
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        
    return None, None, None, "none"

def extract_audio_features(data, sample_rate):
    """Extracts MFCC, Chroma, and Mel spectrogram features from loaded audio data."""
    try:
        data, _ = librosa.effects.trim(data)
        
        # 1. MFCC
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # 2. Chroma STFT
        stft = np.abs(librosa.stft(data))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # 3. Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
        mel_mean = np.mean(mel.T, axis=0)
        
        # Concatenate into 180 length vector
        return np.hstack((mfccs_mean, chroma_mean, mel_mean))
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def main():
    # Sidebar
    st.sidebar.markdown("### 🎙️ SER Deep Learning System")
    st.sidebar.write("A deep neural network solution to detect emotions from human voice recordings.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Technology Stack")
    st.sidebar.code("• Python 3.8+\n• TensorFlow (CNN)\n• Scikit-Learn (MLP)\n• Librosa\n• Streamlit")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Reference")
    st.sidebar.write("**RAVDESS Speech Corpus**: 24 professional actors, 1,440 vocal clips, 8 emotions.")

    # Main Area
    st.markdown('<div class="title-text">Speech Emotion Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Deep Learning Final Year B.Tech CSE Project</div>', unsafe_allow_html=True)
    
    # Load assets
    model, scaler, encoder, model_type = load_ml_assets()
    
    if model is None:
        st.warning("⚠️ **Trained Model Assets Not Found!**")
        st.info("The neural network model, scaler, and label encoder must be trained first. Run the machine learning pipeline by executing `python main.py` in your terminal to train and save the model.")
        return
        
    st.sidebar.success(f"Loaded Model Type: **{model_type.upper()}**")
    if model_type == "mlp":
        st.sidebar.info("💡 Running Scikit-Learn MLP Fallback due to CPU AVX limits on this machine.")

    # App content layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🎵 Input Audio Clip")
        
        # File uploader or Audio Input recorder
        input_type = st.radio("Choose input method:", ("Upload WAV File", "Record Live Audio"))
        
        audio_file = None
        if input_type == "Upload WAV File":
            audio_file = st.file_uploader("Upload a speech WAV recording", type=["wav"])
        else:
            audio_file = st.audio_input("Record your voice")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        if audio_file is not None:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🔊 Audio Player")
            st.audio(audio_file)
            
            # Load raw data with librosa for visualizations
            try:
                # Need to read byte stream
                audio_bytes = audio_file.read()
                # Use librosa to load from bytes stream by saving temporarily or reading with soundfile
                import io
                import soundfile as sf
                
                audio_file.seek(0)
                data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                
                # If multi-channel, average to mono
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                # Visualizations
                st.subheader("📈 Signal Visualizations")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
                # Plot Waveform
                librosa.display.waveshow(data, sr=sample_rate, ax=ax1, color='#818cf8')
                ax1.set_title("Time Domain: Waveform", color='#e0e6ed')
                ax1.set_ylabel("Amplitude", color='#e0e6ed')
                ax1.set_xlabel("Time (s)", color='#e0e6ed')
                ax1.tick_params(colors='#e0e6ed')
                
                # Plot Spectrogram
                stft_data = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
                img = librosa.display.specshow(stft_data, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
                ax2.set_title("Frequency Domain: Mel-Frequency Spectrogram", color='#e0e6ed')
                ax2.set_ylabel("Frequency (Hz)", color='#e0e6ed')
                ax2.set_xlabel("Time (s)", color='#e0e6ed')
                ax2.tick_params(colors='#e0e6ed')
                fig.colorbar(img, ax=ax2, format="%+2.0f dB")
                
                # Style figure
                fig.patch.set_facecolor('#0e1117')
                ax1.set_facecolor('#1e293b')
                ax2.set_facecolor('#1e293b')
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error reading audio file: {e}")
                data = None
                
            st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        if audio_file is not None and data is not None:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🧠 Model Classification Results")
            
            with st.spinner("Analyzing speech patterns..."):
                # Extract features (librosa load requires same sampling rate as training)
                features = extract_audio_features(data, sample_rate)
                
                if features is not None:
                    # Apply scaler
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    
                    # Run inference based on loaded model type
                    if model_type == "cnn":
                        features_cnn = np.expand_dims(features_scaled, axis=2)
                        predictions = model.predict(features_cnn)[0]
                    else: # mlp
                        predictions = model.predict_proba(features_scaled)[0]
                    
                    # Get predicted emotion
                    pred_idx = np.argmax(predictions)
                    predicted_emotion = encoder.classes_[pred_idx]
                    prob = predictions[pred_idx]
                    
                    # Display top emotion
                    meta = EMOTION_META.get(predicted_emotion, {'emoji': '🎙️', 'color': '#6366f1'})
                    
                    st.markdown(
                        f"Primary Detected Emotion: <span class='emotion-highlight' style='color:{meta['color']}'>{meta['emoji']} {predicted_emotion.upper()}</span>", 
                        unsafe_allow_html=True
                    )
                    st.write(f"**Confidence Level**: `{prob * 100:.2f}%`")
                    
                    st.write("---")
                    
                    # Class probabilities distribution
                    st.write("📊 **Emotion Probability Distribution:**")
                    classes = encoder.classes_
                    for cls_name, pred_prob in zip(classes, predictions):
                        cls_meta = EMOTION_META.get(cls_name, {'emoji': '', 'color': '#6366f1'})
                        st.write(f"{cls_meta['emoji']} **{cls_name.capitalize()}**: `{pred_prob * 100:.1f}%`")
                        st.progress(float(pred_prob))
                        
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Please upload a voice recording or record audio on the left column to run the emotion recognition system.")

if __name__ == "__main__":
    main()
