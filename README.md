# Speech Emotion Recognition using Deep Learning (SER-DL)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An AI-powered speech recognition system that detects human emotional states from voice/audio recordings. The system uses audio signal processing (via Librosa) to extract acoustic parameters (MFCCs, Chroma, Mel Spectrograms) and classifies them using a 1D Convolutional Neural Network (CNN) into 8 distinct emotional categories.

Built as a B.Tech Computer Science and Engineering final-year major project.

---

## 🎙️ Target Emotions
- **Neutral** (😐)
- **Calm** (😌)
- **Happy** (😊)
- **Sad** (😢)
- **Angry** (😠)
- **Fearful** (😨)
- **Disgust** (🤢)
- **Surprised** (😲)

---

## 🛠️ Technology Stack
- **Core Programming**: Python 3.8+
- **Signal Processing**: Librosa, SoundFile, Wave
- **Scientific Computing**: NumPy, Pandas, Scikit-learn
- **Deep Learning**: TensorFlow, Keras
- **Visualization**: Matplotlib, Seaborn
- **User Interface**: Streamlit
- **Model Serialization**: Joblib

---

## 📂 Project Directory Structure
```text
Speech-Emotion-Recognition/
├── dataset/                    # Dataset scripts and audio storage
│   ├── download_ravdess.py     # Script to download official RAVDESS Speech corpus
│   ├── generate_mock_data.py   # Script to generate synthetic WAV files for local test runs
│   └── raw/                    # Audio recordings folder (Actor_01 to Actor_24)
├── notebooks/                  # Jupyter notebooks for EDA and prototyping
│   └── exploratory_data_analysis.ipynb
├── models/                     # Saved models and scaler assets
│   ├── cnn_model.keras         # Saved TensorFlow/Keras CNN model
│   ├── scaler.joblib           # StandardScaler for features
│   └── encoder.joblib          # LabelEncoder for target emotions
├── app/                        # Streamlit web application
│   └── app.py                  # Streamlit dashboard script
├── reports/                    # B.Tech project report & evaluation charts
│   ├── project_report.md       # Full B.Tech final-year project report (IEEE format)
│   ├── confusion_matrix.png    # Visual evaluation matrix
│   └── classification_report.txt
├── ppt/                        # Academic presentation files
│   └── presentation_outline.md # PowerPoint outline slide-by-slide
├── requirements.txt            # Python dependencies
├── README.md                   # Repository guide
└── main.py                     # Training & evaluation pipeline orchestrator
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

### 2. Install Dependencies
Set up a Python virtual environment and install the required modules:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Obtain Dataset
You can choose either the official dataset or a quick test configuration:

*   **Option A: Quick Testing (Recommended for dry runs)**
    Generate synthetic, small WAV files mimicking RAVDESS to run the pipeline instantly:
    ```bash
    python dataset/generate_mock_data.py
    ```
*   **Option B: Official RAVDESS Dataset**
    Download the complete RAVDESS Speech zip (approx 200MB) and extract:
    ```bash
    python dataset/download_ravdess.py
    ```

---

## 🚀 Execution Guide

### Step 1: Feature Extraction and Model Training
Run `main.py` to extract features from audio files, scale them, partition the dataset (80-20 train-test split), and train the 1D CNN classifier:
```bash
python main.py
```
This script will:
- Extract 180 acoustic parameters per file (MFCC, Chroma, Mel Spectrogram).
- Save a CSV cache at `dataset/extracted_features.csv`.
- Train the CNN and save model components to the `models/` folder.
- Output metrics and save a confusion matrix to `reports/confusion_matrix.png`.

### Step 2: Launch the Web Application
Run the Streamlit app to launch the interactive browser interface:
```bash
streamlit run app/app.py
```
The app will open at `http://localhost:8501`. You can upload WAV files or record directly from your microphone to view real-time emotion predictions.

---

## 📊 Model Architecture (1D CNN)
The classification model processes a 180-dimensional feature vector reshaped to `(180, 1)`:
1.  **Conv1D Layer**: 256 filters, kernel size 5, stride 1, ReLU activation.
2.  **Batch Normalization & MaxPooling1D**: Rescaling and pool size 5, stride 2.
3.  **Dropout**: 30% dropout to prevent overfitting.
4.  **Conv1D Layer**: 128 filters, kernel size 5, ReLU activation.
5.  **Batch Normalization & MaxPooling1D**: Pool size 5, stride 2.
6.  **Dropout**: 30% dropout.
7.  **Conv1D Layer**: 64 filters, kernel size 5, ReLU activation.
8.  **Batch Normalization & MaxPooling1D**: Pool size 5, stride 2.
9.  **Dropout**: 30% dropout.
10. **Flatten & Fully Connected Dense Layer**: 64 neurons, ReLU, 40% dropout.
11. **Output Layer**: 8 neurons, Softmax activation (corresponds to emotions).

---

## 👨‍🎓 B.Tech Project Group Details
This project was developed by a team of 5 students :
- **Data Engineer**: Dataset gathering and pipeline caching.
- **Signal Analyst**: Audio digital signal processing (DSP) and extraction.
- **ML Researcher**: Exploratory data analysis and model design.
- **Deep Learning Engineer**: Model compiling, training loops, and validation.
- **Full Stack Developer**: Streamlit dashboard construction and microphone API.

---

## 📝 License
This project is licensed under the MIT License.
