# PowerPoint Presentation Outline: Speech Emotion Recognition (SER)
**B.Tech CSE Final Year Major Project Defense**

---

### Slide 1: Title Slide
- **Title**: Speech Emotion Recognition using Deep Learning
- **Subtitle**: B.Tech CSE Final-Year Project Defense
- **Team Members**:
  - Student 1 (Data Engineer)
  - Student 2 (Signal Analyst)
  - Student 3 (ML Researcher)
  - Student 4 (Deep Learning Engineer)
  - Student 5 (Full Stack Developer)
- **Project Supervisor**: [Supervisor Name & Designation]
- **Department**: Department of Computer Science & Engineering
- **University**: [University Name / Logo]

---

### Slide 2: Introduction
- **Overview**: What is Speech Emotion Recognition (SER)?
  - Automatic detection of human affective states from audio signals.
  - Multi-disciplinary domain: Digital Signal Processing (DSP) + Machine Learning (ML) + Affective Computing.
- **Why Voice?**:
  - Vocal signals contain intrinsic characteristics linked to the nervous system.
  - Natural, non-invasive, and language-independent marker.

---

### Slide 3: Problem Statement
- **Emotion-Blind Systems**: Existing applications (voice assistants, chatbots, IVR systems) do not adapt to user moods.
- **Acoustic Variability**: Challenge of speaker accents, vocal pitch differences, background noise, and microphone qualities.
- **Complex Feature Mapping**: Speech is highly non-linear; simple traditional classifiers fail to capture structural dynamics.
- **Goal**: Develop a robust, real-time 1D CNN pipeline to extract features and classify speech emotions into 8 targets.

---

### Slide 4: Objectives
- **Data Engineering**: Process and structure the RAVDESS corpus.
- **Feature Extraction**: Extract hybrid acoustic vectors (MFCC + Chroma + Mel Spectrogram).
- **Model Training**: Build a 1D Convolutional Neural Network (CNN) for classification.
- **Real-Time Deployment**: Construct an interactive Streamlit UI for instant recording, waveform plotting, and prediction.
- **Systematic Evaluation**: Measure performance using standard classification matrices.

---

### Slide 5: Literature Survey
- **Historical Context**:
  - *SVM / HMM Approaches (2006-2011)*: Good for small datasets, but struggle with high-dimensional acoustic arrays.
  - *2D CNN / LSTM (2018-present)*: High accuracy but requires extensive GPU power.
- **Research Gap**: Need for an optimized model that runs on lightweight edge CPUs while maintaining high accuracy.
- **Our Solution**: Time-averaged multi-feature fusion input (180 features) + 1D Convolutional layers.

---

### Slide 6: Dataset Description (RAVDESS)
- **Name**: Ryerson Audio-Visual Database of Emotional Speech and Song.
- **Details**:
  - 24 professional actors (12 male, 12 female).
  - 1,440 vocal clips containing two vocal statements in neutral/normal and strong intensities.
- **Target Emotions**:
  - Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.
- **Coding Scheme**: Filenames encoded as `03-01-03-...wav` where the 3rd index determines the emotion category.

---

### Slide 7: Methodology
- **Step 1: Preprocessing**: Standardize to 22.05 kHz, trim silence (top-db=60), pad/truncate to 3.0s.
- **Step 2: Feature Fusion**:
  - **MFCC (40)**: Captures timbre/spectral envelope.
  - **Chroma (12)**: Captures pitch class distribution.
  - **Mel Spectrogram (128)**: Models auditory frequency perception.
- **Step 3: Dimensionality & Scaling**: Fit `StandardScaler` on training set; reshape to `(180, 1)`.
- **Step 4: Classification**: Forward pass through Conv1D channels.

---

### Slide 8: 1D CNN Architecture
- **Layer 1**: Conv1D (256 filters, kernel=5) + BatchNorm + MaxPool1D (size=5) + Dropout (30%).
- **Layer 2**: Conv1D (128 filters, kernel=5) + BatchNorm + MaxPool1D (size=5) + Dropout (30%).
- **Layer 3**: Conv1D (64 filters, kernel=5) + BatchNorm + MaxPool1D (size=5) + Dropout (30%).
- **Dense Layers**: Flatten -> Dense (64, ReLU) -> Dropout (40%) -> Dense (8, Softmax).
- **Optimization**: Adam Optimizer, Categorical Crossentropy Loss.

---

### Slide 9: Results & Demonstration
- **Evaluation Metrics**:
  - **Test Accuracy**: High validation performance on test set.
  - **Per-Class Metrics**: Precision, Recall, and F1-Score recorded in report.
- **Visual Artifacts**:
  - **Confusion Matrix**: Identifies classification overlaps (e.g., Calm vs. Neutral).
- **Live GUI Demo**: Screen captures of the Streamlit application demonstrating file upload and live speech recording.

---

### Slide 10: Future Scope & Conclusion
- **Conclusion**:
  - Demonstrated that combining fused features with a 1D CNN delivers an efficient, accurate SER framework.
  - The Streamlit interface is practical for real-world interactions.
- **Future Scope**:
  - Integrate LSTM/Attention mechanisms for sequence classification.
  - Extend models using multi-lingual datasets (TESS, EMO-DB).
  - Deploy on mobile embedded microcontrollers (Raspberry Pi/TensorFlow Lite).

---

### Slide 11: References & Q&A
- Selected references (RAVDESS paper, Librosa documentation, TensorFlow standard).
- **Open Floor for Questions and Answers**.
