# SPEECH EMOTION RECOGNITION USING DEEP LEARNING
**A B.Tech Computer Science & Engineering Major Project Report**

---

## ABSTRACT
Speech Emotion Recognition (SER) is a rapidly growing field of research in human-computer interaction (HCI) and affective computing. Detecting human emotions from voice recordings enables computers to respond dynamically and empathetically to users. This project presents the design and implementation of a deep learning-based SER system utilizing Mel-Frequency Cepstral Coefficients (MFCC), Chroma features, and Mel-scale spectrograms for acoustic feature extraction, combined with a 1D Convolutional Neural Network (CNN) classifier. The system is trained and evaluated on the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset. The model successfully classifies speech into eight distinct emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised. The system features a responsive, dark-themed Streamlit web application that allows real-time voice recording, spectrogram visualization, and classification probability mapping. The project achieves robust performance, serving as an effective model for B.Tech final-year CSE students over two academic semesters.

---

## CHAPTER 1: INTRODUCTION

### 1.1 Project Overview
Human emotion expression is multi-modal, spanning facial expressions, gestures, text, and vocal acoustics. While facial expression recognition has progressed significantly, vocal acoustics contain rich, language-independent features that directly correlate with the autonomic nervous system's response (e.g., changes in respiration and vocal fold tension due to arousal). Speech Emotion Recognition (SER) aims to extract these acoustic features and model the underlying emotions.

### 1.2 Problem Statement
Traditional human-computer interfaces are emotion-blind, treating all user inputs with uniform logic. This leads to rigid and frustrating interactions in customer service chatbots, virtual assistants (Siri, Alexa), and healthcare systems. In addition, speech signals are highly variable, containing differences in speaker accents, pitch, background noise, and recording equipment. The challenge is to extract robust, speaker-independent acoustic features and build a deep learning architecture that can accurately map these features to emotional states (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised) in real time.

### 1.3 Project Objectives
The main objectives of this project are:
1. **Acoustic Preprocessing**: Develop a signal processing pipeline to trim silence, normalize amplitude, and process vocal recordings.
2. **Multi-Feature Fusion**: Extract and fuse Mel-Frequency Cepstral Coefficients (MFCC), Chroma STFT, and Mel Spectrogram features to represent pitch, timbre, and frequency distributions.
3. **Deep Learning Classification**: Design and train a multi-layered 1D Convolutional Neural Network (CNN) that excels at extracting spatial patterns from serialized acoustic features.
4. **Interactive GUI**: Build a responsive Streamlit web application supporting file uploads and direct microphone recording for real-time inference.
5. **System Evaluation**: Validate the system using standard metrics, including Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

### 1.4 Scope and Applications
The scope of this project is limited to processing English-language audio signals from the RAVDESS corpus and user-supplied recordings. Applications of this technology include:
- **Healthcare**: Monitoring depressive or anxious states in patients through telemedicine interfaces.
- **Customer Service**: Route customer calls to specialized agents by identifying angry or frustrated callers.
- **Smart Vehicles**: Detect driver stress or sleepiness through vocal queries to adjust cabin environmental parameters.
- **Education**: Adapt e-learning course pace based on the student's vocal engagement or confusion levels.

---

## CHAPTER 2: LITERATURE SURVEY

| Author & Year | Methodology / Model | Features Used | Dataset | Key Outcomes / Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Ververidis et al. (2006)** | Support Vector Machines (SVM) & GMM | Pitch, Formants, Energy | Danish Emotional Speech | SVM achieved 50% accuracy on 5 classes. Feature selection was highly manual and lacked scaling. |
| **El Ayadi et al. (2011)** | Hidden Markov Models (HMM) | MFCC, Pitch | Berlin Database (EMO-DB) | HMM modeled temporal transitions well but struggled with short utterances and overlapping emotional classes. |
| **Koolagudi et al. (2012)** | Artificial Neural Network (ANN) | Pitch, Formant frequencies | IITKGP-SEHAC | Sound quality differences affected results. Traditional ANNs suffered from spatial feature loss. |
| **Tzirakis et al. (2018)** | 2D CNN + LSTM | Raw Waveform, Log-Mel | RECOLA Dataset | End-to-end learning achieved high temporal accuracy, but requires immense computational resources and GPUs. |
| **Our Proposed System (2026)** | **1D CNN + Dense Classifier** | **MFCC, Chroma, Mel Spectrogram** | **RAVDESS** | **Optimal performance-computation trade-off, fast feature processing, runs on standard CPUs, Streamlit UI.** |

---

## CHAPTER 3: SYSTEM DESIGN & ARCHITECTURE

### 3.1 System Flow Architecture
The following text-based block diagram illustrates the end-to-end data flow:

```text
+-----------------------+      +-----------------------+
|  User Audio Input     | ---> |  Signal Preprocessing |
| (WAV Upload / Mic)    |      | (Silence Trim/Normal) |
+-----------------------+      +-----------------------+
                                           |
                                           v
+-----------------------+      +-----------------------+
|   Feature Fusion      | <--- |   Feature Extraction  |
| (MFCC+Chroma+Mel spec)|      | (Librosa Analysis)    |
+-----------------------+      +-----------------------+
           |
           v
+-----------------------+      +-----------------------+
| Standard Scaling      | ---> |  1D CNN Classifier    |
| (scaler.joblib)       |      | (Conv1D/BatchNorm/    |
+-----------------------+      |  Pooling/Dropout)     |
                               +-----------------------+
                                           |
                                           v
+-----------------------+      +-----------------------+
|   Label Decoder       | ---> | Streamlit Dashboard   |
| (encoder.joblib)      |      | (Emoji/Confidence     |
+-----------------------+      |  Bar Chart Display)   |
                               +-----------------------+
```

### 3.2 1D CNN Model Layer Design
The feature vector length is 180 (40 MFCCs + 12 Chroma + 128 Mel Spectrograms). The shape input to the 1D CNN is `(180, 1)`.

```text
    [Input Shape: (180, 1)]
               │
               ▼
┌──────────────────────────────┐
│  Conv1D (256 filters, k=5)   │  --> Spatial feature extraction
├──────────────────────────────┤
│  Batch Normalization         │  --> Faster convergence, regularizer
├──────────────────────────────┤
│  MaxPooling1D (pool=5, s=2)   │  --> Dim reduction (180 -> 90)
├──────────────────────────────┤
│  Dropout (30%)               │  --> Prevents overfitting
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Conv1D (128 filters, k=5)   │
├──────────────────────────────┤
│  Batch Normalization         │
├──────────────────────────────┤
│  MaxPooling1D (pool=5, s=2)   │  --> Dim reduction (90 -> 45)
├──────────────────────────────┤
│  Dropout (30%)               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Conv1D (64 filters, k=5)    │
├──────────────────────────────┤
│  Batch Normalization         │
├──────────────────────────────┤
│  MaxPooling1D (pool=5, s=2)   │  --> Dim reduction (45 -> 22)
├──────────────────────────────┤
│  Dropout (30%)               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Flatten                     │  --> Reshapes 22 x 64 -> 1408 vector
├──────────────────────────────┤
│  Dense (64, ReLU)            │  --> Fully connected representation
├──────────────────────────────┤
│  Dropout (40%)               │
├──────────────────────────────┤
│  Dense (8, Softmax)          │  --> Emotion probabilities
└──────────────────────────────┘
```

---

## CHAPTER 4: IMPLEMENTATION METHODOLOGY

### 4.1 Audio Preprocessing
1. **Sampling Rate Standardization**: Resample all recordings to 22,050 Hz using Librosa's Kaiser Fast algorithm.
2. **Silence Removal**: Apply `librosa.effects.trim` with a top-decibel threshold of 60dB to remove leading and trailing non-speech segments.
3. **Fixed Duration Windowing**: Truncate/pad recordings to a fixed length of 3.0 seconds (66,150 samples) to ensure input uniformity.

### 4.2 Feature Extraction
- **Mel-Frequency Cepstral Coefficients (MFCCs)**:
  Mathematically, MFCCs are computed by taking the Fourier transform of the signal, mapping the power spectrum to the Mel scale (using triangular bandpass filters), taking the log, and applying the Discrete Cosine Transform (DCT):
  \[ c(n) = \sum_{m=0}^{M-1} \log(S(m)) \cos\left( \frac{\pi n (m + 0.5)}{M} \right) \]
  We extract 40 MFCC coefficients.
- **Chroma STFT**: Computes the energy distribution across 12 pitch classes (semitones) to represent musical octave structures.
- **Mel Spectrogram**: Maps the signal energy to the Mel-frequency scale (128 frequency bands), aligning computer perception with human auditory acoustics.

All features are averaged along the frame axis to obtain a single time-independent vector:
\[ \mathbf{v}_{\text{fuse}} = \left[ \bar{\mathbf{v}}_{\text{mfcc}} \; \bar{\mathbf{v}}_{\text{chroma}} \; \bar{\mathbf{v}}_{\text{mel}} \right] \in \mathbb{R}^{180} \]

### 4.3 Semester-wise Development Schedule (5 Students, 2 Semesters)
To make the project realistic and manageable, the team roles and tasks are structured as follows:

```text
Semester 1: Foundation & Pipeline
┌──────────────────────────────┬───────────────────────────────────────────┐
│ Student Role                 │ Primary Responsibilities                  │
├──────────────────────────────┼───────────────────────────────────────────┤
│ Student 1: Data Engineer     │ Dataset gathering (RAVDESS), cleaning,    │
│                              │ structure validation, and backup.         │
├──────────────────────────────┼───────────────────────────────────────────┤
│ Student 2: Signal Analyst    │ Preprocessing pipelines, trimming,        │
│                              │ resampling, and audio feature extraction. │
├──────────────────────────────┼───────────────────────────────────────────┤
│ Student 3: ML Researcher     │ Exploratory Data Analysis, feature scaling│
│                              │ testing, and model architecture research. │
└──────────────────────────────┴───────────────────────────────────────────┘

Semester 2: Training, Optimization, & Web GUI
┌──────────────────────────────┬───────────────────────────────────────────┐
│ Student Role                 │ Primary Responsibilities                  │
├──────────────────────────────┼───────────────────────────────────────────┤
│ Student 4: Deep Learning Eng │ 1D CNN coding, training loops,            │
│                              │ hyperparameter tuning, checkpoints, loss. │
├──────────────────────────────┼───────────────────────────────────────────┤
│ Student 5: Full Stack Dev    │ Streamlit application development, live   │
│                              │ mic recording integration, plot renders.  │
└──────────────────────────────┴───────────────────────────────────────────┘
```

---

## CHAPTER 5: RESULTS AND ANALYSIS

### 5.1 Experimental Metrics
The model is evaluated on five key performance indicators:
1. **Accuracy**: Total percentage of correct predictions.
2. **Precision**: The ratio of true positives to all predicted positives for a class:
   \[ \text{Precision} = \frac{TP}{TP + FP} \]
3. **Recall**: The ratio of true positives to all actual positives:
   \[ \text{Recall} = \frac{TP}{TP + FN} \]
4. **F1-Score**: Harmonic mean of precision and recall:
   \[ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]
5. **Confusion Matrix**: Visual representation of predicted vs. true labels.

### 5.2 Confusion Matrix Discussion
The confusion matrix (`reports/confusion_matrix.png`) indicates:
- **Calm** and **Neutral** emotions show minor overlap due to shared low-arousal features (slow tempo, lower pitch variance).
- **Angry** and **Surprised** emotions share high-arousal characteristics, but the combination of Chroma features and Mel Spectrogram distribution helps the 1D CNN distinguish them.
- Adding Dropout layers prevents overfitting on actor-specific vocal features, improving test accuracy.

---

## CHAPTER 6: CONCLUSION & FUTURE SCOPE

### 6.1 Conclusion
We designed and implemented a Speech Emotion Recognition system using deep learning. Mel-scale spectral features and 1D CNNs prove to be a highly effective combination for classifying vocal expressions. By structuring the dataset and preprocessing pipeline, we achieved stable convergence. The Streamlit web dashboard demonstrates interactive utility for human-computer interaction research.

### 6.2 Future Scope
1. **Multi-dataset Fusion**: Incorporate SAVEE, TESS, and EMO-DB datasets to evaluate model generalization across varying accents and cultures.
2. **Bi-Directional LSTM Integration**: Fuse Conv1D layers with Bi-LSTM cells to model long-term temporal dependencies in spontaneous speech.
3. **Noise Robustness**: Incorporate data augmentation (adding white noise, speed perturbation) to handle varying environments during real-world deployments.

---

## REFERENCES
1. Librosa: McFee, B., Raffel, C., Liang, D., Ellis, D.P., McVicar, M., Battenberg, E. and Nietschenwitsch, O., 2015. librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in science conference.
2. RAVDESS: Livingstone, S.R. and Russ, Z.P., 2018. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLOS ONE 13(5).
3. TensorFlow: Abadi, M. et al., 2016. TensorFlow: A system for large-scale machine learning. 12th USENIX Symposium on Operating Systems Design and Implementation.
4. IEEE Standard for Software Test Documentation, IEEE Std 829-2008.
5. Emotion Recognition in Speech: El Ayadi, M., Kamel, M.S. and Karray, F., 2011. Survey on speech emotion recognition: Features, classification schemes, and databases. Pattern Recognition, 44(3), pp.572-587.
