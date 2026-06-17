import os
import glob
import joblib
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Disable TensorFlow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Try to import TensorFlow for CNN training
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except Exception as e:
    print("\n" + "="*80)
    print("WARNING: TensorFlow could not be loaded on this CPU.")
    print("Reason:", str(e).strip())
    print("\n[B.Tech CSE Project Context]:")
    print("- This is usually caused by a virtual machine CPU lacking AVX/AVX2 instruction sets.")
    print("- The full 1D CNN Deep Learning model code is fully implemented below.")
    print("- For local sandboxed execution, we will automatically fallback to training a")
    print("  Scikit-Learn MLPClassifier (Multi-Layer Perceptron Neural Network).")
    print("- This allows the pipeline, evaluation, and Streamlit app to run end-to-end successfully.")
    print("="*80 + "\n")
    
    from sklearn.neural_network import MLPClassifier

# Emotion mapping for RAVDESS dataset
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    """
    Extracts audio features (MFCC, Chroma, Mel Spectrogram) from a WAV file.
    Averages features over time to create a 1D feature vector.
    """
    try:
        # Load audio (duration truncated to 3 seconds, offset 0.5s to skip initial silence)
        data, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=3.0, offset=0.5)
        
        # Trim silent portions
        data, _ = librosa.effects.trim(data)
        
        # 1. MFCC (Mel-frequency cepstral coefficients) - 40 features
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # 2. Chroma STFT - 12 features
        stft = np.abs(librosa.stft(data))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # 3. Mel Spectrogram - 128 features
        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
        mel_mean = np.mean(mel.T, axis=0)
        
        # Concatenate features into a single 1D array of size 180 (40 + 12 + 128)
        feature_vector = np.hstack((mfccs_mean, chroma_mean, mel_mean))
        return feature_vector
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def build_dataset(raw_dir, cache_path):
    """
    Loads all WAV files, extracts features, and saves them to a CSV cache.
    If cache exists, loads pre-extracted features instead.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        df = pd.read_csv(cache_path)
        return df

    print("Extracting features from dataset (this might take a few minutes)...")
    file_pattern = os.path.join(raw_dir, "Actor_*", "*.wav")
    audio_files = glob.glob(file_pattern)
    
    if not audio_files:
        raise ValueError(f"No WAV files found in {raw_dir}. Please run dataset download/mock scripts first.")

    features = []
    labels = []
    filepaths = []
    
    # Process audio files with simple console progress feedback
    total_files = len(audio_files)
    print(f"Found {total_files} audio files. Starting processing...")
    
    for idx, file in enumerate(audio_files):
        if (idx + 1) % 150 == 0 or idx == 0 or idx == total_files - 1:
            print(f"Processing progress: {idx + 1}/{total_files} files...")
            
        filename = os.path.basename(file)
        # RAVDESS format: 03-01-06-02-01-01-12.wav (3rd index is emotion)
        parts = filename.split('-')
        if len(parts) >= 7:
            emotion_code = parts[2]
            emotion = EMOTIONS.get(emotion_code)
            if emotion:
                feat = extract_features(file)
                if feat is not None:
                    features.append(feat)
                    labels.append(emotion)
                    filepaths.append(file)
    
    # Create DataFrame
    feature_cols = [f"feat_{i}" for i in range(180)]
    df = pd.DataFrame(features, columns=feature_cols)
    df['emotion'] = labels
    df['file_path'] = filepaths
    
    # Save cache
    df.to_csv(cache_path, index=False)
    print(f"Saved extracted features ({len(df)} samples) to {cache_path}.")
    return df

def build_cnn_model(input_shape, num_classes):
    """
    Builds a 1D Convolutional Neural Network for speech emotion classification.
    Only called if TensorFlow is successfully imported.
    """
    model = Sequential([
        # First Conv block
        Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        Dropout(0.3),
        
        # Second Conv block
        Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        Dropout(0.3),
        
        # Third Conv block
        Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        Dropout(0.3),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "dataset", "raw")
    cache_path = os.path.join(base_dir, "dataset", "extracted_features.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Build/Load dataset
    try:
        df = build_dataset(raw_dir, cache_path)
    except ValueError as e:
        print(e)
        return

    # Extract X (features) and y (labels)
    feature_cols = [f"feat_{i}" for i in range(180)]
    X = df[feature_cols].values
    y = df['emotion'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, os.path.join(models_dir, "encoder.joblib"))
    
    # Stratified Train-Test Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    
    num_classes = len(np.unique(y_encoded))
    emotion_labels = label_encoder.classes_
    
    # 2. Train Model
    if HAS_TENSORFLOW:
        print("Training 1D Convolutional Neural Network (TensorFlow/Keras)...")
        # Convert labels to one-hot encoding
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
        
        # Reshape features for 1D CNN: (samples, steps, channels) -> (samples, 180, 1)
        X_train_cnn = np.expand_dims(X_train, axis=2)
        X_test_cnn = np.expand_dims(X_test, axis=2)
        
        # Build CNN
        model = build_cnn_model((X_train_cnn.shape[1], 1), num_classes)
        model.summary()
        
        # Callbacks
        checkpoint_path = os.path.join(models_dir, "cnn_model.keras")
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        
        # Fit model
        history = model.fit(
            X_train_cnn, y_train_cat,
            epochs=40,
            batch_size=32,
            validation_data=(X_test_cnn, y_test_cat),
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )
        
        # Load best weights and predict
        if os.path.exists(checkpoint_path):
            model = tf.keras.models.load_model(checkpoint_path)
            
        y_pred_probs = model.predict(X_test_cnn)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
        print(f"\nCNN Test Accuracy: {test_acc:.4f}")
        
    else:
        print("Training Multi-Layer Perceptron (MLP) Neural Network (Scikit-Learn Fallback)...")
        # We use a 3-layer neural network with 256, 128, and 64 hidden nodes
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=150,
            random_state=42,
            verbose=True
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Save model
        model_path = os.path.join(models_dir, "mlp_model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved trained MLP model to {model_path}")
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        test_acc = np.mean(y_pred == y_test)
        test_loss = 0.0 # Standard loss not printed for classification report
        print(f"\nMLP Test Accuracy: {test_acc:.4f}")

    # 3. Save Evaluation Reports
    report = classification_report(y_test, y_pred, target_names=emotion_labels)
    print("\nClassification Report:")
    print(report)
    
    # Save text report
    report_file_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_file_path, "w") as f:
        f.write("Speech Emotion Recognition - Classification Report\n")
        f.write("==================================================\n")
        f.write(f"Model Type: {'1D CNN (TensorFlow)' if HAS_TENSORFLOW else 'MLP Neural Network (Scikit-Learn Fallback)'}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    print(f"Saved classification report to {report_file_path}")

    # Generate and save Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title(f"Speech Emotion Recognition - Confusion Matrix\n(Model: {'1D CNN' if HAS_TENSORFLOW else 'MLP Neural Network'})")
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Actual Emotion')
    plt.tight_layout()
    
    cm_plot_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")

if __name__ == "__main__":
    main()
