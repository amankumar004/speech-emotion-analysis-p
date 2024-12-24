# accuracy - 86%


# Required libraries (Make sure to install before running)
# pip install librosa soundfile numpy sklearn matplotlib seaborn tensorflow keras

import librosa
import soundfile as sf
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dense, Flatten, MaxPooling1D
# from tensorflow.keras.utils import plot_model

# Function to extract audio features (similar to before)
def extract_features(file_path, mfcc=True, chroma=True, mel=True, delta=True, spectral_contrast=True):
    audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    result = np.array([])

    if mfcc:
        # Use 60 MFCCs instead of 40
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=60).T, axis=0)
        result = np.hstack((result, mfccs))

        # Delta and Delta-Delta (to capture changes in MFCCs over time)
        delta_mfccs = np.mean(librosa.feature.delta(mfccs).T, axis=0)
        delta2_mfccs = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)
        result = np.hstack((result, delta_mfccs, delta2_mfccs))

    if chroma:
        stft = np.abs(librosa.stft(audio_data))
        chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_feature))

    if mel:
        mel_feature = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_feature))

    if spectral_contrast:
        # Add spectral contrast features for timbre information
        spec_contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, spec_contrast))

    return result

# Emotion labels in the dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Function to load the dataset
def load_data(test_size=0.25):
    x, y = [], []
    # for file in glob.glob("dataset/*/.wav", recursive=True):
    for file in glob.glob("dataset/**/*.wav", recursive=True):
        print(f"Processing {file}")
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion in observed_emotions:
            features = extract_features(file)
            x.append(features)
            y.append(emotion)
    print(f"Loaded {len(x)} samples.")
    return train_test_split(np.array(x), y, test_size=test_size, random_state=42)

# Load dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.2)

# Convert y_train, y_test into numerical labels (necessary for VDCNN and MLP)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Reshape input for CNN (VDCNN expects 3D input: samples, timesteps, features)
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# Define the VDCNN Model
def build_vdcnn(input_shape):
    model = Sequential()

    # Convolutional Block 1
    model.add(Conv1D(64, kernel_size=3, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2))

    # Convolutional Block 2
    model.add(Conv1D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2))

    # Convolutional Block 3
    model.add(Conv1D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2))

    # Flatten for MLP input
    model.add(Flatten())
    return model


# # Build VDCNN model
vdcnn = build_vdcnn((x_train.shape[1], 1))

# # Save model architecture as an image
# plot_model(vdcnn, to_file='vdcnn_model_architecture.png', show_shapes=True, show_layer_names=True)

# Extract features from VDCNN
train_features = vdcnn.predict(x_train)
test_features = vdcnn.predict(x_test)

# MLP Classifier for final classification
# mlp = MLPClassifier(
#     alpha=0.0005,
#     batch_size=128,
#     epsilon=1e-08,
#     hidden_layer_sizes=(400, 300),
#     learning_rate='adaptive',
#     max_iter=800,
#     activation='relu',
#     solver='adam'
# )



# new mlp
mlp = MLPClassifier(
    alpha=0.0005,
    batch_size=128,
    epsilon=1e-08,
    hidden_layer_sizes=(500, 400, 300),  # Increased depth and size
    learning_rate='adaptive',
    learning_rate_init=0.0001,           # Lower initial learning rate
    max_iter=1000,                       # Increased max iterations
    activation='relu',                   # Keep relu, or try tanh/logistic
    solver='adam'
)

# Train MLP on extracted features
mlp.fit(train_features, y_train)

# Predict on test data
y_pred = mlp.predict(test_features)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=observed_emotions, yticklabels=observed_emotions)
plt.title('Confusion Matrix of Emotion Recognition')
plt.xlabel('Predicted Emotions')
plt.ylabel('Actual Emotions')
plt.show()

# Loss curve for MLP (optional)
plt.plot(mlp.loss_curve_)
plt.title('Training Loss Across Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
