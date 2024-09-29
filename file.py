# Required library installation (run in terminal before running code)
# pip install librosa soundfile numpy sklearn matplotlib seaborn

# Importing required libraries
import librosa
import soundfile as sf
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to extract audio features
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    result = np.array([])
    
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    
    if chroma:
        stft = np.abs(librosa.stft(audio_data))
        chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_feature))
    
    if mel:
        mel_feature = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_feature))
    
    return result

# Emotion labels in dataset
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
    for file in glob.glob("dataset/**/*.wav", recursive=True):
        print(f"Processing {file}")  # Add this line
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion in observed_emotions:
            features = extract_features(file)
            x.append(features)
            y.append(emotion)
    print(f"Loaded {len(x)} samples.")  # Print number of samples loaded
    return train_test_split(np.array(x), y, test_size=test_size, random_state=42)



# Load dataset and split it
x_train, x_test, y_train, y_test = load_data(test_size=0.2)

# Display dataset shape
print(f"Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}")

# Initialize MLP Classifier with tuned hyperparameters
model = MLPClassifier(
    alpha=0.001, 
    batch_size=128, 
    epsilon=1e-08, 
    hidden_layer_sizes=(300, 150), 
    learning_rate='adaptive', 
    max_iter=600, 
    activation='relu',  # Changed to relu for better learning
    solver='adam'
)

# Train the model
model.fit(x_train, y_train)

# Predict on test data
y_pred = model.predict(x_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=observed_emotions, yticklabels=observed_emotions)
plt.title('Confusion Matrix of Emotion Recognition')
plt.xlabel('Predicted Emotions')
plt.ylabel('Actual Emotions')
plt.show()

# Plot accuracy trend across iterations (optional)
plt.plot(model.loss_curve_)
plt.title('Training Loss Across Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
