import librosa
from moviepy.editor import AudioFileClip
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Flatten, MaxPooling1D, Dropout

# Load the trained MLP model
with open("mlp_model.pkl", "rb") as model_file:
    mlp_model = pickle.load(model_file)

# Define observed emotions
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define the VDCNN model (architecture must match the training code)
def build_vdcnn(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    return model

# Build the VDCNN model
vdcnn = build_vdcnn((180, 1))  # Adjust input shape if needed

def segment_audio(audio_path, segment_duration=2.5, sample_rate=22050):
    """
    Segment the audio into smaller chunks of specified duration.
    """
    try:
        audio_data, _ = librosa.load(audio_path, sr=sample_rate)
        num_segments = int(np.ceil(len(audio_data) / (segment_duration * sample_rate)))
        segments = [
            audio_data[i * int(segment_duration * sample_rate): (i + 1) * int(segment_duration * sample_rate)]
            for i in range(num_segments)
        ]
        return segments
    except Exception as e:
        print(f"Error segmenting audio: {e}")
        return []

def extract_audio_from_video(video_path, output_audio_path="extracted_audio.wav"):
    """
    Extract audio from a video file and save it as a .wav file.
    """
    try:
        print("Extracting audio from video...")
        audio_clip = AudioFileClip(video_path)
        audio_clip.write_audiofile(output_audio_path, fps=22050, codec="pcm_s16le")
        audio_clip.close()
        print(f"Audio extracted and saved at: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"Error during audio extraction: {e}")
        return None

def extract_features_from_segment(segment, sample_rate=22050, mfcc=True, chroma=True, mel=True, delta=True, spectral_contrast=True):
    """
    Extract features from a single audio segment.
    """
    try:
        result = np.array([])

        # MFCCs
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=60).T, axis=0)
            result = np.hstack((result, mfccs))
            if delta:
                delta_mfcc = np.mean(librosa.feature.delta(mfccs).T, axis=0)
                delta_delta_mfcc = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)
                result = np.hstack((result, delta_mfcc, delta_delta_mfcc))

        # Chroma
        if chroma:
            stft = np.abs(librosa.stft(segment))
            chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feature))

        # Mel Spectrogram
        if mel:
            mel_feature = np.mean(librosa.feature.melspectrogram(y=segment, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feature))

        # Spectral Contrast
        if spectral_contrast:
            spectral_contrast_feature = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sample_rate).T, axis=0)
            result = np.hstack((result, spectral_contrast_feature))

        return result
    except Exception as e:
        print(f"Error extracting features from segment: {e}")
        return None

def predict_emotions_over_time(audio_path, segment_duration=2.5):
    """
    Predict emotions for each segment of the audio.
    """
    segments = segment_audio(audio_path, segment_duration)
    emotion_scores_list = []

    for segment in segments:
        features = extract_features_from_segment(segment)
        if features is not None:
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=2)
            vdcnn_features = vdcnn.predict(features)
            predicted_probabilities = mlp_model.predict_proba(vdcnn_features)[0]
            emotion_scores = dict(zip(emotions, predicted_probabilities))
            emotion_scores_list.append(emotion_scores)

    return emotion_scores_list

def plot_emotions_over_time(emotion_scores_list, segment_duration):
    """
    Plot emotion probabilities over time.
    """
    time_stamps = [i * segment_duration for i in range(len(emotion_scores_list))]
    emotion_names = list(emotion_scores_list[0].keys())

    for emotion in emotion_names:
        plt.plot(time_stamps, [scores[emotion] for scores in emotion_scores_list], label=emotion)

    plt.title("Emotion Probabilities Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

def main(video_path):
    """
    Main function to process video and predict emotion.
    """
    # Extract audio from video
    audio_path = extract_audio_from_video(video_path)
    if audio_path is None:
        print("Audio extraction failed. Exiting.")
        return

    # Predict emotions over time
    emotion_scores_list = predict_emotions_over_time(audio_path)
    if not emotion_scores_list:
        print("Emotion prediction failed. Exiting.")
        return

    # Plot emotions over time
    plot_emotions_over_time(emotion_scores_list, segment_duration=2.5)

if __name__ == "__main__":
    video_file_path = "test_vid2.mp4"  # Replace with the path to your video file
    main(video_file_path)
