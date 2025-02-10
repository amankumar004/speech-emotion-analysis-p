import cv2
import mediapipe as mp
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
from moviepy.editor import AudioFileClip
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam

# Load the trained VDCNN model
vdcnn_model = load_model("vdcnn_model.keras")

# Compile the VDCNN model manually
vdcnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the trained MLP model
with open("mlp_model.pkl", "rb") as model_file:
    mlp_model = pickle.load(model_file)

# Define observed emotions
observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define the VDCNN model
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

vdcnn = build_vdcnn((180, 1))

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_frames_from_video(video_path, interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_timestamps = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
            frame_timestamps.append(frame_count / fps)
        frame_count += 1

    cap.release()
    return frames, frame_timestamps

def analyze_and_display_emotions_and_hands(frames, frame_timestamps):
    consolidated_results = []

    for idx, (frame, timestamp) in enumerate(zip(frames, frame_timestamps)):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(frame_rgb)
            hand_landmarks = []
            if hand_results.multi_hand_landmarks:
                for hand_landmark in hand_results.multi_hand_landmarks:
                    hand_landmarks.append([(lm.x, lm.y) for lm in hand_landmark.landmark])
            hand_movement = bool(hand_landmarks)

            result = DeepFace.analyze(img_path=frame_rgb, actions=['emotion'])

            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if 'emotion' in first_result:
                    emotions = first_result['emotion']
                    consolidated_results.append({
                        'frame_index': idx,
                        'timestamp': timestamp,
                        'emotions': emotions,
                        'hand_movement': hand_movement
                    })

        except Exception as e:
            print(f"Error analyzing face for frame {idx}: {e}")

    return consolidated_results

def extract_audio_from_video(video_path, output_audio_path="extracted_audio.wav"):
    try:
        audio_clip = AudioFileClip(video_path)
        audio_clip.write_audiofile(output_audio_path, fps=22050, codec="pcm_s16le")
        audio_clip.close()
        return output_audio_path
    except Exception as e:
        print(f"Error during audio extraction: {e}")
        return None

def segment_audio(audio_path, segment_duration=2.5, overlap=0.5, sample_rate=22050):
    try:
        audio_data, _ = librosa.load(audio_path, sr=sample_rate)
        segment_length = int(segment_duration * sample_rate)
        hop_length = int((1 - overlap) * segment_length)
        
        segments = []
        timestamps = []
        
        for i in range(0, len(audio_data) - segment_length + 1, hop_length):
            segment = audio_data[i:i + segment_length]
            if len(segment) == segment_length:
                segments.append(segment)
                timestamps.append(i / sample_rate)
                
        return segments, timestamps
    except Exception as e:
        print(f"Error segmenting audio: {e}")
        return [], []

def pad_features(features_list):
    max_length = max(len(features) for features in features_list)
    padded_features = np.array([np.pad(features, (0, max_length - len(features)), mode='constant') for features in features_list])
    return padded_features

def extract_features_from_segment(segment, sample_rate=22050):
    try:
        # Extract fixed-length features
        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=60)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Match feature extraction from training
        stft = np.abs(librosa.stft(segment))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=segment, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sample_rate).T, axis=0)
        
        # Combine features
        features = np.concatenate([mfccs_scaled, chroma, mel, contrast])
        
        # Ensure exact feature length
        target_length = 6656
        if len(features) < target_length:
            features = np.pad(features, (0, target_length - len(features)), mode='constant')
        else:
            features = features[:target_length]
            
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_emotions_over_time(audio_path, segment_duration=2.5, overlap=0.5, sample_rate=22050):
    segments, timestamps = segment_audio(audio_path, segment_duration, overlap, sample_rate)
    features_list = [extract_features_from_segment(segment) for segment in segments]
    features_list = [features for features in features_list if features is not None]
    
    if not features_list:
        return [], []
    
    padded_features = pad_features(features_list)
    padded_features = np.expand_dims(padded_features, axis=2)  # Reshape for CNN
    
    # Flatten the features for MLP
    flattened_features = padded_features.reshape(padded_features.shape[0], -1)
    
    predicted_probabilities = mlp_model.predict_proba(flattened_features)
    
    emotion_scores_list = []
    for probs in predicted_probabilities:
        emotion_scores = {emotion: prob for emotion, prob in zip(observed_emotions, probs)}
        emotion_scores_list.append(emotion_scores)
    
    return emotion_scores_list, timestamps

def plot_image_analysis(consolidated_results, frames):
    if not consolidated_results:
        print("No data available for image analysis.")
        return

    frame_indices = [result['frame_index'] for result in consolidated_results]
    emotions_list = [result['emotions'] for result in consolidated_results]
    hand_movements = [int(result['hand_movement']) for result in consolidated_results]

    emotion_names = list(emotions_list[0].keys())
    emotion_sums = {emotion: sum([emotions[emotion] for emotions in emotions_list]) for emotion in emotion_names}

    plt.figure(figsize=(10, 6))
    plt.bar(emotion_sums.keys(), emotion_sums.values(), color='skyblue', label='Emotion Scores')
    plt.title("Total Emotion Scores from Image Analysis")
    plt.xlabel("Emotions")
    plt.ylabel("Total Score")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    for emotion in emotion_names:
        plt.plot(frame_indices, [emotions[emotion] for emotions in emotions_list], label=emotion)
    plt.plot(frame_indices, hand_movements, label="Hand Movements", linestyle="--", color="orange")
    plt.title("Emotion Scores and Hand Movements Over Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Emotion Score / Hand Movement")
    plt.legend()
    plt.show()

    # # Display consolidated frames with annotations
    # plt.figure(figsize=(15, len(consolidated_results) * 5))
    # for idx, result in enumerate(consolidated_results):
    #     frame_index = result['frame_index']
    #     emotions = result['emotions']
    #     timestamp = result['timestamp']
    #     hand_movement = result['hand_movement']

    #     frame_rgb = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2RGB)

    #     plt.subplot(len(consolidated_results), 1, idx + 1)
    #     plt.imshow(frame_rgb)
    #     plt.axis('off')
    #     plt.title(f'Emotions and Hand Movements for Frame {frame_index} at {timestamp:.2f}s')

    #     for emotion, value in emotions.items():
    #         plt.text(40, 40 + list(emotions.keys()).index(emotion) * 20, f'{emotion}: {value:.2f}', fontsize=8, color='red')

    # plt.tight_layout()
    # plt.show()

def plot_total_emotion_scores(emotion_scores_list):
    if not emotion_scores_list:
        return
    
    # Calculate total emotion scores
    emotion_sums = {emotion: 0 for emotion in observed_emotions}
    for scores in emotion_scores_list:
        for emotion, score in scores.items():
            emotion_sums[emotion] += score
    
    # Plot total emotion scores
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_sums.keys(), emotion_sums.values(), color='lightgreen', label='Emotion Scores')
    plt.title("Total Emotion Scores from Speech Analysis")
    plt.xlabel("Emotions")
    plt.ylabel("Total Score")
    plt.legend()
    plt.show()

def plot_speech_analysis(emotion_scores_list, timestamps):
    if not emotion_scores_list:
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot emotion probabilities over time
    for emotion in observed_emotions:
        scores = [scores[emotion] for scores in emotion_scores_list]
        plt.plot(timestamps, scores, label=emotion, alpha=0.7)
        
    plt.title("Emotion Analysis Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Emotion Probability")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main(video_path):
    frames, frame_timestamps = extract_frames_from_video(video_path)
    consolidated_results = analyze_and_display_emotions_and_hands(frames, frame_timestamps)
    plot_image_analysis(consolidated_results, frames)

    audio_path = extract_audio_from_video(video_path)
    if audio_path:
        emotion_scores_list, timestamps = predict_emotions_over_time(audio_path)
        plot_speech_analysis(emotion_scores_list, timestamps)
        plot_total_emotion_scores(emotion_scores_list)

if __name__ == "__main__":
    video_file_path = "test_sample.mp4"  # Replace with your video file path
    main(video_file_path)