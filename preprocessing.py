import os
import librosa
import pandas as pd

classical_path = 'data/classical/' #hon men hot l path tabaa l folders wen rah ykoun fi l music
disco_path = 'data/disco/'

# Function to extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)  # 13 coefficients to capture more nuanced features of the sound 
    return [chroma, tempo, spectral_centroid, zcr, *mfcc]

# Process the datasets
def process_dataset(dataset_path, label): #it loops through kel l audio files in the directory and extracts the features for each, also labels them with the genre
    features = []
    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)
        try:
            data = extract_features(file_path)
            data.append(label)  # Add genre label
            features.append(data)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    return features

# Extract features for both genres
classical_features = process_dataset(classical_path, 'classical') 
disco_features = process_dataset(disco_path, 'disco')

columns = ['Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate', 'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13', 'Label']
df = pd.DataFrame(classical_features + disco_features, columns=columns)

df.to_csv('audio_features.csv', index=False) #audio_features.csv rah ykoun bel current working directory 

print("Feature extraction completed and saved to 'audio_features.csv'")