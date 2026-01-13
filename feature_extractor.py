import os
import json
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "genres_original"

CSV_PATH = "features.csv"

# Define constants for audio processing
SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS=30
NUM_SEGMENTS = 10

# Define constants for MFCC extraction
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Calculate the number of audio samples we expect per 30-second track
SAMPLE_PER_TRACK = SAMPLE_RATE*TRACK_DURATION_SECONDS


def process_dataset(dataset_path,csv_path):
    data= {
        "mapping":[],
        "labels":[],
        "features":[]
    }
    print("Starting feature extraction...")

    for i,genre_folder in enumerate(sorted(os.listdir(dataset_path))):
        genre_path = os.path.join(dataset_path,genre_folder)

        if os.path.isdir(genre_path):

            data["mapping"].append(genre_folder)
            print(f"\nProcessing genre: {genre_folder}")

            for filename in sorted(os.listdir(genre_path)):

                if filename.endswith(".wav"):

                    file_path = os.path.join(genre_path,filename)

                    try:
                        signal,sr= librosa.load(file_path,sr=SAMPLE_RATE)

                        if len(signal) >=SAMPLE_PER_TRACK:
                            num_samples_per_segment = int(SAMPLE_PER_TRACK/NUM_SEGMENTS)#amplitude readings

                            for s in range(NUM_SEGMENTS):
                                start_sample = s * num_samples_per_segment
                                end_sample = start_sample + num_samples_per_segment

                                segment = signal[start_sample:end_sample]

                                # Extract features
                                mfccs = librosa.feature.mfcc(y=segment, sr=sr , n_mfcc=NUM_MFCC,n_fft =N_FFT,hop_length=HOP_LENGTH)
                                mfccs_processed = np.mean(mfccs,axis = 1)

                                chroma = librosa.feature.chroma_stft(y=segment,sr=sr,n_fft=N_FFT,hop_length=HOP_LENGTH)
                                chroma_processed = np.mean(chroma,axis=1)

                                spectral_centroid = librosa.feature.spectral_centroid(y=segment,  sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
                                spectral_centroid_processed = np.mean(spectral_centroid)

                                spectral_rolloff = librosa.feature.spectral_rolloff(y=segment,  sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
                                spectral_rolloff_processed = np.mean(spectral_rolloff)

                                zcr = librosa.feature.zero_crossing_rate(y=segment,hop_length=HOP_LENGTH)
                                zcr_processed = np.mean(zcr)

                                # Combine all features into a single feature vector
                                feature_vector = np.hstack((mfccs_processed, chroma_processed, spectral_centroid_processed, spectral_rolloff_processed,zcr_processed))

                                # Store the feature vector and its corresponding label
                                data["features"].append(feature_vector.tolist())
                                data["labels"].append(i)


                    except Exception as e:
                        print(f"Error loading file {file_path}:{e}")
                        continue
    
    print("\nConverting data to pandas DataFrame...")

    features_df = pd.DataFrame(data["features"])
    features_df["genre_label"]=data["labels"]
    #print("DataFrame created successfully . Here are the first 5 rows:")
    #print(features_df.head())

    print(f"Saving DataFrame to {csv_path}...")
    features_df.to_csv(csv_path, index=False)

    print("\nFeature extraction complete. The file 'features.csv' has been created")

if __name__=="__main__":
    process_dataset(DATASET_PATH,CSV_PATH)