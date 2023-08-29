import os
import librosa
import numpy as np
import pandas as pd

data_combined = []

data_labelled = {
    0: ['dataset/fsd50k_sample/' + f for f in os.listdir("dataset/fsd50k_sample/")],
    1: ['dataset/positve data/generated_clips/' + f for f in os.listdir("dataset/positve data/generated_clips/")]
}

for label, files in data_labelled.items():
    for file in files:
        data, sample_rate = librosa.load(file)
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfcc_processed = np.mean(mfccs.T, axis=0)
        data_combined.append([mfcc_processed, label])
    print(f"Processed label : {label}")

df = pd.DataFrame(data_combined, columns=["features", "label"])
df.to_pickle("processed_data.csv")