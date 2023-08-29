import sounddevice as sd
from scipy.io.wavfile import write, read
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import time
import pydub

fs = 16000
seconds = 2
filename = "prediction_sample.wav"
class_names = ["wake word NOT detected", "Wake word detected"]

model = load_model("wake_word_model/WWD.h5")

print("Prediction starts...")

i = 0

while True:
    print("Say hey d-tech : ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    # time.sleep(2)
    print("Recorded")
    sd.wait()
    write(filename, fs, myrecording)
    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    print(f"Prediction info :: {prediction[:,1]}")
    print(f"confidence : {prediction[:,-1]}")

    if prediction[:,1] > 0.95:
        print(f"Wake Word detected ({i})")
        break


# audio, sample_rate = librosa.load("record_out (5).wav")
# mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
# mfcc_processed = np.mean(mfcc.T, axis=0)

# prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
# print(f"Prediction info :: {prediction[:,1]}")
# print(f"confidence : {prediction[:,-1]}")

# if prediction[:,1] > 0.95:
#     print(f"Wake Word detected ({i})")
