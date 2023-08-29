from pydub import *
import os

# [AudioSegment.from_file(f, format="wav").speedup(playback_speed=0.75) for f in os.listdir("dataset/positve data/slowed")]


audio = AudioSegment.from_file("dataset/uOd8YG31.wav", format="wav") # wav
audio.speedup(1.75)
audio.export("dataset/uOd8YG31-slowed.wav", format="wav")
