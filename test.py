import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import librosa
import numpy as np
fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file