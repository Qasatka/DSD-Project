import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import librosa
import numpy as np
def cut_song(song):
  start = 0
  end = len(song)
  
  song_pieces = []

  while start + 10000 < end:
    song_pieces.append(song[start:start+10000])

    start += 10000
  
  return song_pieces
def prepare_song(song_path):
  list_matrices = []
  y,sr = librosa.load(song_path,sr=22050)
  song_pieces = cut_song(y)
  for song_piece in song_pieces:
    melspect = librosa.feature.melspectrogram(song_piece)
    list_matrices.append(melspect)
  return list_matrices
fs = 44100  # Sample rate
seconds = 3  # Duration of recording
print("speak")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file
song_path = 'output.wav'
x,sr = librosa.load(song_path,sr=22050)
x = prepare_song(song_path)
x=np.expand_dims(x, axis=-1)
new_model = tf.keras.models.load_model('saved_model/first_model')
ypred = new_model.predict(x)
print(ypred)