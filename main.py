import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import librosa
import speech_recognition as sr
def cut_song(song):
  start = 0
  end = len(song)
  
  song_pieces = []

  while start + 100000 < end:
    song_pieces.append(song[start:start+100000])

    start += 100000

  return song_pieces
def prepare_song(song_path):
  list_matrices = []
  y,sr = librosa.load(song_path,sr=22050)
  song_pieces = cut_song(y)
  for song_piece in song_pieces:
    melspect = librosa.feature.melspectrogram(song_piece)
    list_matrices.append(melspect)
  return list_matrices

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('saved_model/first_model')

