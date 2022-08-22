import os
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import librosa
import librosa.display
import pandas as pd

#simple plot
file = '/home/forhad/Study/office/townhall_new (95).wav'
x, sr = librosa.load(file, sr=42000)
# print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, Fs, x.shape, x.dtype))
plt.figure(figsize=(8, 2))
librosa.display.waveplot(x, sr=sr)
plt.show()

#spectogram
audio_stft = librosa.stft(x)
audio_stft_db = librosa.amplitude_to_db(abs(audio_stft))
plt.figure(figsize=(14,5))
librosa.display.specshow(audio_stft_db, sr=sr, x_axis = 'time', y_axis='log')
plt.colorbar()
# plt.show()

#spectral centroid
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
(775,)
# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
plt.show()