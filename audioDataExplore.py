import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd 

import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

audio_path = Path('H:/prac/speech commands/yes/00f0204f_nohash_0')
#filename = 'wavs/LJ001-0001.wav'

sample_rate, samples  =librosa.load(audio_path)
# sample_rate = np.array(sample_rate)
# samples = np.array(samples)
def log_spectogram(audio, sample_rate, window_size = 20,
                    step_size  =10, eps = 1e-10):
    nperseg  = int(round(window_size * sample_rate / 1e3))
    noverlap  = int(round(step_size * sample_rate / 1e3))

    freq, times, spec = signal.spectrogram(audio, fs = sample_rate, window = 'hann',
                                        nperseg  = nperseg , noverlap  = noverlap , detrend = False)
    return freq, times, np.log(spec.T.astype(np.float32) + eps)

freq, time, spectrogram = log_spectogram(samples, sample_rate)

print(len(samples), "----", sample_rate)
# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(211)
# ax1.set_title('Raw wave of ' + audio_path.name)
# ax1.set_ylabel('Amplitude')
# ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

# ax2 = fig.add_subplot(212)
# ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
#            extent=[time.min(), time.max(), freq.min(), freq.max()])
# ax2.set_yticks(freq[::16])
# ax2.set_xticks(time[::16])
# ax2.set_title('Spectrogram of ' + audio_path.name)
# ax2.set_ylabel('Freqs in Hz')
# ax2.set_xlabel('Seconds')

mean = np.mean(spectrogram, axis = 0)
std  = np.std(spectrogram, axis = 0)
spectrogram = (spectrogram -mean) / std
print(spectrogram)

S = librosa.feature.melspectrogram(y = samples, sr = sample_rate, n_mels  = 128, fmax = 8000)

log_S = librosa.power_to_db(S, ref = np.max)

plt.figure(figsize = (14, 8))
librosa.display.specshow(log_S, sr = sample_rate, x_axis = 'time', y_axis = 'mel')
plt.title('mel power sepctrogram')
plt.colorbar(format = '%+02.0f dB')
plt.tight_layout()
#print(freq,"_____________",  time)