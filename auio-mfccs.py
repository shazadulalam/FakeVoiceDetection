

import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib, IPython.display
import librosa.display



x, fs = librosa.load("/home/shul/audio/originalData/th_th_europa_170519 (478).wav")
#librosa.display.waveplot(x, sr=fs)

mfccs = librosa.feature.mfcc(x, sr=fs)
print (mfccs.shape)
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print (mfccs.mean(axis=1))
print (mfccs.var(axis=1))

librosa.display.specshow(mfccs, sr=fs, x_axis='time')
plt.savefig('/home/shul/code/plot_image/th_th_europa_170519(478).png', dpi=100)