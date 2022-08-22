import os 
import pandas as pd 
# from memory_profiler import memory_usage
from glob import glob
import numpy as np 
from numpy import argmax
import librosa 
import librosa.display
import pylab 
import matplotlib.pyplot as plt 
from matplotlib import figure
import gc 
import csv
# from path import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
# import keras.backend.tensorflow_backend as k
 # import tensorflow.keras.layers.LeakyRelu as LeakyRelu
# from tf.keras.optimizers import Adam
# import tf.keras.backend as K
from matplotlib.pyplot import specgram
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 




def create_spectogram(filename, name):

    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate, n_mels = 224)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = '/home/shul/audio/FoR/for-rerecorded/validation/real_spec/' + name + '.jpg'
    plt.savefig(filename, dpi = 100, bbox_inches = 'tight', pad_inches  = 0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S

Data_dir = np.array(glob("/home/shul/audio/FoR/for-rerecorded/validation/real/*"))

i = 0
for file in Data_dir[i : i + 5100]:

    filename, name = file, file.split('/')[-1].split('.') [0]
    create_spectogram(filename, name)

# gc.collect()

# i = 4000
# for file in Data_dir[i : i + 4000]:
#     filename, name = file, file.split('/')[-1].split('.')[0]
#     create_spectogram(filename, name)

# gc.collect()



# for file in Data_dir[i : i + 8000]:
#     classname, naame = file, file.split('/')[-1].split('.')[0].split(' (')[0]
#     className.append(className)
#     name.append(naame)'


# model.save('audioDaTa.model')
# y_pred = model.predict_classes(X_train)

# prediction = []
# for i in range(len(X_train)):
#     predictedAudio = y_pred[i]
#     prediction.append(predictedAudio)

#     print("X=%s, Predicted=%s" % (X_train[i], y_pred[i]))

# plt.plot(prediction[1])
# plt.show()