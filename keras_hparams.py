
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib.pyplot import specgram
from tensorflow.keras.models import Sequential
from kerastuner.tuners import RandomSearch
from sklearn.decomposition import PCA
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Maxpooling2D

tf.compat.v1.reset_default_graph()


def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    for fp in  file_paths:
        X, sample_rate = librosa.load(fp, res_type='kaiser_fast', duration=3.00)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=64).T,axis=0)
        # new_shape = librosa.feature.melspectrogram(y = X, sr = sample_rate)
        label = fp.split('.wav')

        raw_sounds.append(mfccs)
        # raw_sounds.append(new_shape)
        labels.append(label)
    raw_sounds = np.asarray(raw_sounds)
    labels = np.asarray(labels)
    return raw_sounds, labels


sound_file_paths = glob.glob("/home/grmi/audio/audio0209/wavs/*.wav")

raw_sounds, labels = load_sound_files(sound_file_paths[:8000])
raw_sounds = np.reshape(raw_sounds, (8000, 4,4,4))
# labels = 8000
# print(len(raw_sounds[1]))
# print("##################################################################", raw_sounds.shape)
# print(labels[1:10])

################################Principal component analysis start ##################################

pca = PCA(n_components=2)
# pca.fit(raw_sounds)

s = pca.fit_transform(raw_sounds)
print("________________________________________________", s)

S_ = pca.components_

print("#########################", S_)

x=[]
for i in range(len(S_)):
    ver_vec = S_[i]
    x.append(ver_vec)

x = np.asarray(x)
new_raw_sounds = x.reshape(8000,4,4,4)

print(new_raw_sounds)

weights= raw_sounds
bias = np.zeros(8000)

batch_size = 32

num_classes = 16

def hp_model(hp):

    model = tf.keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units', min_value=32,
                                                max_value=512,
                                                step=32),
                                        activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(
                    hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model

tuner = RandomSearch(hp_model, objective='val_acccuracy', max_trials=5)

tuner.search(raw_sounds, epochs=5)
tuner.results_summary()

