import os 
import random
import pandas as pd 
# from memory_profiler import memory_usage
from glob import glob
import numpy as np 
from numpy import argmax
import librosa 
import librosa.display as ld
import pylab 
import matplotlib.pyplot as plt 
from matplotlib import figure
# from tensorflow.keras.utils import np_utils # from keras import utils as np_utils
# from comet_ml import Experiment
import gc 
import csv
# from path import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import keras.backend.tensorflow_backend as k
 # import tensorflow.keras.layers.LeakyRelu as LeakyRelu
# from tf.keras.optimizers import Adam
from scipy.io import wavfile as wav
from matplotlib.pyplot import specgram
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
# experiment = Experiment(api_key="API_KEY",
#                         project_name="customSound")


def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    for fp in  file_paths:
        X, sample_rate = librosa.load(fp, res_type='kaiser_fast', duration=3.00) #sample_rate=22050Hz, audio_length=3.00sec(on average)
        # print("Sample rate: {0}Hz".format(sample_rate))
        # print("Audio duration: {0}s".format(len(X) / sample_rate))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=25).T, axis=0)
        # librosa.display.specshow(mfccs)
        # plt.plot(mfccs)
        # plt.show()
        raw_sounds.append(mfccs)
    raw_sounds = np.asarray(raw_sounds)
    return raw_sounds

Data_dir = np.array(glob("/home/shul/audio/originalData/*"))

original_class = []
name = []
labelName1 = []
original_feature = []
i = 0
for file in Data_dir[i : i + 8000]:


    name.append(0)
# labelName = np.asarray(original_class)
original_class = np.asarray(name)


label_encoder = LabelEncoder()
label_concoded_original = label_encoder.fit_transform(original_class)
# print(label_concoded)

sounds_file_original = glob("/home/shul/audio/originalData/*.wav")
original_feature = load_sound_files(sounds_file_original[:8000])
original_feature = original_feature.reshape(8000, 5, 5 , 1)

# print(original_feature.shape)
data = dict()


# print(df)
Data_dir2 = np.asarray(glob("/home/shul/audio/fakeData/*"))
fake_class = []
label_name2 = []
fake_feature = []
j = 0

for file in Data_dir2[j: j + 3000]:


    fake_class.append(1)

# label_name2 = np.asarray(fake_class)
fake_class = np.asarray(fake_class)

# print(fake_class)
label_encoder2 = LabelEncoder()
label_encoded_fake = label_encoder2.fit_transform(fake_class)

sounds_file_fake = glob("/home/shul/audio/fakeData/*.wav")
fake_feature = load_sound_files(sounds_file_fake[:3000])
fake_feature = fake_feature.reshape(3000, 5, 5 , 1)

# print(len(original_feature), len(fake_feature))
data2 = dict()


finalDataDict = {}
total_feature = []
# for key in (data.keys() | data2.keys()):
#     if key in data:
#         finalDataDict.setdefault(key, []).append(data[key])
#     if key in data2:
#         finalDataDict.setdefault(key, []).append(data2[key])
# finalDataDict = data.append(data2, ignore_index=True) 

total_feature = np.concatenate((original_feature, fake_feature), axis=0)
# print(finalDataDict)

new_data = np.concatenate((original_feature, fake_feature))
new_label = np.concatenate((label_concoded_original, label_encoded_fake))

shuffle_feature, shuffle_label = shuffle(new_data, new_label)

# print(shuffle_feature[0:5], shuffle_label[0:5])


X_train = shuffle_feature[0:9000].reshape(9000, 5, 5, 1)
x_test = shuffle_feature[9000:].reshape(2000, 5, 5, 1)
Y_train = shuffle_label[0:9000]
y_test = shuffle_label[9000:]
new_y_train= tf.keras.utils.to_categorical(Y_train, num_classes=2)

batch_size = 64
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (5,5),input_shape=(5,5,1)))
model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(64, (3,3)))
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPool2D(pool_size=(2,2)))
# model.add(layers.Dropout(0.25))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(128, (2,2)))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(128, (2,2)))
# model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128))
model.add(layers.Activation('relu'))
model.add(layers.Dense(2))
model.add(layers.Activation('sigmoid'))

model.summary()
# sgd = tf.compat.v2.keras.optimizers.Adadelta(lr=0.0001, rho=0.95, epsilon = 1e-07)

sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.0001, decay = 1e-6,momentum=0.9, nesterov=False)
model.compile(loss =  'sparse_categorical_crossentropy', optimizer = "Adam", metrics = ['accuracy'])



model.fit(X_train, Y_train, batch_size=batch_size, epochs = 1000)