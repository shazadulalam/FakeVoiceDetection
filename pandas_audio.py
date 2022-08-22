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
import keras.backend.tensorflow_backend as k
 # import tensorflow.keras.layers.LeakyRelu as LeakyRelu
# from tf.keras.optimizers import Adam
# import tf.keras.backend as K
from matplotlib.pyplot import specgram
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 

#gpu uses

# from keras import backend as k
# config = tf.ConfigProto()                                   
# config.gpu_options.allow_growth = True                      
# config.gpu_options.per_process_gpu_memory_fraction = 0.8    
# k.set_session(tf.Session(config=config)) 



def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    for fp in  file_paths:
        X, sample_rate = librosa.load(fp, res_type='kaiser_fast', duration=3.00)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0)

        # print(mfccs)
        # new_shape = librosa.feature.melspectrogram(y = X, sr = sample_rate)
        # plt.figure(figsize=(10,4))
        # librosa.display.specshow(mfccs, x_axis='time')
        # plt.colorbar()
        # plt.title('MFSS')
        # plt.tight_layout()
        # plt.show()

        label = fp.split('.wav')

        raw_sounds.append(mfccs)
        # raw_sounds.append(new_shape)
        labels.append(label)
    raw_sounds = np.asarray(raw_sounds)
    # print(raw_sounds.shape)
    labels = np.asarray(labels)
    return raw_sounds, labels 
sound_file_paths = glob("/home/shul/audio/folder1/*.wav", recursive=True)

# def create_spectogram(filename, name):

#     plt.interactive(False)
#     clip, sample_rate = librosa.load(filename, sr=None)
#     fig = plt.figure(figsize=[0.72, 0.72])
#     ax = fig.add_subplot(111)
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#     ax.set_frame_on(False)
#     S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
#     librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
#     filename = '/home/shul/audio/train/' + name + '.jpg'
#     plt.savefig(filename, dpi = 400, bbox_inches = 'tight', pad_inches  = 0)
#     plt.close()
#     fig.clf()
#     plt.close(fig)
#     plt.close('all')
#     del filename, name, clip, sample_rate, fig, ax, S

Data_dir = np.array(glob("/home/shul/audio/folder1/*"))

# i = 0
# for file in Data_dir[i : i + 4000]:

#     filename, name = file, file.split('/')[-1].split('.') [0]
#     create_spectogram(filename, name)

# gc.collect()

# i = 4000
# for file in Data_dir[i : i + 4000]:
#     filename, name = file, file.split('/')[-1].split('.')[0]
#     create_spectogram(filename, name)

# gc.collect()

className = []
name = []
labelName = []
i = 0

# with open('/home/shul/audio/train/train.csv') as csvDataFile:
#     readCsv = csv.reader(csvDataFile)
#     for row in readCsv:
#         # name = row
#         labelName.append(row)

# labelName = np.asarray(labelName)

# print(labelName)

for file in Data_dir[i : i + 8000]:
    classname, naame = file, file.split('/')[-1].split('.')[0].split(' (')[0]
    className.append(className)
    name.append(naame)

labelName = np.sort(name)
# print(labelName)

label_encoder = LabelEncoder()
label_concoded = label_encoder.fit_transform(labelName)
# print(label_concoded)

df = pd.DataFrame({'id': label_concoded,
                    'class name': labelName}).drop_duplicates()
                    
# df.to_csv('/home/shul/audio/train/train.csv', encoding='utf-8', index=False)
print(df['class name'])

encoded_test = tf.keras.utils.to_categorical(label_concoded)
inverted_test = argmax(encoded_test[0])
# print(encoded_test, inverted_test)

label_to_int = {k:v for v,k in enumerate(labelName)}
# print(label_to_int)
values = []
int_to_label = {values.append(k) for k,v in label_to_int.items()}
# for k, v in int_to_label:
#     values.append(v)
values = np.asarray(values)
values = values.reshape(1369, 1)
# print(values)


raw_sounds, labels = load_sound_files(sound_file_paths[:8000])
raw_sounds = np.reshape(raw_sounds, (8000, 8, 8, 1))

X_train = np.asarray(raw_sounds[0:6000])
X_test = np.asarray(raw_sounds[6000:])
y_train = X_train.shape[0:1]
y_test = X_test.shape[0:1]
# print(X_train.shape, "_________", X_test.shape[0:1])
# print(y_train, "_________", y_test)

labelEncoder = LabelEncoder()
y_train_encoded = labelEncoder.fit_transform(values)
y_test_encoded = labelEncoder.fit_transform(y_test)

Y_train = np.array(tf.keras.utils.to_categorical(y_train_encoded, len(labelName)))
Y_test = np.array(tf.keras.utils.to_categorical(y_test_encoded, len(labelName)))

# print(Y_train, ".................", Y_test)
# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.20, random_state=42)
batch_size = 32
# input_shape = X_train.shape[1:1]
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3,3),input_shape=(8,8,1)))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3,3)))
model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(128, (3, 3)))
# model.add(layers.Activation('relu'))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.25))
# model.add(layers.Dense(512))
# model.add(layers.Activation('relu'))
# model.add(layers.Dense(1))
# model.add(layers.Activation('sigmoid'))
model.summary()

sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.01, decay = 1e-6,momentum=0.9, nesterov=False)
model.compile(loss =  'binary_crossentropy', optimizer = "Adam", metrics = ['accuracy'])



model.fit(X_train, batch_size=batch_size, epochs = 10)
# model.save('audioDaTa.model')
# y_pred = model.predict_classes(X_train)

# prediction = []
# for i in range(len(X_train)):
#     predictedAudio = y_pred[i]
#     prediction.append(predictedAudio)

#     print("X=%s, Predicted=%s" % (X_train[i], y_pred[i]))

# plt.plot(prediction[1])
# plt.show()