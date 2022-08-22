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


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# (rate,sig) = wav.read("file.wav")
# mfcc_feat = mfcc(sig,rate)

# ig, ax = plt.subplots()
# mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)
# cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
#___________________________________MFCCS_______________________________________#
# y, sr = librosa.load("/home/shul/audio/in_house/originalData/townhall_new (98).wav")
# librosa.feature.mfcc(y=y, sr=sr)
# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
# librosa.feature.mfcc(S=librosa.power_to_db(S))
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.savefig('mfccs.png', dpi=100)
# #Showing mfcc_data
# plt.show()
#________________________________END___________________________________________#
#________________________________Sspectogram___________________________________#

import matplotlib.pyplot as plot
from scipy.io import wavfile


# Read the wav file (mono)

samplingFrequency, signalData = wavfile.read("/home/shul/audio/in_house/originalData/townhall_new (98).wav")
 
# Plot the signal read from wav file
plot.subplot(211)
plot.title('Spectrogram of a wav file')

plot.plot(signalData)
plot.xlabel('Sample')
plot.ylabel('Amplitude')
 
plot.subplot(212)
plot.specgram(signalData,Fs=samplingFrequency)
plot.xlabel('Time')
plot.ylabel('Frequency')

plt.savefig('spectogram.png', dpi=100)
plot.show()

#gpu uses

# # from keras import backend as k
# config = tf.ConfigProto()                                   
# config.gpu_options.allow_growth = True                      
# config.gpu_options.per_process_gpu_memory_fraction = 0.8    
# k.set_session(tf.Session(config=config)) 




# Data_dir = np.array(glob("/home/shul/audio/audios/folder1/*"))

# className = []
# name = []
# labelName = []
# i = 0


# for file in Data_dir[i : i + 8000]:

#     naame = file.split('/')[-1]
#     classname =  file.split('/')[-1].split('.')[0]
#     className.append(classname)
#     name.append(naame)
# className = np.asarray(className)
# labelName = np.asarray(name)
# # print(className)
# # print(labelName)
# # print(len(labelName))

# label_encoder = LabelEncoder()
# label_concoded = label_encoder.fit_transform(labelName)
# df = pd.DataFrame({ 'id' : label_concoded,
#                     'file_name': labelName,
#                     'folder' : 1,
#                     'class': className
#                     })

# Data_dir2 = np.asarray(glob("/home/shul/audio/audios/folder2/*"))
# name_class = []
# label_name = []
# j = 0

# for file in Data_dir2[j: j + 3015]:

#     filename = file.split('/')[-1]
#     classname = file.split('/')[-1].split('.')[0]
#     name_class.append(classname)
#     label_name.append(filename)

# name_class = np.asarray(name_class)
# label_name = np.asarray(label_name)

# label_encoder2 = LabelEncoder()
# label_encoded = label_encoder2.fit_transform(label_name)
# df1 = pd.DataFrame({ 'id' : label_encoded,
#                     'file_name' : label_name,
#                     'folder' : 2,
#                     'class' : name_class
#                     })

# finalDf = df.append(df1, ignore_index=True) 
# # print(finalDf.head())

# finalDf.to_csv('/home/shul/audio/train/train.csv', encoding='utf-8', index=False)
# print("writing complete ... ")



# (X_train, Y_train) = combine_feature()
#     batch_size = 32
#     epochs = 60
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(64, (5, 3),input_shape=(8,5, 1)))
#     model.add(layers.Activation('elu'))
#     # model.add(layers.Conv2D(16, (3, 3)))
#     # model.add(layers.MaxPool2D(2, 2))
    
#     # model.add(layers.Conv2D(64, (3, 3)))
#     # model.add(layers.Activation('elu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dropout(0.25))
#     # model.add(layers.Dense(32))
#     # model.add(layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l1(l=0.01)))
#     # model.add(layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(l=0.01)))
#     # model.add(layers.Activation('elu'))
#     model.add(layers.Activation('softmax'))
#     # model.add(layers.Activation('sigmoid'))