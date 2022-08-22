
import os 
import random
import pandas as pd 
import glob
import glob2
import numpy as np 
from numpy import argmax
from librosa import display
import numpy as np
from numpy import genfromtxt
import librosa
import pylab 
import matplotlib.pyplot as plt 
from matplotlib import figure
from time import time
# import pickle
import gc 
import csv
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
# from tf.keras.utils import plot_model
from scipy.io import wavfile as wav
from matplotlib.pyplot import specgram
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
import _pickle as cPickle
import gzip
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib.request, urllib.parse, urllib.error
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import backend
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LeakyReLU,Activation,MaxPool2D
from tensorflow.keras.regularizers import l2
import torchaudio
import torchvision
import torch


df=pd.read_csv("/home/prah/audio/UrbanSound8K/metadata/UrbanSound8K.csv")

y,sr=librosa.load("/home/prah/audio/UrbanSound8K/audio/fold5/100032-3-0-0.wav")
mfccs = librosa.feature.mfcc(y, sr, n_mfcc=40)
melspectrogram =librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000)
chroma_stft=librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40)
melspectrogram.shape,chroma_stft.shape,mfccs.shape

y,sr=librosa.load("/home/prah/audio/UrbanSound8K/audio/fold5/100263-2-0-137.wav")
mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
melspectrogram.shape,chroma_stft.shape,mfccs.shape

features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft)),(40,3))
features.shape

x_train=[]
x_test=[]
y_train=[]
y_test=[]
path="/home/prah/audio/UrbanSound8K/audio/fold"
for i in tqdm(range(len(df))):
    fold_no=str(df.iloc[i]["fold"])
    file=df.iloc[i]["slice_file_name"]
    label=df.iloc[i]["classID"]
    filename=path+fold_no+"/"+file
    y,sr=librosa.load(filename)
    mfccs = librosa.feature.mfcc(y, sr).T
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr,fmax=8000).T
    chroma_stft= librosa.feature.chroma_stft(y=y, sr=sr).T
    features= mfccs,melspectrogram,chroma_stft
    

    
    if(fold_no!='10'):
      x_train.append(features)
      y_train.append(label)
    else:
      x_test.append(features)
      y_test.append(label)

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

x_train_2d=np.reshape(x_train,(x_train.shape[0]*x_train.shape[1]))
x_test_2d=np.reshape(x_test,(x_test.shape[0]*x_test.shape[1]))
x_train_2d.shape,x_test_2d.shape



y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
y_train.shape,y_test.shape

x_train=np.reshape(x_train,(x_train.shape[0],3))
x_test=np.reshape(x_test,(x_test.shape[0],3))
x_train.shape,x_test.shape




    
model = tf.keras.Sequential()

model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(40,3,1)))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(10,activation="softmax"))


sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.001, decay = 1e-6, nesterov=False)
adam = tf.compat.v2.keras.optimizers.Adam(lr=0.001)
    
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=50,epochs=50,validation_data=(x_test,y_test))

train_loss_score=model.evaluate(x_train,y_train)
test_loss_score=model.evaluate(x_test,y_test)
print(train_loss_score)
print(test_loss_score)


    
    
    


