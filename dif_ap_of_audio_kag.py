import os 
import random
import pandas as pd 
from tqdm import tqdm
import glob2
import numpy as np 
from numpy import argmax
import librosa 
import librosa.display as ld
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
import keras.backend.tensorflow_backend as k
from scipy.io import wavfile as wav
from matplotlib.pyplot import specgram
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
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
from sklearn.externals import joblib 
import cPickle as pickle
import gzip

# experiment = Experiment(api_key="API_KEY",
#                         project_name="customSound")


def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    for fp in  tqdm(file_paths):
        signal, sample_rate = librosa.load(fp, res_type='kaiser_fast', duration=2.00)

        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=2048, hop_length=512)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # print(mel_spec.shape)
        mfccs = (librosa.feature.mfcc(signal, sample_rate, n_mfcc=25, dct_type=3))
        # print(mfccs.shape)
        # ld.specshow(mfccs, sr=sample_rate, x_axis='time')
    #     raw_sounds.append(mfccs)
    # raw_sounds = np.asarray(raw_sounds)
    # print(raw_sounds.shape)
    # print(sample_rate)
    # print(signal.shape)
    return mfccs,signal

sounds_file_original = glob2.glob("/home/shul/audio/originalData/*.wav", recursive=True)
# original_feature, signal = load_sound_files(sounds_file_original[:8000])
# print(original_feature.shape)



X , y = [] , []
for data in tqdm(sounds_file_original):
    sig, sr = librosa.load(data,res_type='kaiser_fast')
    sig_sr=(len(sig))-(sr*2)
    # print(sig)
    for i in range(3):
        n = np.random.randint(0, sig_sr)
        # print(n)
        sig_ = sig[n : int(n+(sr*2))]
        print(sig_.shape)
        mfcc_ = librosa.feature.mfcc(sig_ , sr=sr, n_mfcc=25)
        # print((mfcc_))
        # X.append(mfcc_)
        
        # y.append(data[1][1])
X.append(mfcc_)
X = np.array(X)
# convert list to numpy array
# X = np.asarray(X)
# X = np.asarray(X)
# print(X.shape)    
# y = np.array(y)

#one-hot encoding the target
# y = tf.keras.utils.to_categorical(y , num_classes=10)

# our tensorflow model takes input as (no_of_sample , height , width , channel).
# here X has dimension (no_of_sample , height , width).
# So, the below code will reshape it to (no_of_sample , height , width , 1).
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
print(X.shape)
