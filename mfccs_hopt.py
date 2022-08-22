
import os 
import random
import pandas as pd 
import glob2
import numpy as np 
from numpy import argmax, mean, std
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
from timeit import default_timer as timer
from scipy.io import wavfile as wav
from matplotlib.pyplot import specgram
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from tune_sklearn import TuneSearchCV

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import _pickle as cPickle
import gzip
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib.request, urllib.parse, urllib.error
from sklearn.model_selection import cross_val_score



def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    sr_=[]
    signal = []
    for fp in  file_paths:
        X, sample_rate = librosa.load(fp)
        # print(sample_rate)
        # stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=16).T,axis=0)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=0)
        # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        raw_sounds.append(mfccs)
        sr_.append(sample_rate)
        signal.append(X)
    raw_sounds = np.array(raw_sounds)
    sr_ = np.array(sr_)
    signal = np.array(signal)
    # print(raw_sounds.shape())
    return raw_sounds


#train test split
def split_train_test(dataframe, target):
    train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)
    X_train, X_test = train_set.drop(target, axis=1), test_set.drop(target, axis=1)
    y_train, y_test = train_set[target], test_set[target]
    
    return X_train, X_test, y_train, y_test


def model():

    Data_dir = (load_sound_files(np.array(glob2.glob("/home/shul/audio/FoR/for-rerecorded/training/real/*"))))

    original_class = []
    name = []
    labelName1 = []
    original_feature = []
    i = 0
    for file in Data_dir[i : i + 5000]:


        name.append(0)
    # labelName = np.asarray(original_class)
    original_class = np.asarray(name)



    label_encoder = LabelEncoder()
    label_concoded_original = label_encoder.fit_transform(original_class)

    sounds_file_original = glob2.glob("/home/shul/audio/FoR/for-rerecorded/training/real/*.wav", recursive=True)
    original_feature = load_sound_files(sounds_file_original[:5000])
    norm_original_feature = preprocessing.normalize(original_feature, norm='l2')
    print((norm_original_feature))
    original_feature = norm_original_feature.reshape(5000, 16)



    Data_dir2 = np.asarray(glob2.glob("/home/shul/audio/FoR/for-rerecorded/training/fake/*"))
    fake_class = []
    label_name2 = []
    fake_feature = []
    j = 0

    for file in Data_dir2[j: j + 5000]:
        fake_class.append(1)

    fake_class = np.asarray(fake_class)

    # print(fake_class)
    label_encoder2 = LabelEncoder()
    label_encoded_fake = label_encoder2.fit_transform(fake_class)
    sounds_file_fake = glob2.glob("/home/shul/audio/FoR/for-rerecorded/training/fake/*.wav", recursive=True)
    fake_feature = load_sound_files(sounds_file_fake[:5000])
    norm_fake_feature = preprocessing.normalize(fake_feature, norm='l2')

    fake_feature = norm_fake_feature.reshape(5000, 16)


    df1 = pd.DataFrame(data=original_feature)
    df2 = pd.DataFrame(data=fake_feature)

    df1['label'] = pd.Series([0 for x in range(len(df1.index))])
    df2['label'] = pd.Series([1 for x in range(len(df2.index))])



    #append dataframes
    final_data = df1.append(df2, ignore_index=True)


    #shuffle rows
    final_data = final_data.sample(frac = 1)


    X_train, X_test, y_train, y_test = (split_train_test(final_data, 'label'))


    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



    start = 0
    x_train = []
    x_test = []
    Y_train = []
    Y_test = []
    for i in range(0, len(X_train)):
        result = X_train.values.reshape(8000, 4, 4, 1)
    for i in range(0, len(X_test)):
        test = X_test.values.reshape(2000, 4, 4, 1)
    for i in range(0, len(X_train)):
        ytrain = y_train.values.reshape(8000)
    for i in range(0, len(X_train)):
        ytest = y_test.values.reshape(2000)

    x_train = np.array(result)
    x_test = np.array(test)
    Y_train = np.array(ytrain)
    Y_test = np.array(ytest)
    newx_train = x_train.reshape(8000, 16)
    newy_train = Y_train

    batch_size = 64
    epochs = 200
    nb_layers = 3
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (2, 2),input_shape=(4, 4, 1)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(120, (2, 2)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(32))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(64))
    # model.add(layers.Activation('relu'))
    # model.add(layers.Dropout(0.4))
    model.add(layers.Activation('softmax'))



    model.summary()
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss =  'sparse_categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    model.fit(x_train, Y_train, validation_split=0.1, shuffle=True, verbose = 1, batch_size=batch_size, epochs = epochs)
    
    
    # return model, x_train, Y_train

    space = {
            "n_estimators" : hp.choice("n_estimators", [100,200,300,400]),
            "max_depth" : hp.quniform("max_depth", 1, 15, 1),
            "critetion" : hp.choice("criterion", [ "entropy"]),
        }

    # def hyperparameter_tuning(params):
    #     model, x_train, Y_train = model()
    #     acc = cross_val_score(model, x_train, Y_train, scoring="accuracy").mean()
    #     return {"loss": acc, "status":STATUS_OK}

    trials = Trials()

    best = fmin(
        fn = model,
        space = space,
        algo = tpe.suggest,
        max_evals = 100,
        trials = trials
    )

    print(best)
    print(trials.result)

model()