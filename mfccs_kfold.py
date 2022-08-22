
import os 
import random
from matplotlib import pyplot
import pandas as pd 
import glob2
import numpy as np 
from numpy import argmax, mean, std
import librosa 
import librosa.display as ld
import keras 
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
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, ParameterGrid, LeaveOneOut, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import tensorflow.compat.v1 as tf2
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
from sklearn.metrics import classification_report
import _pickle as cPickle
import gzip
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib.request, urllib.parse, urllib.error
from sklearn.model_selection import cross_val_score

# config = tf2.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
# sess = tf2.Session(config=config) 
# tf.compat.v1.keras.backend.set_session(sess)

def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    sr_=[]
    signal = []
    for fp in  file_paths:
        # features, labels = np.empty((0,193)), np.empty(0)
        # print(features.shape)
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




def get_dataset():

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

    # label_name2 = np.asarray(fake_class)
    fake_class = np.asarray(fake_class)

    # print(fake_class)
    label_encoder2 = LabelEncoder()
    label_encoded_fake = label_encoder2.fit_transform(fake_class)
    sounds_file_fake = glob2.glob("/home/shul/audio/FoR/for-rerecorded/training/fake/*.wav", recursive=True)
    fake_feature = load_sound_files(sounds_file_fake[:5000])
    norm_fake_feature = preprocessing.normalize(fake_feature, norm='l2')

    fake_feature = norm_fake_feature.reshape(5000, 16)



    # total_feature = np.concatenate((original_feature, fake_feature), axis=0)

    # new_data = np.concatenate((original_feature, fake_feature))
    # new_label = np.concatenate((label_concoded_original, label_encoded_fake))
    # print(len(new_label))
    # shuffle_feature, shuffle_label = shuffle(new_data, new_label)
    # print(shuffle_label)


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



    # X_train = [X_train.values.reshape(8000, 8, 5, 1) for i in range(0, len(X_train))]


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

    return x_train, x_test, Y_train, Y_test, newx_train, newy_train

def get_model():

    x_train, x_test, Y_train, Y_test, newx_train, newy_train = get_dataset()
    batch_size = 64
    epochs = 150
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cvscore = []

    #_________________________kfold approach_________________________#

    for train, test in kfold.split(x_train, Y_train):

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


        # model.summary()
        # sgd = tf.compat.v2.keras.optimizers.Adadelta(lr=0.0002, rho=0.95, epsilon = 1e-06)

        # sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.001, decay = 1e-6, nesterov=False,momentum=0.9)
        adam = tf.keras.optimizers.Adam(lr=0.01)
        model.compile(loss =  'sparse_categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

        model.fit(x_train[train], Y_train[train], validation_split=0.1, shuffle=True, verbose = 0, batch_size=batch_size, epochs = epochs)

        scores = model.evaluate(x_train[test], Y_train[test], verbose = 0)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        cvscore.append(scores[1] * 100)

        model.save('my_model/mfcc_kfold.h5')

    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscore), np.std(cvscore)))


    
    #_________________________kfold approach end_________________________#

    #_________________________kfold different approach_________________________#

    # model = tf.keras.Sequential()
    # model.add(layers.Conv2D(64, (2, 2),input_shape=(4, 4, 1)))
    # model.add(layers.Activation('relu'))
    # model.add(layers.Conv2D(120, (2, 2)))
    # model.add(layers.Activation('relu'))
    # model.add(layers.MaxPool2D(2, 2))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(32))
    # model.add(layers.Activation('relu'))
    # model.add(layers.Dropout(0.25))
    # # model.add(layers.Dense(64))
    # # model.add(layers.Activation('relu'))
    # # model.add(layers.Dropout(0.4))
    # model.add(layers.Activation('softmax'))

    # # model.summary()
    # # sgd = tf.compat.v2.keras.optimizers.Adadelta(lr=0.0002, rho=0.95, epsilon = 1e-06)

    # # sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.001, decay = 1e-6, nesterov=False,momentum=0.9)
    # adam = tf.keras.optimizers.Adam(lr=0.01)
    # model.compile(loss =  'sparse_categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model

get_model()

# def evaluate_model(cv):

#     batch_size = 64
#     epochs = 150
#     x_train, x_test, Y_train, Y_test, newx_train, newy_train = get_dataset()
#     model = get_model() 
#     model.fit(x_train, Y_train, validation_split=0.1, shuffle=True, verbose = 1, batch_size=batch_size, epochs = epochs)
#     scores = cross_val_score(model, newx_train, newy_train, scoring='accuracy', cv = cv, verbose = 1, n_jobs=-1)

#     return mean(scores)

# ideal, _, _= evaluate_model(LeaveOneOut())
# print('Ideal: %.3f' % ideal)
# folds = range(2,10)
# means, mins, maxs = list(), list(), list()
# for k in folds:    
#     cv = KFold(n_splits=k, shuffle=True, random_state=1)
#     k_mean, k_min, k_max =evaluate_model(cv)
#     print('> folds = %d, accuracy = %.3f (%.3f, %.3f)')
#     means.append(k_mean)
#     mins.append(k_mean - k_min)
#     maxs.append(k_max - k_mean)

# pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
# # plot the ideal case in a separate color
# pyplot.plot(folds, [ideal for _ in range(len(folds))], color='r')
# # show the plot
# pyplot.show()

#_________________________kfold different approach end_________________________#