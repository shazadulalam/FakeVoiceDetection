import os 
import random
import pandas as pd 

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
# from tf.keras.utils import plot_model
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import _pickle as cPickle
import gzip
import math
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib.request, urllib.parse, urllib.error
# experiment = Experiment(api_key="API_KEY",
#                         project_name="customSound")

SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    sr_=[]
    signal = []
    samples_per_segment = int(SAMPLES_PER_TRACK / 10)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / 512)
    data = {
        "mfcc": []
    }
    for fp in  file_paths:
        # features, labels = np.empty((0,193)), np.empty(0)
        # print(features.shape)
        X, sample_rate = librosa.load(fp)
        # print(sample_rate)
        # stft = np.abs(librosa.stft(X))
        mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512).T
        mfcc = mfcc.T
        # if len(mfcc) == num_mfcc_vectors_per_segment:
        #     data["mfcc"].append(mfcc.tolist())

    raw_sounds = np.array(len(mfcc))
    
    print(raw_sounds)
    return raw_sounds

def combine_feature():
    Data_dir = np.array(glob2.glob("/home/prah/audio/training/real/*"))

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

    sounds_file_original = glob2.glob("/home/prah/audio/training/fake/*.wav", recursive=True)
    original_feature = load_sound_files(sounds_file_original[:5000]).reshape(5000, 40)
    df1 = pd.DataFrame(data=original_feature)
    print((df1))

    # norm_original_feature = preprocessing.normalize(original_feature, norm='l2')
    # print((norm_original_feature))
    # original_feature = norm_original_feature.reshape(5000, 8, 5, 1)

    Data_dir2 = np.asarray(glob2.glob("/home/prah/audio/training/fake/*"))
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
    sounds_file_fake = glob2.glob("/home/prah/audio/training/fake/*.wav", recursive=True)
    fake_feature = load_sound_files(sounds_file_fake[:5000]).reshape(5000, 40)
    # norm_fake_feature = preprocessing.normalize(fake_feature, norm='l2')

    # fake_feature = norm_fake_feature.reshape(5000, 8, 5, 1)

    df2 = pd.DataFrame(data=fake_feature)
    print((df2))

    df1['label'] = pd.Series([0 for x in range(len(df1.index))])
    df2['label'] = pd.Series([1 for x in range(len(df2.index))])

    total_feature = np.concatenate((original_feature, fake_feature), axis=0)

    new_data = np.concatenate((original_feature, fake_feature))
    new_label = np.concatenate((label_concoded_original, label_encoded_fake))
    # print(len(new_label))
    shuffle_feature, shuffle_label = shuffle(new_data, new_label)
    # print(shuffle_label)

    X_train = shuffle_feature[0:5000].reshape(5000, 8, 5, 1)
    x_test = shuffle_feature[5000:].reshape(5000, 8, 5, 1)
    Y_train = shuffle_label[0:5000]
    y_test = shuffle_label[5000:]
    
    

    # print(X_train, "______", Y_train)
    print(np.unique(Y_train))
    
    # print("________________________",label_encoded_fake)


    new_x_train = shuffle_feature[0:5000].reshape(5000, 40)
    new_x_test = shuffle_feature[5000:].reshape(5000, 40)
    new_Y_train = shuffle_label[0:5000]
    new_y_test = shuffle_label[5000:]
    new_y_train= tf.keras.utils.to_categorical(Y_train, num_classes=2)
    # print((Y_train.shape))

    return  X_train,Y_train, new_x_train, new_y_train, new_Y_train


#-----------------------previous model--------------------------------------#
def model():

    (X_train, Y_train, new_x_train, new_y_train, new_Y_train) = combine_feature()
    batch_size = 64
    epochs = 100
    nb_layers = 3
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (2, 2),input_shape=(8,5, 1)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(48, (2, 2)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(120, (2, 2)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(32))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Activation('softmax'))

    model.summary()
    sgd = tf.compat.v2.keras.optimizers.Adadelta(lr=0.0002, rho=0.95, epsilon = 1e-06)

    #sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.001, decay = 1e-6, nesterov=False,momentum=0.9)
    adam = tf.compat.v2.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss =  'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    history = model.fit(X_train, Y_train, validation_split=0.1, shuffle=True, verbose = 2, batch_size=batch_size, epochs = epochs)


    # n_estimators = [i for i in range(50,150,10)]
    # max_features = ['auto', 'sqrt', 'log2']
    # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # min_samples_split = [2, 5, 10]
    # min_samples_leaf = [1, 2, 4]
    # bootstrap = [True, False]
    # rfc = RandomForestRegressor() 

    # param_grid = { 
    #     'n_estimators': n_estimators,
    #     'max_features': max_features,
    #     'max_depth': max_depth,
    #     'min_samples_split': min_samples_split,
    #     'min_samples_leaf': min_samples_leaf,
    #     'bootstrap': bootstrap
        
    # }

    # CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    # CV_rfc.fit(new_x_train, new_Y_train)
    # print(CV_rfc.best_params_) 
    print(history.history.keys())
    
    # model.save("my_model.h5")
    # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

   # random_search = {'criterion': ['entropy', 'gini'],
               #'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],
               #'max_features': ['auto', 'sqrt','log2', None],
               #'min_samples_leaf': [4, 6, 8, 12],
               #'min_samples_split': [5, 7, 10, 14],
               #'n_estimators': list(np.linspace(151, 1200, 10, dtype = int))}

    #clf = RandomForestClassifier()
    #model.fit(new_x_train,new_y_train)

    #predictionforest = model.best_estimator_.predict(new_x_train)
    # print(confusion_matrix(new_y_train,predictionforest))
    #print(classification_report(new_y_train,predictionforest))
    #acc3 = accuracy_score(new_y_train,predictionforest)
    
    return model

model()


# n_estimators = [i for i in range(50,150,10)]
# max_features = ['auto', 'sqrt', 'log2']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False
# rfc = RandomForestRegressor() 

# param_grid = { 
#     'n_estimators': n_estimators,
#     'max_features': max_features,
#     'max_depth': max_depth,
#     'min_samples_split': min_samples_split,
#     'min_samples_leaf': min_samples_leaf,
#     'bootstrap': bootstrap
    
# }

# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# CV_rfc.fit(new_x_train, new_Y_train)
# print(CV_rfc.best_params_) 
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

#_____________________________________________________#
# from sklearn.model_selection import cross_val_score

# rf_reg = RandomForestRegressor(n_estimators=20,max_features='sqrt')
# cv_score = cross_val_score(rf_reg, new_x_train, new_Y_train, scoring='r2', cv=10)

# print(cv_score) 
# print("Mean: ", cv_score.mean())

#_____________________________________________________#

#-----------------------------------------------------------------------------#
#random forest
# hyp_parameters = {
#     "random_state" : [0],
#     "n_estimators" : [100, 1000],
#     "max_depth" : [None, 2, 4],
#     "max_features" : ['auto', 'sqrt']
# }

# config_count = 0
# total_config = 20
# max_f1  = 0

# for config in ParameterGrid(hyp_parameters):
#     config_count += 1
#     print(f'Analizing config {config_count} of {total_config} || Config: {config}')

#     rfc = RandomForestClassifier(**config)
#     rfc.fit(new_x_train, new_Y_train)
#     y_test_pred = rfc.predict(new_x_test)
#     acc = accuracy_score(y_test_pred, new_y_test)
#     p1, r1, f11, s1 = precision_recall_fscore_support(new_y_test, y_test_pred)
#     marco_f1 = f11.mean()

#     if marco_f1 > max_f1:
#         max_f1 = marco_f1
#         print(f"-->Score: {marco_f1}")

#####################################-------------1---------------##################################
# encoding_dim = 32

# input_audio = tf.keras.layers.Input(shape=(shuffle_feature.shape[1:]))

# #encoder
# conv1 = tf.keras.layers.Conv2D(encoding_dim, (5, 5), strides=(1,1), activation='relu', padding='same')(input_audio)
# pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1)
# # conv2 = tf.keras.layers.Conv2D(64,(4,4), activation='relu', padding='same',)(pool1)
# # pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1)
# dense1 = tf.keras.layers.Dense(encoding_dim, activation='relu')(pool1)

# dense2 = tf.keras.layers.Dense(64, activation='sigmoid')(dense1)

# #decoder
# conv3 = tf.keras.layers.Conv2D(128,(5, 5), activation='relu', padding='same')(dense2)
# upsampling1 = tf.keras.layers.UpSampling2D((2,2))(conv3)
# conv4 = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same')(upsampling1)
# upsampling2 = tf.keras.layers.UpSampling2D((2,2))(conv4)
# decode = tf.keras.layers.Conv2D(1, (5, 5), activation='sigmoid', padding='same')(upsampling2)

# autoencoder = tf.keras.models.Model(input_audio, decode)

# autoencoder.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics = ['accuracy'])

# tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time()))

# autoencoder.summary()
# autoencoder.fit(X_train, Y_train, epochs=10, batch_size=32, shuffle=True, verbose=1)

#####################################-------------1---------------##################################