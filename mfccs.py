
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
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from timeit import default_timer as timer
from scipy.io import wavfile as wav
from matplotlib.pyplot import specgram
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing
import warnings
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
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib.request, urllib.parse, urllib.error
from sklearn.model_selection import cross_val_score
import torch
from tensorflow.keras.utils import plot_model

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)

def warn(*args, **kwargs):
    pass
warnings.warn = warn



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
        # contrast = np.mean(librosa.feature.spectral_contrast(S=stft,
            # sr=sample_rate).T,axis=0)
        # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
            # sr=sample_rate).T,axis=0)
        # ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        # print(ext_features.shape)
        # features = np.vstack([features,ext_features])
        # ld.specshow(mfccs, sr=sample_rate, x_axis='time')
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
print(len(sounds_file_original))
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
print(df1)


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

batch_size = 16
epochs = 100
# nb_layers = 3
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
# sgd = tf.compat.v2.keras.optimizers.Adadelta(lr=0.0002, rho=0.95, epsilon = 1e-06)

# sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.001, decay = 1e-6, nesterov=False,momentum=0.9)
adam = tf.keras.optimizers.Adam(lr=0.01)
model.compile(loss =  'sparse_categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

model.fit(x_train, Y_train, validation_split=0.1, shuffle=True, verbose = 1, batch_size=batch_size, epochs = epochs)
# #tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True)

# model.save('/home/shul/code/my_model/model_for_tl.h5')
# print(history.history.keys())
# #  "Accuracy"
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['loss'])
# plt.title('accuracy vs loss')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# # plt.savefig('/home/shul/code/plot_image/accvsloss.png', dpi=100)
# plt.show()







# "Loss"
# plt.plot(history.history['val_accuracy'])
# plt.plot(history.history['val_loss'])
# plt.title('val accuracy vs val loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.savefig('val_acc vs valloss.png', dpi=100)
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
    #______________________Grid Search________________________#

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
    # CV_rfc.fit(newx_train, newy_train)
    # print(CV_rfc.best_params_) 

    #__________________________________________________________#


    #_____________k-fold cross validation__________________#

    # rf_reg = RandomForestRegressor(bootstrap = False, max_depth = 40, min_samples_split = 2, n_estimators=20,max_features='sqrt')
    # cv_score = cross_val_score(rf_reg, newx_train, newy_train, scoring='r2', cv=10)

    # print(cv_score) 
    # print("Mean: ", cv_score.mean())

    #_____________________________________________________#

    #___________________random forest search______________#

    # random_search = {'criterion': ['entropy', 'gini'],
    #             'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],
    #             'max_features': ['auto', 'sqrt','log2', None],
    #             'min_samples_leaf': [4, 6, 8, 12],
    #             'min_samples_split': [5, 7, 10, 14],
    #             'n_estimators': list(np.linspace(151, 1200, 10, dtype = int))}

    # clf = RandomForestClassifier()
    # model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 80, 
    #                             cv = 4, verbose= 5, random_state= 101, n_jobs = -1)
    # model.fit(newx_train,newy_train)

    # predictionforest = model.best_estimator_.predict(newx_train)
    # # print(confusion_matrix(newy_train,predictionforest))
    # print(classification_report(newy_train,predictionforest))
    # acc3 = accuracy_score(newy_train,predictionforest)

    # print(acc3)

    # start = timer()

    # print("total running time: ", timer() - start)

    #______________________________________________________#

    #____________________k-fold ___________________________#

    # cv = KFold(n_splits=3, random_state=1, shuffle=True)
    
    # #regmodel = LogisticRegression()
    
    # scores = cross_val_score(model, newx_train, newy_train, scoring='accuracy', cv=cv, n_jobs=-1)
    
    # print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    #______________________________________________________#

#model()