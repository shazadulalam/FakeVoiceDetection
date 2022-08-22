
import seaborn as sns
import pandas as pd 
import glob2
import numpy as np 
from numpy import argmax, mean, std
import librosa 
import librosa.display as ld
import pylab 
import matplotlib.pyplot as plt 
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import _pickle as cPickle
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib.request, urllib.parse, urllib.error
from sklearn.model_selection import cross_val_score
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AddBackgroundNoise

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
#     print("Running on the GPU")
# else:
#     device = torch.device("cpu")
#     print("Running on the CPU")

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)


# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
# sess = tf.compat.v1.Session(config=config) 
# tf.compat.v1.keras.backend.set_session(sess)

def warn(*args, **kwargs):
    pass
warnings.warn = warn

filters = 64
pool_size = 2
kernel_size = (2, 2)

input_shape = (4, 4, 1)

augment = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
])



def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    sr_=[]
    signal = []
    for fp in  file_paths:
        
        X, sample_rate = librosa.load(fp)
        # print(sample_rate)
        augmented_samples = augment(samples=X, sample_rate=sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=augmented_samples, sr=sample_rate, n_mfcc=16).T,axis=0)
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
        # sr_.append(sample_rate)
    raw_sounds = np.array(raw_sounds)
    # sr_ = np.array(sr_)
    # signal = np.array(signal)
    # print(raw_sounds.shape())
    return raw_sounds


#train test split
def split_train_test(dataframe, target):
    train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)
    X_train, X_test = train_set.drop(target, axis=1), test_set.drop(target, axis=1)
    y_train, y_test = train_set[target], test_set[target]
    
    return X_train, X_test, y_train, y_test

def train_model(model, train, test, classes):

    adam = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(loss =  'sparse_categorical_crossentropy', optimizer = adam, metrics = ['accuracy']) 

    history = model.fit(x_train, y_train, validation_split=0.1, shuffle=True, verbose = 1, batch_size=batch_size, epochs = epochs, validation_data = (x_test, y_test))

    score = model.evaluate(x_test, y_test)
    print('score', score)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # y_pred = model.predict(x_train, verbose=2) # take prediction 
    # confusion = tf.math.confusion_matrix(
    #             labels = Y_train.reshape(-1),             # get true labels 
    #             predictions = np.argmax(y_pred, axis=1),  # get predicted labels
    #             )  
    # print(confusion)
    

    # cm = pd.DataFrame(confusion.numpy(), # use .numpy(), because now confusion is tensor
    #             range(num_of_classess),range(num_of_classess))

    # plt.figure(figsize = (10,10))
    # ax = sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
    # ax.set_title('Confusion Matrix with labels\n\n')
    # ax.set_xlabel('\nPredicted Values')
    # ax.set_ylabel('Actual Values ')

    # ## Ticket labels - List must be in alphabetical order
    # ax.xaxis.set_ticklabels(['Fake','Original'])
    # ax.yaxis.set_ticklabels(['Fake','Original'])
    # plt.savefig('/home/shul/code/plot_image/confusion_tf_labels_1.png', dpi=100)

    #  "Accuracy"
    # plt.plot(history.history['accuracy'])
    # # plt.plot(history.history['loss'])
    # plt.title('Model Accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['acc'], loc='center right', bbox_to_anchor=(1, 0.5),fancybox = True, shadow = True)
    # plt.savefig('/home/shul/code/plot_image/accvsloss_after_freeze.png', dpi=100)
    # plt.show()


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
for i in range(0, len(y_train)):
    ytrain = y_train.values.reshape(8000)
for i in range(0, len(y_test)):
    ytest = y_test.values.reshape(2000)

x_train = np.array(result)
x_test = np.array(test)
Y_train = np.array(ytrain)
Y_test = np.array(ytest)
newx_train = x_train.reshape(8000, 16)
newy_train = Y_train

batch_size = 16
epochs = 100
nb_layers = 3
num_of_classess = 2
feature_layers = [
    layers.Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    layers.Activation('relu'),
    layers.Conv2D(120, (2, 2)),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    layers.Flatten(),
]

classification_layers = [
    layers.Dense(128),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(64),
    layers.Activation('softmax')
]

model = tf.keras.Sequential(feature_layers + classification_layers)

adam = tf.keras.optimizers.Adam(lr=0.01)
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = adam, metrics = ['accuracy']) 
model.summary()
history = model.fit(x_train, Y_train, validation_split=0.1, shuffle=True, verbose = 1, batch_size=batch_size, epochs = epochs, validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test)
print('score', score)
print('Test score:', score[0])
print('Test accuracy:', score[1])


#_________________________________________#
y_pred = model.predict(x_train, verbose=2) # take prediction 
confusion = tf.math.confusion_matrix(
              labels = Y_train.reshape(-1),             # get trule labels 
              predictions = np.argmax(y_pred, axis=1),  # get predicted labels
              )  
# print(confusion)
# import seaborn as sns 
# import pandas as pd 

cm = pd.DataFrame(confusion.numpy(), # use .numpy(), because now confusion is tensor
               range(num_of_classess),range(num_of_classess))

# plt.figure(figsize = (10,10))
# ax = sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size

# ax.set_title('Confusion Matrix with labels\n\n')
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ')

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['Fake','Original'])
# ax.yaxis.set_ticklabels(['Fake','Original'])
# plt.savefig('/home/shul/code/plot_image/confusion_tf_labels.png', dpi=100)

#_________________________________________#
# Accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc'], loc='center right')
plt.savefig('/home/shul/code/plot_image/distortion_acc_transferL_100epoch.png', dpi=100)
plt.show()

# history = train_model(model, (x_train, Y_train), (x_test, y_test), 2)

for l in feature_layers:
    l.trainable = False

# train_model(model, (x_train, Y_train), (x_test, y_test), 2)

