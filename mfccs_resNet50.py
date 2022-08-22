
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
from sklearn.model_selection import GridSearchCV, KFold
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib.request, urllib.parse, urllib.error
from sklearn.model_selection import cross_val_score
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
# from tensorflow.keras.applications.vgg19 import VGG19

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



batch_size = 64
pool_size = 2
IMAGE_SIZE = [224, 224]


Data_dir = '/home/shul/audio/FoR/for-rerecorded/training/images'
val_dir = '/home/shul/audio/FoR/for-rerecorded/validation/images'
test_dir = '/home/shul/audio/FoR/for-rerecorded/testing/images'


def load_files(file_path):
    array=[]
    for folder in os.listdir(file_path):

        sub_path=file_path+"/"+folder
        for img in os.listdir(sub_path):
            image_path=sub_path+"/"+img
            img_arr=cv2.imread(image_path)
            img_arr=cv2.resize(img_arr,(224,224))
            array.append(img_arr)

    result=np.asarray(array)
    result = result/255.0
    return result

train_x = load_files(Data_dir)
test_x = load_files(test_dir)
val_x = load_files(val_dir)
print(len(train_x))
datagen = ImageDataGenerator(
                    rescale=1./255)
# datagen = ImageDataGenerator(
#                     rescale=1./255,
#                     validation_split = 0.2)
train_set = datagen.flow_from_directory(
    Data_dir,
    target_size=(224,224),
    batch_size = batch_size,
    shuffle=True,
    class_mode = 'sparse'
)

test_set = datagen.flow_from_directory(
    Data_dir,
    target_size=(224,224),
    batch_size = batch_size,
    shuffle=True,
    class_mode = 'sparse'
)

val_set = datagen.flow_from_directory(
    Data_dir,
    target_size=(224,224),
    batch_size = batch_size,
    shuffle=True,
    class_mode = 'sparse'
)

train_y = np.array(train_set.classes)
test_y = np.array(test_set.classes)
val_y = np.array(val_set.classes)
print(len(train_y))


base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=IMAGE_SIZE + [3],
    pooling=None,
    classes=2
)
for layer in base_model.layers[0:100]:
    layer.trainable = True
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
print("____________________",x.shape)
adam = tf.keras.optimizers.Adam(lr=0.0001)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# prediction = Dense(2, activation='softmax')(x)
prediction = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction)
model.summary()

for layer in base_model.layers:
    layer.trainable = False
# training
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_set, epochs=10, batch_size=batch_size, validation_data=test_set, verbose=1)

score = model.evaluate(x_test, y_test)
print('score', score)
print('Test score:', score[0])
print('Test accuracy:', score[1])


#_________________________________________#
# y_pred = model.predict(x_train, verbose=2) # take prediction 
# confusion = tf.math.confusion_matrix(
#               labels = Y_train.reshape(-1),             # get trule labels 
#               predictions = np.argmax(y_pred, axis=1),  # get predicted labels
#               )  
# print(confusion)
# import seaborn as sns 
# import pandas as pd 

# cm = pd.DataFrame(confusion.numpy(), # use .numpy(), because now confusion is tensor
#                range(num_of_classess),range(num_of_classess))

# plt.figure(figsize = (10,10))
# sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
# plt.savefig('/home/shul/code/plot_image/confusion_tf.png', dpi=100)

#_________________________________________#
# Accuracy
# print(history.history.keys())
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('accuracy vs loss')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['acc',  'loss'], loc='center right')
# plt.savefig('/home/shul/code/plot_image/acc_loss_tf_1000epoch.png', dpi=100)
# plt.show()


