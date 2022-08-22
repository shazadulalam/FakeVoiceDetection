import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib.pyplot import specgram
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from time import time
from matplotlib import pyplot
# from keras.callbacks import TensorBoard
# from kerastuner.tuners import RandomSearch
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Maxpooling2D

tf.compat.v1.reset_default_graph()


def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    for fp in  file_paths:
        X, sample_rate = librosa.load(fp, res_type='kaiser_fast', duration=3.00)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=64).T,axis=0)
        # new_shape = librosa.feature.melspectrogram(y = X, sr = sample_rate)
        label = fp.split('.wav')

        raw_sounds.append(mfccs)
        # raw_sounds.append(new_shape)
        labels.append(label)
    raw_sounds = np.asarray(raw_sounds)
    labels = np.asarray(labels)
    return raw_sounds, labels


sound_file_paths = glob.glob("/home/grmi/audio/audio0209/wavs/*.wav")

raw_sounds, labels = load_sound_files(sound_file_paths[:8000])
raw_sounds = np.reshape(raw_sounds, (8000, 4,4,4))
# labels = 8000
# print(len(raw_sounds[1]))
# print("##################################################################", raw_sounds.shape)
# print(labels[1:10])

################################Principal component analysis start ##################################

b = []
new_sound = []

for i in range(len(raw_sounds)):
    b = raw_sounds[i]
    b = b.reshape((1, 64))
    new_sound.append(b)

new_sound = np.asarray(new_sound)
new_sound = new_sound.reshape(new_sound.shape[0], 64)
print(new_sound.shape)


pca = PCA(n_components=64)
# pca.fit(raw_sounds)

s = pca.fit_transform(new_sound)
print("________________________________________________", s)

S_ = pca.components_

print("#########################", S_.shape)

x=[]
for i in range(len(S_)):
    ver_vec = S_[i]
    x.append(ver_vec)

x = np.asarray(x)
new_raw_sounds = x.reshape(64,4,4,4)

print(new_raw_sounds.shape)


######################################principal component analysis end here ########################

######################################autoencoder starts from here #################################


sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.01, decay = 1e-6,momentum=0.9, nesterov=False)



encoding_dim = 32

input_audio = tf.keras.layers.Input(shape=(new_raw_sounds.shape[1:]))

# encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_audio)

# decoded = tf.keras.layers.Dense(64, activation='sigmoid')(encoded)

# autoencoder = tf.keras.models.Model(input_audio, decoded)

# autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta')

# autoencoder.fit(new_raw_sounds, epochs=500, batch_size=32, shuffle=True)

#####################################-------------1---------------##################################
#encoder
conv1 = tf.keras.layers.Conv2D(encoding_dim, (4,4), strides=(1,1), activation='relu', padding='same')(input_audio)
pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1)
conv2 = tf.keras.layers.Conv2D(64,(4,4), activation='relu', padding='same',)(pool1)
pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1)
dense1 = tf.keras.layers.Dense(encoding_dim, activation='relu')(pool1)

dense2 = tf.keras.layers.Dense(64, activation='sigmoid')(dense1)

#decoder
conv3 = tf.keras.layers.Conv2D(128,(4,4), activation='relu', padding='same')(dense2)
upsampling1 = tf.keras.layers.UpSampling2D((2,2))(conv3)
conv4 = tf.keras.layers.Conv2D(256, (4,4), activation='relu', padding='same')(upsampling1)
upsampling2 = tf.keras.layers.UpSampling2D((2,2))(conv4)
decode = tf.keras.layers.Conv2D(32, (4,4), activation='sigmoid', padding='same')(upsampling2)

autoencoder = tf.keras.models.Model(input_audio, decode)

autoencoder.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics = ['accuracy'])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time()))

autoencoder.summary()
autoencoder.fit(new_raw_sounds, epochs=1000, batch_size=32, shuffle=True, verbose=1)
# pyplot.plot(history.history['acc'], label='train')
# pyplot.plot(history.history['loss'], label='loss')
# pyplot.legend()
# pyplot.show()
#####################################-------------1---------------##################################

###################################### end here ####################################################
tf.keras.utils.plot_model(
    autoencoder,
    to_file='model1.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB'
)


#########------------------------------------------------------------------------------------
# model = tf.keras.Sequential()
# model.add(layers.Conv2D(64,4,4, input_shape=(new_raw_sounds.shape[1:])))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(128,1,1))
# model.add(layers.Activation('relu'))
# model.add(layers.Dropout(0.25))


# sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.01, decay = 1e-6,momentum=0.9, nesterov=False)
# model.compile(loss =  'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
########-------------------------------------------------------------------------------------
