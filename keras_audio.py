import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib.pyplot import specgram
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
# from kerastuner.tuners import RandomSearch
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Maxpooling2D

tf.compat.v1.reset_default_graph()


def load_sound_files(file_paths):
    raw_sounds = []
    labels = []
    for fp in  file_paths:
        X, sample_rate = librosa.load(fp, res_type='kaiser_fast', duration=3.00)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=64).T, axis=0)

        # print(mfccs)
        # new_shape = librosa.feature.melspectrogram(y = X, sr = sample_rate)
        # plt.figure(figsize=(10,4))
        # librosa.display.specshow(mfccs, x_axis='time')
        # plt.colorbar()
        # plt.title('MFSS')
        # plt.tight_layout()
        # plt.show()

        label = fp.split('.wav')

        raw_sounds.append(mfccs)
        # raw_sounds.append(new_shape)
        labels.append(label)
    raw_sounds = np.asarray(raw_sounds)
    # print(raw_sounds.shape)
    labels = np.asarray(labels)
    return raw_sounds, labels


sound_file_paths = glob.glob("/home/shul/audio/wavs/*.wav", recursive=True)

raw_sounds, labels = load_sound_files(sound_file_paths[:8000])
raw_sounds = np.reshape(raw_sounds, (8000, 4, 4, 4))
# labels = labels.reshape(labels, (8000,4,4,4))
print(raw_sounds.shape[1:])
# label_data=[]
# for i in labels:
#     path= os.remove(i)
#     label_data.append(path)
# label_data = np.asarray(labels)
# print(label_data)


# X_train, X_test, y_train, y_test = train_test_split(raw_sounds, labels, test_size = 0.2)
X_train = np.asarray(raw_sounds[0:6000])
X_test = np.asarray(raw_sounds[6000:])
y_train = X_train.shape[0:1]
y_test = X_test.shape[0:1]
print(X_train.shape[0:1], "_________", X_test.shape[0:1])
print(y_train, "_________", y_test)
batch_size = 32

# num_classes = 16

# transform = LabelEncoder()
# X_train = transform.fit(X_train)
# X_test = transform.transform(X_test)


model = tf.keras.Sequential()
model.add(layers.Conv2D(32, 4, 4,input_shape=(4,4,4)))
model.add(layers.Activation('relu'))
# model.add(layers.MaxPool2D())
# model.add(layers.Conv2D(128, 4, 4))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(128,1,1))
# model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))

sgd = tf.compat.v2.keras.optimizers.SGD(lr=0.01, decay = 1e-6,momentum=0.9, nesterov=False)
model.compile(loss =  'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])



model.fit(X_train, steps_per_epoch=6000,batch_size=batch_size, epochs = 100)
# model.save('audioDaTa.model')
y_pred = model.predict(X_test)
model.summary()