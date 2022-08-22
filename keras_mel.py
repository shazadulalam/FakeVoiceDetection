import glob
import os
from tqdm import tqdm
import cv2
import librosa
from pathlib import Path, PureWIndowsPath
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib.pyplot import specgram

DATADIR =Path("/home/shul/audio/")
datadir = PureWIndowsPath(DATADIR)

print(datadir)
CATEGORIES = ["originalData", "fakeData"]

def training_audio_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        print(path)
        for audion in tqdm(os.listdir(path)):
            try:
                X, sample_rate = cv2.imread(os.path.join(path, audio))
                # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=64).T, axis=0)
                print(x)
            except Exception as e:
                pass

training_audio_data()