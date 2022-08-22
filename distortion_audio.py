from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import pandas as pd 
# from memory_profiler import memory_usage
from glob import glob
import numpy as np 
from numpy import argmax
import librosa 
import librosa.display
import pylab 
import matplotlib.pyplot as plt 
from matplotlib import figure
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000

augment = Compose([
    # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    # PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])


# def create_distortion(filename, name):
filename = "/home/shul/audio/FoR/for-rerecorded/training/real/recording12465.wav_norm_mono.wav"
clip, sample_rate = librosa.load(filename, sr=None)

# Augment/transform/perturb the audio data
augmented_samples = augment(samples=clip, sample_rate=SAMPLE_RATE)



    # return  filename, name, augmented_samples

# Data_dir = np.array(glob("/home/shul/audio/FoR/for-rerecorded/validation/real/*"))

# i = 0
# for file in Data_dir[i : i + 5100]:

#     filename, name = file, file.split('/')[-1].split('.') [0]
#     create_distortion(filename, name)

