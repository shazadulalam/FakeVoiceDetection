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

from pydub import AudioSegment
from pydub.utils import make_chunks
# import pickle
# experiment = Experiment(api_key="API_KEY",
#                         project_name="customSound")


# raw_sounds = []
# labels = []
# sr_=[]
# signal = []
# i = 0
# X, sample_rate = librosa.load('/home/forhad/Study/office/townhall_new (95).wav')
# print(len(X))
# while True:
#     if(len(X)) < i * sample_rate:
#         break
#     else:
#         raw_sounds.append(X[i * sample_rate : (i + 1) * sample_rate])
        
#         raw_sounds = np.array(raw_sounds)
    
# print(raw_sounds)

# Data_dir = np.array(glob2.glob("/home/shul/audio/originalData/*"))



# sounds_file_original = glob2.glob("/home/shul/audio/originalData/*.wav", recursive=True)
# original_feature = load_sound_files(sounds_file_original[:8000])



myaudio = AudioSegment.from_file("/home/shul/audio/originalData/townhall_new (95).wav", "wav") 
chunk_length_ms = 1000 
chunks = make_chunks(myaudio, chunk_length_ms) 

#Export individual seconds
for i, chunk in enumerate(chunks):
    chunk_name = "chunk1{0}.wav".format(i)
    print ("exporting", chunk_name)
    chunk.export(chunk_name, format="wav")
