
DATADIR = './C:/Office/prac/mozilla_common_speech' # unzipped train and test data
OUTDIR = './C:/Office/prac/model' # just a random name
# Data Loading
import os
import re
from glob import glob


# POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
# id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
# name2id = {name: i for i, name in id2name.items()}


def load_data(data_dir):

    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'C:/Office/prac/mozilla_common_speech/cv-invalid/*wav'))

    # with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
    #     validation_files = fin.readlines()
    # valset = set()
    # for entry in validation_files:
    #     r = re.match(pattern, entry)
    #     if r:
    #         valset.add(r.group(3))

    # possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        print(r)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    

# trainset, valset = load_data(DATADIR)
# print( trainset, valset)
