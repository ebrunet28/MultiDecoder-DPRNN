from glob import glob
from sphfile import SPHFile
import torch
import numpy as np
import pandas as pd
files = list(glob("../egs/**/*.wv1", recursive = True))+list(glob("../egs/**/*.wv2", recursive = True))
files.sort()
np.random.seed(12345)

spkr2files = {}
for filename in files:
    spkr = SPHFile(filename).format['speaker_id']
    if spkr not in spkr2files:
        spkr2files[spkr] = [filename]
    else:
        spkr2files[spkr].append(filename)
for spkr, filenames in spkr2files.items():
    print(spkr, len(filenames))
spkrs = list(spkr2files.keys())
spkrs.sort()
print(len(spkrs))

## trainset
train_csv = []
train_spkrs = spkrs[:100]
for spkr1 in train_spkrs:
    for spkr2 in train_spkrs:
        if spkr1 == spkr2:
            continue
        anchor_indices = np.arange(len(spkr2files[spkr1]))
        positive_indices = np.arange(len(spkr2files[spkr1]))
        negative_indices = np.arange(len(spkr2files[spkr2]))
        np.random.shuffle(anchor_indices)
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        for i in range(min(20, len(positive_indices), len(negative_indices))):
            row = {'anchor_spkr':spkr1, 'other_spkr':spkr2, 'anchor_audio':spkr2files[spkr1][anchor_indices[i]], 
            'positive_audio':spkr2files[spkr1][positive_indices[i]], 'negative_audio':spkr2files[spkr2][negative_indices[i]]}
            train_csv.append(row)
train_csv = pd.DataFrame(train_csv).sample(frac = 1, random_state = 12345)
train_csv.to_csv('../csv/train_triplets.csv')

## testset
test_csv = []
test_spkrs = spkrs[100:]
for spkr1 in test_spkrs:
    for spkr2 in test_spkrs:
        if spkr1 == spkr2:
            continue
        anchor_indices = np.arange(len(spkr2files[spkr1]))
        positive_indices = np.arange(len(spkr2files[spkr1]))
        negative_indices = np.arange(len(spkr2files[spkr2]))
        np.random.shuffle(anchor_indices)
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        for i in range(min(20, len(positive_indices), len(negative_indices))):
            row = {'anchor_spkr':spkr1, 'other_spkr':spkr2, 'anchor_audio':spkr2files[spkr1][anchor_indices[i]], 
            'positive_audio':spkr2files[spkr1][positive_indices[i]], 'negative_audio':spkr2files[spkr2][negative_indices[i]]}
            test_csv.append(row)
test_csv = pd.DataFrame(test_csv).sample(frac = 1, random_state = 12345)
test_csv.to_csv('../csv/test_triplets.csv')

