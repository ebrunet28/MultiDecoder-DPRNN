import numpy as np
import torch
import torch.utils.data as data

import librosa
from sphfile import SPHFile

class MixtureDataset(data.Dataset):
    def __init__(self, txtfile, sample_rate=8000, segment=4.0, cv_maxlen=8.0):
        self.specs = np.loadtxt(txtfile, dtype = str)
    def __len__(self):
        return len(self.specs)
    def __getitem__(self, idx):
        spec = self.specs(idx)
        num_speakers = len(spec)//2
        input = 
        for i in range(num_speakers):


if __name__ == "__main__":
    dataset = MixtureDataset("../csv/mixtures/mix_2_spk_tr.txt", 0)
    print(dataset.mixtures[0])
    print(len(dataset))