"""
   filename                             chapter_id speaker_id dataset_id
0  1272/128104/1272-128104-0000.wav     128104       1272  dev-clean
1  1272/128104/1272-128104-0001.wav     128104       1272  dev-clean
2  1272/128104/1272-128104-0002.wav     128104       1272  dev-clean
3  1272/128104/1272-128104-0003.wav     128104       1272  dev-clean
4  1272/128104/1272-128104-0004.wav     128104       1272  dev-clean
5  1272/128104/1272-128104-0005.wav     128104       1272  dev-clean
6  1272/128104/1272-128104-0006.wav     128104       1272  dev-clean
7  1272/128104/1272-128104-0007.wav     128104       1272  dev-clean
8  1272/128104/1272-128104-0008.wav     128104       1272  dev-clean
9  1272/128104/1272-128104-0009.wav     128104       1272  dev-clean
"""

import numpy as np
import pandas as pd

import sid_constants as c
from pre_process import data_catalog

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import sys

class stochastic_mini_batch(Dataset):
    
    def __init__(self,  libri):
       
        self.libri=libri
        self.candidates = {}
        self.speaker_list = libri['speaker_id'].unique()
        
    def __len__(self):
        return len(self.libri)
    
    def __candidates__(self, speaker_id):
        if speaker_id not in self.candidates:
            self.candidates[speaker_id] = self.libri[self.libri['speaker_id'] == speaker_id]
        return self.candidates[speaker_id]
    
    def __pickone__(self, dataframe):
        return dataframe.iloc[random.randrange(len(dataframe))]
    
    def __random_sample__(self, anchor=None, same_as=True):
        if anchor is None:
            return self.__pickone__(self.libri)
        else:
            if same_as:
                can_df = self.__candidates__(anchor.speaker_id)
                while True:
                    candidate = self.__pickone__(can_df)
                    if candidate.filename != anchor.filename:
                        return candidate
            else:
                while True:
                    n_spkr = self.speaker_list[random.randrange(len(self.speaker_list))]
                    if n_spkr != anchor.speaker_id:
                        break
                can_df = self.__candidates__(n_spkr)
                return self.__pickone__(can_df)
    
    def __getitem__(self, index):
        anchor = self.__random_sample__()
        positive = self.__random_sample__(anchor)
        negative = self.__random_sample__(anchor, same_as=False)
        triplet = [anchor, positive, negative]
        data_list = []
        for r in triplet:
            data = np.load(r.filename)
            clipped = clipped_audio(data)
            # print(clipped.shape)
            # sys.exit(0)
            data_tensor = transforms.ToTensor().__call__(clipped)
            # print(data_tensor.shape)
            data_list.append(data_tensor)
         # hint 
        # 1. sample anchor file ,positive file, negative file
        # 2. np.load(...)
        # 3. clipped_audio(...)
        # 4. torch.from_numpy(....transpose ((2, 0, 1)))
        
        return data_list[0], data_list[1], data_list[2], anchor.speaker_id, positive.speaker_id, negative.speaker_id
        # return anchor_file, positive_file, negative_file, anchor_label, positive_label, negative_label



def clipped_audio(x, num_frames=c.NUM_FRAMES):
    # print(x.shape)
    # sys.exit(0)
    if x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x

def main():
    libri = data_catalog()
    #                          filename                                       speaker_id
    #   0    audio/LibriSpeechSamples/train-clean-100-npy/1-100-0001.npy        1
    #   1    audio/LibriSpeechSamples/train-clean-100-npy/1-100-0002.npy        1        
    unique_speakers = libri['speaker_id'].unique() # 251 speaker
    print(libri.head())
    print(unique_speakers)
    dataset = stochastic_mini_batch(libri)
    dataset.__getitem__(0)


if __name__ == '__main__':
    main()

