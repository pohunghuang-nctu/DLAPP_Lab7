
# train-clean-100: 251 speaker, 28539 utterance
# train-clean-360: 921 speaker, 104104 utterance
# test-clean: 40 speaker, 2620 utterance
# merged test: 80 speaker, 5323 utterance
# batchisize 32*3 : train on triplet: 5s - > 3.1s/steps , softmax pre train: 3.1 s/steps


import os
import sys
import numpy as np
import random

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchsummary import summary

import sid_constants as c
from pre_process import data_catalog, preprocess_and_save
from random_batch import stochastic_mini_batch
from model import DeepSpeakerModel,TripletMarginLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_dict(files,labels,spk_uniq):
    train_dict = {}
    for i in range(len(spk_uniq)):
        train_dict[spk_uniq[i]] = []
    # train_dict[19]=[] train_dict[26]=[]
    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])
    # train_dict={19: ['audio/LibriSpeechSamples/train-clean-100-npy/19-150-0001.npy', 'audio/LibriSpeechSamples/train-clean-100-npy/19-150-0001.npy'],
    #             26: ['audio/LibriSpeechSamples/train-clean-100-npy/26-100-0001.npy']}
    for spk in spk_uniq:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)
    # 26 pop out
    # train_dict={19: ['audio/LibriSpeechSamples/train-clean-100-npy/19-150-0001.npy', 'audio/LibriSpeechSamples/train-clean-100-npy/19-150-0001.npy']}
    unique_speakers=list(train_dict.keys())
    # unique_speakers=[19]
    return train_dict, unique_speakers

def main(libri_dir=c.DATASET_DIR):


    print('Looking for fbank features [.npy] files in {}.'.format(libri_dir))
    libri = data_catalog(libri_dir)
    #                          filename                                       speaker_id
    #   0    audio/LibriSpeechSamples/train-clean-100-npy/1-100-0001.npy        1
    #   1    audio/LibriSpeechSamples/train-clean-100-npy/1-100-0002.npy        1        
    unique_speakers = libri['speaker_id'].unique() # 251 speaker
    transform=transforms.Compose([transforms.ToTensor()])
                                               
    train_dir = stochastic_mini_batch(libri)
    train_loader = DataLoader(train_dir, batch_size=c.BATCH_SIZE, shuffle=True)
    model = DeepSpeakerModel(embedding_size=c.EMBEDDING_SIZE,num_classes=c.NUM_SPEAKERS)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    epoch = 0
    model.cuda()
    summary(model, input_size=(1, 160, 64))
    for epoch in range(100):       
        model.train()
      
        for batch_idx, (data_a, data_p, data_n,label_a,label_p,label_n) in tqdm(enumerate(train_loader)):
            
            data_a, data_p, data_n = data_a.type(torch.FloatTensor),data_p.type(torch.FloatTensor),data_n.type(torch.FloatTensor)
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
            data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)
            out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
            
            triplet_loss = TripletMarginLoss(0.2).forward(out_a, out_p, out_n)
            loss = triplet_loss
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('selected_triplet_loss', triplet_loss.data)
        print("epoch:",epoch)
        torch.save(model.state_dict(),"checkpoint_{}.pt".format(epoch))

    

if __name__ == '__main__':
    
    main()