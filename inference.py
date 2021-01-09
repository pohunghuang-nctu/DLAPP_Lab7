#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pre_process import data_catalog
from model import DeepSpeakerModel,TripletMarginLoss
from torchsummary import summary
import numpy as np
import pandas as pd
#from tabulate import tabulate
import random
from scipy import stats
import heapq
import time
from pdb import set_trace as bp
import os
import shutil
from glob import glob
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', dest='checkpoint_path', default='checkpoint_99.pt',
        help='specify the path of model parameters')
    parser.add_argument('--database', dest='db_folder', default='database-npy',
        help='specify the database dataset')
    parser.add_argument('--testdata', dest='testdata', default='inference-npy',
        help='specify the folder where test data locates')
    return parser.parse_args()


def load_model(ckp_path):
    model = DeepSpeakerModel(embedding_size=512,num_classes=251)
    checkpoint = torch.load(ckp_path,map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def main():
    args = arg_parse()
    model = load_model(args.checkpoint_path)
    print('model loaded... %s' % args.checkpoint_path)
    libri = data_catalog(args.db_folder,pattern='*.npy') 
    print('database loaded... %s' % args.db_folder)
    labels, embedding = eval(model, libri)
    print('database data evaluated... ')
    np.save('emb',embedding)
    np.save('emb_label',labels)
    print('database data saved... ')
    database = Database(20000)
    for i in range(len(labels)):
        test_array, test_label = embedding[i],labels[i] 
        database.insert(test_label, test_array)
    print("inserting database completed") 
    # import test data as dataframe
    libri = data_catalog(args.testdata,pattern='*.npy')
    print('test data loaded...%s' % args.testdata)
    labels, embedding = eval(model, libri)
    print('test data evaluated')
    score(database, labels, embedding)


def score(database, labels, embedding):
    correct = 0
    for i in range(len(labels)):
        idx, similarity, ref_emb = database.get_most_similar(embedding[i])
        print(type(similarity))
        print(ref_emb.shape)
        pred_label = database.get_label_by_id(idx)
        if pred_label == labels[i]: # hit
            correct += 1
    accuracy = correct / len(labels)
    print('Identification accuracy = %.3f' % accuracy)


def eval(model, libri):
    model.eval()
    labels = []
    with torch.no_grad():
        for i in range(int(len(libri))):
            new_x = []
            filename =libri[i:i + 1]['filename'].values[0]
            filelabel=libri[i:i + 1]['speaker_id'].values[0]
            x = np.load(filename)
            if(x.shape[0]>160):
                for bias in range(0,x.shape[0]-160,160):
                    clipped_x = x[bias:bias+160]
                    new_x.append(clipped_x)
                    labels.append(filelabel)
            else:
                clipped_x = x
                new_x.append(clipped_x)
                labels.append(filelabel)

            x = np.array(new_x)
            # print(x.shape)
            x_tensor = Variable(torch.from_numpy(x.transpose ((0,3, 1, 2))).type(torch.FloatTensor).contiguous())
            # print(x_tensor.shape)
            embedding = model(x_tensor)
            if i == 0 :
                temp_embedding = embedding
            else :
                temp_embedding = torch.cat((temp_embedding,embedding),0)
        temp_embedding = temp_embedding.cpu().detach().numpy()
        labels=np.array(labels)
        labels = labels.astype("int32")
        # print(labels.shape)
        # print(temp_embedding.shape)
        return labels, temp_embedding


class Database():
    "Simulated data structure"
    def __init__(self, data_num):
        self.embs = np.ndarray((data_num,512), dtype=float)
        self.labels = []
        self.indices = 0
    

    def __len__(self):
        return self.indices

    def insert(self, label, emb,index=None):
        " Insert testing data "

        self.embs[self.indices] = emb
        self.labels.append(label)
        self.indices += 1

   
    def get_most_similar(self, embTest):
        testTiles = np.tile(embTest, (self.indices, 1))
        similarities = np.sum(testTiles*self.embs[0:self.indices], axis=1)
        max_similarity = np.max(similarities)
        max_id = np.argmax(similarities)
        return max_id, max_similarity,self.embs[max_id]
    

    def get_label_by_id(self, id):
        return self.labels[id]
    

def get_similarity(embA, embB): # inner product
    ans = np.sum(embA*embB)
    return ans

if __name__ == '__main__':
    main()



