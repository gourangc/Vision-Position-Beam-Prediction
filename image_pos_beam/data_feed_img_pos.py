'''
A data feeding class. It generates a list of data samples, each of which is
a tuple of a string (extracted feature vectors) and an integer (beam index), and it defines
a data-fetching method.
Author: Gouranga Charan
Nov. 2020
'''


import os
import random
import pandas as pd
import torch
import numpy as np
import random
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ast
import json

############### Create data sample list #################
def create_samples(root, shuffle=False, nat_sort=False):
    f = pd.read_csv(root)
    data_samples = []
    pred_val = []
    for idx, row in f.iterrows():
        feature = row.values[7]
        feature = json.loads(feature)
        loc_beam = row.values[4:7]
        tmp = [feature, loc_beam]
        tmp_flatten = [item for items in tmp for item in items]
        data_samples.append(tmp_flatten)

    return data_samples
#############################################################

class DataFeed(Dataset):
    '''
    A class retrieving a tuple of (image,label). It can handle the case
    of empty classes (empty folders).
    '''
    def __init__(self,root_dir, nat_sort = False, transform=None, init_shuflle = True):
        self.root = root_dir
        self.samples = create_samples(self.root,shuffle=init_shuflle,nat_sort=nat_sort)
        self.transform = transform
        self.seq_len = 2051


    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pos_data = torch.zeros((self.seq_len,))
        for i,s in enumerate(sample):
            x = s
            pos_data[i] = torch.tensor(x, requires_grad=False)
        return pos_data
