'''
A data feeding class. It generates a list of data samples, each of which is
a tuple of a string (image path) and an integer (beam index), and it defines
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

############### Create data sample list #################
def create_samples(root, shuffle=False, nat_sort=False):
    f = pd.read_csv(root)
    data_samples = []
    pred_val = []
    for idx, row in f.iterrows():
        img_paths = row.values[1:5]
        data_samples.append(img_paths)
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


    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample[0])
        if self.transform:
            img = self.transform(img)
        label = sample[3]
        return (img,label)
