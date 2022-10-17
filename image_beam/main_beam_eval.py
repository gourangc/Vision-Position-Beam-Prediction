'''
Main script for training and testing a DL model (resnet18) for mmWave beam prediction
Author: Gouranga Charan
Nov. 2020
'''

import os
import datetime
import sys
import torch as t
import torch.cuda as cuda
import torch.optim as optimizer
import torch.nn as nn
import torch.nn.functional as F
from data_feed import DataFeed
from torch.utils.data import DataLoader
import torchvision.transforms as transf
import numpy as np
import matplotlib.pyplot as plt
from build_net import resnet50
from build_net import resnet18_mod
from build_net import resnet34
from skimage import io, transform
from scipy import io
from torchsummary import summary
import pandas as pd
import csv
#import gc 
#np.set_printoptions(threshold=np.inf)



#Model Hyperparameters
val_batch_size = 1
train_size = [1]
lr = 1e-4
decay = 1e-4

#Loading the saved checkpoint
model = resnet18_mod(pretrained=True, progress=True, num_classes=64)
model = model.cuda()

checkpoint_path = 'checkpoint/resnet18_beam_pred_vehicle_data'
model.load_state_dict(t.load(checkpoint_path))
model.eval()
summary(model.cuda(), (3, 224, 224))



# Data pre-processing:
img_resize = transf.Resize((224, 224))
img_norm = transf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
proc_pipe = transf.Compose(
    [transf.ToPILImage(),
     img_resize,
     transf.ToTensor(),
     img_norm]
    )
    
    
val_dir = './example_test_data.csv'
val_loader = DataLoader(DataFeed(val_dir, transform=proc_pipe),
                        batch_size=val_batch_size,
                        #num_workers=8,
                        shuffle=False)
criterion = nn.CrossEntropyLoss()
opt = optimizer.Adam(model.parameters(), lr=lr, weight_decay=decay)  
val_acc = []   
feature_vec = []   
                

top_1 = np.zeros( (1,len(train_size)) )
top_2 = np.zeros( (1,len(train_size)) )
top_3 = np.zeros( (1,len(train_size)) )

running_top1_acc = []
running_top2_acc = []
running_top3_acc = []

print('Start validation')
ave_top1_acc = 0
ave_top2_acc = 0
ave_top3_acc = 0
ind_ten = t.as_tensor([0, 1, 2], device='cuda:0')
top1_pred_out = []
top2_pred_out = []
top3_pred_out = []   
total_count = 0
for val_count, (imgs, labels) in enumerate(val_loader):
    x = imgs.cuda()
    opt.zero_grad()
    labels = labels.cuda()
    
    total_count += labels.size(0)
    _, out = model.forward(x)
    feature_cnn, _ = model.forward(x)
    
    feature_cnn = feature_cnn.detach().cpu().numpy()[0]
    feature_cnn_lst = list(feature_cnn)
    feature_vec.append(feature_cnn_lst)
    
    _, top_1_pred = t.max(out, dim=1)
    top1_pred_out.append(top_1_pred.cpu().numpy())
    
    sorted_out = t.argsort(out, dim=1, descending=True)
    
    top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
    top2_pred_out.append(top_2_pred.cpu().numpy())
    
    top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten)
    top_3_pred_0 = t.index_select(sorted_out, dim=1, index=ind_ten[0])
    top_3_pred_1 = t.index_select(sorted_out, dim=1, index=ind_ten[1])
    top_3_pred_2 = t.index_select(sorted_out, dim=1, index=ind_ten[2])
    top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten)
    tmp = [top_3_pred_0.cpu().numpy()[0], top_3_pred_1.cpu().numpy()[0], top_3_pred_2.cpu().numpy()[0] ]        
    top3_pred_out.append(tmp)   
    
    reshaped_labels = labels.reshape((labels.shape[0], 1))
    tiled_2_labels = reshaped_labels.repeat(1, 2)
    tiled_3_labels = reshaped_labels.repeat(1, 3)
    batch_top1_acc = t.sum(top_1_pred == labels, dtype=t.float32)
    batch_top2_acc = t.sum(top_2_pred == tiled_2_labels, dtype=t.float32)
    batch_top3_acc = t.sum(top_3_pred == tiled_3_labels, dtype=t.float32)
    ave_top1_acc += batch_top1_acc.item()
    ave_top2_acc += batch_top2_acc.item()
    ave_top3_acc += batch_top3_acc.item()
print("total test examples are", total_count)
running_top1_acc.append(ave_top1_acc / total_count)  # (batch_size * (count_2 + 1)) )
running_top2_acc.append(ave_top2_acc / total_count)
running_top3_acc.append(ave_top3_acc / total_count)  # (batch_size * (count_2 + 1)))

print('Average Top-1 accuracy {}'.format( running_top1_acc[-1]))
print('Average Top-2 accuracy {}'.format( running_top2_acc[-1]))
print('Average Top-3 accuracy {}'.format( running_top3_acc[-1]))


 
import csv
with open("test_pos_beam_img_updated_GC_v2_w_vec.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(zip(feature_vec))


