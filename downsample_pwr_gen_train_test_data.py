# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:16:55 2022

@author: gcharan
"""




import os
import numpy as np
import pandas as pd
import random
import itertools
import pickle
import ast
import matplotlib.pyplot as plt
from matplotlib import colors

from numpy.random import RandomState
from sklearn.utils import shuffle

import argparse



###############################################
#### Input dataset name
###############################################
scenario_num = 6
root_folder = f'Scenario{scenario_num}'
data_csv = f'./Scenario{scenario_num}/scenario{scenario_num}.csv'




###############################################
# Read dataset to create a list of the input sequence   
###############################################

df = pd.read_csv(data_csv)
image_data_lst = df['unit1_rgb'].values
pwr_data_lst = df['unit1_pwr_60ghz'].values
#original_beam = df['unit1_beam_index'].values


###############################################
#### subsample the power and generate the 
#### updated beam indices
###############################################
updated_beam = []
original_beam = []
for entry in pwr_data_lst:
    print(entry)
    data_to_read = f'./{root_folder}{entry[1:]}'
    pwr_data = np.loadtxt(data_to_read)
    original_beam.append(np.argmax(pwr_data)+1)
    updated_pwr = []
    j = 0
    while j < (len(pwr_data)- 1):
        tmp_pwr = pwr_data[j]
        updated_pwr.append(tmp_pwr)
        j += 2
    updated_beam.append(np.argmax(updated_pwr)+1)
    




def create_img_beam_dataset():   
    
    folder_to_save = f'./ML_code/{root_folder}/image_beam'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    
    #############################################
    ###### created updated image path ###########
    #############################################
    updated_img_path = []
    for entry in image_data_lst:
        img_path = entry.split('./')[1]
        updated_path = f'../../../{root_folder}/{img_path}'
        updated_img_path.append(updated_path)
    
    
    #############################################
    # saving the image-beam development dataset for training and validation
    #############################################
                            
    indx = np.arange(1, len(updated_beam)+1,1)
    df_new = pd.DataFrame()
    df_new['index'] = indx   
    df_new['unit1_rgb'] = updated_img_path   
    df_new['unit1_beam_32'] = updated_beam    
    df_new['unit1_beam_64'] = original_beam 
    df_new.to_csv(fr'./{folder_to_save}/scenario{scenario_num}_img_beam.csv', index=False)       
      
    #############################################
    #generate the train and test dataset
    #############################################    
    rng = RandomState(1)
    train, val, test = np.split(df_new.sample(frac=1, random_state=rng ), [int(.6*len(df_new)), int(.9*len(df_new))])
    train.to_csv(f'./{folder_to_save}/scenario{scenario_num}_img_beam_train.csv', index=False)
    val.to_csv(f'./{folder_to_save}/scenario{scenario_num}_img_beam_val.csv', index=False)
    test.to_csv(f'./{folder_to_save}/scenario{scenario_num}_img_beam_test.csv', index=False)


def create_pos_beam_dataset():  
    
    folder_to_save = f'./ML_code/{root_folder}/pos_beam'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    
    ###############################################
    ####### read position values from dataset #####
    ###############################################

    lat = []
    lon = []
    pos_data_path = df['unit2_loc'].values
    for entry in pos_data_path:
        data_to_read = f'./{root_folder}{entry[1:]}'
        pos_val = np.loadtxt(data_to_read)
        #lat_val, lon_val = pos_val[0], pos_val[1]
        lat.append(pos_val[0])
        lon.append(pos_val[1])
        
    def norm_data(data_lst):
        norm_data = []
        for entry in data_lst:
            norm_data.append((entry - min(data_lst))/(max(data_lst) - min(data_lst)))
        return norm_data

    ###############################################
    ##### normalize latitude and longitude data ###
    ###############################################
    lat_norm = norm_data(lat)
    lon_norm = norm_data(lon)

    ###############################################
    ##### generate final pos data #################
    ###############################################
    pos_data = []
    for j in range(len(lat_norm)):
        pos_data.append([lat_norm[j], lon_norm[j]])


    #############################################
    # saving the pos-beam development dataset for training and validation
    #############################################
                            
    indx = np.arange(1, len(updated_beam)+1,1)
    df_new = pd.DataFrame()
    df_new['index'] = indx   
    df_new['unit2_pos'] = pos_data   
    df_new['unit1_beam_32'] = updated_beam
    df_new['unit1_beam_64'] = original_beam    
    df_new.to_csv(fr'./{folder_to_save}/scenario{scenario_num}_pos_beam.csv', index=False) 
    
    #############################################
    #generate the train and test dataset
    #############################################    
    rng = RandomState(1)
    train, val, test = np.split(df_new.sample(frac=1, random_state=rng ), [int(.6*len(df_new)), int(.9*len(df_new))])
    train.to_csv(f'./{folder_to_save}/scenario{scenario_num}_pos_beam_train.csv', index=False)
    val.to_csv(f'./{folder_to_save}/scenario{scenario_num}_pos_beam_val.csv', index=False)
    test.to_csv(f'./{folder_to_save}/scenario{scenario_num}_pos_beam_test.csv', index=False)    
    
def create_img_pos_beam_dataset():  
    
    folder_to_save = f'./ML_code/{root_folder}/img_pos_beam'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
        
    #############################################
    ###### created updated image path ###########
    #############################################
    updated_img_path = []
    for entry in image_data_lst:
        img_path = entry.split('./')[1]
        updated_path = f'../../../{root_folder}/{img_path}'
        updated_img_path.append(updated_path)
    
    ###############################################
    ####### read position values from dataset #####
    ###############################################

    lat = []
    lon = []
    pos_data_path = df['unit2_loc'].values
    for entry in pos_data_path:
        data_to_read = f'./{root_folder}{entry[1:]}'
        pos_val = np.loadtxt(data_to_read)
        #lat_val, lon_val = pos_val[0], pos_val[1]
        lat.append(pos_val[0])
        lon.append(pos_val[1])
        
    def norm_data(data_lst):
        norm_data = []
        for entry in data_lst:
            norm_data.append((entry - min(data_lst))/(max(data_lst) - min(data_lst)))
        return norm_data

    ###############################################
    ##### normalize latitude and longitude data ###
    ###############################################
    lat_norm = norm_data(lat)
    lon_norm = norm_data(lon)

    ###############################################
    ##### generate final pos data #################
    ###############################################
    pos_data = []
    for j in range(len(lat_norm)):
        pos_data.append([lat_norm[j], lon_norm[j]])


    #############################################
    # saving the pos-beam development dataset for training and validation
    #############################################
                            
    indx = np.arange(1, len(updated_beam)+1,1)
    df_new = pd.DataFrame()
    df_new['index'] = indx   
    df_new['unit1_rgb'] = updated_img_path
    df_new['unit2_pos'] = pos_data   
    df_new['unit1_beam_32'] = updated_beam
    df_new['unit1_beam_64'] = original_beam    
    df_new.to_csv(fr'./{folder_to_save}/scenario{scenario_num}_pos_beam.csv', index=False) 
    
    #############################################
    #generate the train and test dataset
    #############################################    
    rng = RandomState(1)
    train, val, test = np.split(df_new.sample(frac=1, random_state=rng ), [int(.6*len(df_new)), int(.9*len(df_new))])
    train.to_csv(f'./{folder_to_save}/scenario{scenario_num}_img_pos_beam_train.csv', index=False)
    val.to_csv(f'./{folder_to_save}/scenario{scenario_num}_img_pos_beam_val.csv', index=False)
    test.to_csv(f'./{folder_to_save}/scenario{scenario_num}_img_pos_beam_test.csv', index=False) 

if __name__ == "__main__":   
    create_img_beam_dataset()
    create_pos_beam_dataset()
    create_img_pos_beam_dataset()

    
