# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 20:33:46 2022

@author: Gouranga
"""

def main():

    import os
    import datetime
    import shutil 
    import ast
    
    import torch
    import torch as t
    import torch.cuda as cuda
    import torch.optim as optimizer
    import torch.nn as nn

    
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    import torchvision.transforms as transf
    from torchsummary import summary
    
    import numpy as np
    import pandas as pd 
    from PIL import Image



    ############################################
    ########### Create save directory ##########
    ############################################
    
    # year month day 
    dayTime = datetime.datetime.now().strftime('%m-%d-%Y')
    # Minutes and seconds 
    hourTime = datetime.datetime.now().strftime('%H_%M')
    print(dayTime + '\n' + hourTime)
    
    pwd = os.getcwd() + '//' + 'saved_folder_eval' + '//' + dayTime + '_' + hourTime 
    print(pwd)
    # Determine whether the folder already exists
    isExists = os.path.exists(pwd)
    if not isExists:
        os.makedirs(pwd)    
        
    
    
    #copy the training files to the saved directory
    shutil.copy('./main_img_beam_eval_ResNet50.py', pwd)
    shutil.copy('./scenario5_img_pos_beam_test.csv', pwd)
    
    #create folder to save analysis files and checkpoint
    
    save_directory = pwd + '//' + 'saved_analysis_files'
    checkpoint_directory = pwd + '//' + 'checkpoint'

    isExists = os.path.exists(save_directory)
    if not isExists:
        os.makedirs(save_directory) 
        
    isExists = os.path.exists(checkpoint_directory)
    if not isExists:
        os.makedirs(checkpoint_directory)         
    
    ############################################
    
    ############################################
    # Define a custom model that takes two inputs: an RGB image and a normalized position value
    class CustomModel(nn.Module):
        def __init__(self, num_classes):
            super(CustomModel, self).__init__()
            self.resnet = models.resnet50(pretrained=True)
            self.position_fc = nn.Linear(2, 512)
            self.output_fc = nn.Linear(2048 + 512, num_classes)
    
        def forward(self, x, pos):
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
    
            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
    
            x = self.resnet.avgpool(x)
            resnet_features = x.view(x.size(0), -1)
    
            pos_features = self.position_fc(pos)
            combined_features = torch.cat([resnet_features, pos_features], dim=1)
    
            output = self.output_fc(combined_features)
    
            return output   
    ############################################   
    # Define a custom dataset class that reads from a CSV file    		
    class CustomDataset(Dataset):
        def __init__(self, csv_file, transform=None):
            self.data = pd.read_csv(csv_file)
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            img_path = self.data.loc[idx, 'unit1_rgb']
            image = Image.open(img_path).convert('RGB')
            position = ast.literal_eval(self.data.loc[idx, 'unit2_pos'])
            label = self.data.loc[idx, 'unit1_beam_32']
    
            if self.transform:
                image = self.transform(image)
    
            position = torch.tensor(position, dtype=torch.float32)
    
            return image, position, label 
    
    ########################################################################

    # Training Hyper-parameters
    val_batch_size = 1
    train_size = [1]
    lr = 1e-4
    decay = 1e-4
    
    ########################################################################    
    ########################################################################
    

    ########################################################################
    ########################### Data pre-processing ########################
    ########################################################################
    img_resize = transf.Resize((224, 224))
    img_norm = transf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    proc_pipe = transf.Compose(
        [img_resize,
         transf.ToTensor(),
         img_norm]
    )
                            
    ########################################################################    
    ######################################################################## 
    
    ########################################################################
    ########################################################################
    ################### Load the model checkpoint ##########################    
    num_classes = 33
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = './scenario5_img_pos_beam_test.csv'
    checkpoint_path = 'saved_folder/best_model_ResNet50/checkpoint/CNN_beam_pred'   
    net = CustomModel(num_classes).to(device)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval() 
    net = net.cuda()   
    
    opt = optimizer.Adam(net.parameters(), lr=lr, weight_decay=decay)
    
    test_loader = DataLoader(CustomDataset(test_dir, transform=proc_pipe),
                            batch_size=val_batch_size,
                            shuffle=False)
    
    print('Start validation')
    running_top1_acc = []
    running_top2_acc = []
    running_top3_acc = []
    ave_top1_acc = 0
    ave_top2_acc = 0
    ave_top3_acc = 0
    ind_ten = t.as_tensor([0, 1, 2, 3, 4], device='cuda:0')
    top1_pred_out = []
    top2_pred_out = []
    top3_pred_out = []
    total_count = 0

    gt_beam = []
    for val_count, (imgs, pos, labels) in enumerate(test_loader):
        net.eval()
        x = imgs.cuda()
        pos = pos.cuda()                   
        opt.zero_grad()
        labels = labels.cuda()
        gt_beam.append(labels.detach().cpu().numpy()[0].tolist())
        total_count += labels.size(0)
        out = net.forward(x, pos)
        _, top_1_pred = t.max(out, dim=1)
        top1_pred_out.append(top_1_pred.detach().cpu().numpy()[0].tolist())
        sorted_out = t.argsort(out, dim=1, descending=True)
        
        top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
        top2_pred_out.append(top_2_pred.detach().cpu().numpy()[0].tolist())

        top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:3])
        top3_pred_out.append(top_3_pred.detach().cpu().numpy()[0].tolist()  )
            
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

    print("Saving the predicted value in a csv file")
    file_to_save = f'{save_directory}//best_epoch_eval.csv'
    indx = np.arange(1, len(top1_pred_out)+1, 1)
    df2 = pd.DataFrame()
    df2['index'] = indx                
    df2['link_status'] = gt_beam
    df2['top1_pred'] = top1_pred_out
    df2['top2_pred'] = top2_pred_out
    df2['top3_pred'] = top3_pred_out
    df2.to_csv(file_to_save, index=False) 
    
    
if __name__ == "__main__":
    main()