'''
Main script for training and testing a DL model (resnet18) for mmWave beam prediction
Author: Gouranga Charan
Nov. 2020
'''


def main():
    import os
    import datetime
    import sys
    import torch as t
    import torch
    import torch.cuda as cuda
    import torch.optim as optimizer
    import torch.nn as nn
    import torch.nn.functional as F
    from data_feed_img_pos import DataFeed
    from torch.utils.data import DataLoader
    import torchvision.transforms as transf
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io, transform
    from scipy import io
    from torchsummary import summary
    import ast
    import torch.nn.functional as F
    import csv
    import os
    
    isExists = os.path.exists('./checkpoint')
    if not isExists:
        os.makedirs('./checkpoint') 
        
    isExists = os.path.exists('./saved_analysis_file')
    if not isExists:
        os.makedirs('./saved_analysis_file') 
    
    
    # Hyper-parameters
    batch_size = 128
    val_batch_size = 1
    lr = 0.001
    decay = 1e-4
    image_grab = False
    num_epochs = 50
    train_size = [1]
    
    # Data pre-processing:
    proc_pipe = transf.Compose(
        [
         transf.ToTensor()
        ]
    )
    train_dir = 'example_train_data.csv'
    val_dir = 'example_test_data.csv'

    train_loader = DataLoader(DataFeed(train_dir, transform = proc_pipe),
                              batch_size=batch_size,
                              shuffle=False)
                              
    val_loader = DataLoader(DataFeed(val_dir, transform=proc_pipe),
                            batch_size=val_batch_size,
                            shuffle=False)
                            
    now = datetime.datetime.now()
    
    
    # Hyperparameters for our network
    input_size = 2050
    hidden_sizes = [512 , 512]
    hidden_sizes1 = [512, 512]
    output_size = 64
    
    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.BatchNorm1d(512),
                          nn.Dropout(0.5),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes1[0]),
                          nn.BatchNorm1d(512),
                          nn.Dropout(0.5),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes1[0], output_size),
                          )    
    
    #Network training:
    val_acc = []
    with cuda.device(0):
        top_1 = np.zeros( (1,len(train_size)) )
        top_2 = np.zeros( (1,len(train_size)) )
        top_3 = np.zeros( (1,len(train_size)) )
        acc_loss = 0
        itr = []
        for idx, n in enumerate(train_size):
            print('```````````````````````````````````````````````````````')
            print('Training size is {}'.format(n))
            # Build the network:
            net = model.cuda()
            layers = list(net.children())
    
            #  Optimization parameters:
            criterion = nn.CrossEntropyLoss()
            opt = optimizer.Adam(net.parameters(), lr=lr, weight_decay=decay)
            LR_sch = optimizer.lr_scheduler.MultiStepLR(opt, [6, 20], gamma=0.1, last_epoch=-1)
            
    
            count = 0
            running_loss = []
            running_top1_acc = []
            running_top2_acc = []
            running_top3_acc = []
            
            for epoch in range(num_epochs):
                print('Epoch No. ' + str(epoch + 1))
                skipped_batches = 0
                for tr_count, y in enumerate(train_loader):
                    net.train()
                    data = y[:, :-1].type(torch.Tensor)                  
                    label = y[:, -1].long()                      
                    x = data.cuda()
                    opt.zero_grad()
                    label = label.cuda()
                    out = net.forward(x)
                    loss = criterion(out, label)
                    loss.backward()
                    opt.step()                    
                    batch_loss = loss.item()
                    acc_loss += batch_loss
                    count += 1
                    if np.mod(count, 10) == 0:
                        print('Training-Batch No.' + str(count))
                        running_loss.append(batch_loss)  # running_loss.append()
                        itr.append(count)
                        print('Loss = ' + str(running_loss[-1]))
    
                print('Start validation')
                ave_top1_acc = 0
                ave_top2_acc = 0
                ave_top3_acc = 0
                
                ind_ten = t.as_tensor([0, 1, 2], device='cuda:0')
                top1_pred_out = []
                top2_pred_out = []
                top3_pred_out = []
                total_count = 0
                intermediate_out = []
                for val_count, y in enumerate(val_loader):
                    net.eval()
                    data = y[:, :-1].type(torch.Tensor) 
                    x = data.cuda()                    
                    labels = y[:, -1].long()  
                    opt.zero_grad()
                    labels = labels.cuda()
                    total_count += labels.size(0)
                    out = net.forward(x)
                    
                    _, top_1_pred = t.max(out, dim=1)
                    top1_pred_out.append(top_1_pred.cpu().numpy())
                    
                    sorted_out = t.argsort(out, dim=1, descending=True)
                    
                    top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
                    top2_pred_out.append(top_2_pred.cpu().numpy())
                    
                    top_3_pred_0 = t.index_select(sorted_out, dim=1, index=ind_ten[0])
                    top_3_pred_1 = t.index_select(sorted_out, dim=1, index=ind_ten[1])
                    top_3_pred_2 = t.index_select(sorted_out, dim=1, index=ind_ten[2])
                    top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten)
                    
                    tmp = [top_3_pred_0.item(), top_3_pred_1.item(), top_3_pred_2.item() ]
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
                print('Training_size {}--No. of skipped batchess {}'.format(n,skipped_batches))
                print('Average Top-1 accuracy {}'.format( running_top1_acc[-1]))
                print('Average Top-2 accuracy {}'.format( running_top2_acc[-1]))
                print('Average Top-3 accuracy {}'.format( running_top3_acc[-1]))
                print("Saving the predicted value in a csv file")
                tmp = [running_top1_acc[-1], running_top2_acc[-1], running_top3_acc[-1]]
                val_acc.append(tmp)
                
                with open("./intermediate_output.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(intermediate_out))
                with open("./saved_analysis_file/top1_pred_beam_val_after_%sth_epoch.csv"%(epoch+1), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(top1_pred_out))
                with open("./saved_analysis_file/top2_pred_beam_val_afte_%sth_epoch.csv"%(epoch+1), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(top2_pred_out))  
                with open("./saved_analysis_file/top3_pred_beam_val_after_%sth_epoch.csv"%(epoch+1), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(top3_pred_out))               
    
                LR_sch.step()
            top_1[0,idx] = running_top1_acc[-1]
            top_2[0,idx] = running_top2_acc[-1]
            top_3[0,idx] = running_top3_acc[-1]
    
    net_name = 'checkpoint/image_pos_beam_pred_ckpt'
    t.save(net.state_dict(), net_name)

    
if __name__ == "__main__":
    #run()
    main()