'''
Main script for training and testing a DL model (resnet18) for mmWave beam prediction
Author: Gouranga Charan

'''

def main():
    import os
    import datetime
    import shutil 
    
    import torch
    import torch as t
    import torch.cuda as cuda
    import torch.optim as optimizer
    import torch.nn as nn

    
    from torch.utils.data import DataLoader
    import torchvision.transforms as transf
    from torchsummary import summary
    
    import numpy as np
    import pandas as pd 

    
    from data_feed import DataFeed
    #from build_net import resnet50
    from build_net import resnet18_mod
    #from build_net import resnet34


    ############################################
    ########### Create save directory ##########
    ############################################
    
    # year month day 
    dayTime = datetime.datetime.now().strftime('%m-%d-%Y')
    # Minutes and seconds 
    hourTime = datetime.datetime.now().strftime('%H_%M')
    print(dayTime + '\n' + hourTime)
    
    pwd = os.getcwd() + '//' + 'saved_folder' + '//' + dayTime + '_' + hourTime 
    print(pwd)
    # Determine whether the folder already exists
    isExists = os.path.exists(pwd)
    if not isExists:
        os.makedirs(pwd)    
        
    
    #copy the training files to the saved directory
    shutil.copy('./main_img_beam.py', pwd)
    shutil.copy('./data_feed.py', pwd)
    shutil.copy('./build_net.py', pwd)
    shutil.copy('./scenario6_img_beam_train.csv', pwd)
    shutil.copy('./scenario6_img_beam_val.csv', pwd)
    shutil.copy('./scenario6_img_beam_test.csv', pwd)

    
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

    ########################################################################
    ######################### Hyperparameters ##############################
    ########################################################################
    
    batch_size = 32
    val_batch_size = 1
    lr = 1e-4
    decay = 1e-4
    num_epochs = 15
    train_size = [1]
    
    ########################################################################    
    ########################################################################
    
    
    ########################################################################
    ########################### Data pre-processing ########################
    ########################################################################
    img_resize = transf.Resize((224, 224))
    img_norm = transf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    proc_pipe = transf.Compose(
        [transf.ToPILImage(),
         img_resize,
         transf.ToTensor(),
         img_norm]
    )
    
    train_dir = 'scenario6_img_beam_train.csv'
    val_dir = 'scenario6_img_beam_test.csv'
    train_loader = DataLoader(DataFeed(train_dir, transform=proc_pipe),
                              batch_size=batch_size,
                              num_workers=8,
                              shuffle=False)
    val_loader = DataLoader(DataFeed(val_dir, transform=proc_pipe),
                            batch_size=val_batch_size,
                            num_workers=8,
                            shuffle=False)
    ########################################################################    
    ######################################################################## 
    
    ########################################################################
    #################### Model Training ####################################
    ########################################################################
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
            net = resnet18_mod(pretrained=True, progress=True, num_classes=33)
            net = net.cuda()
            summary(net.cuda(), (3, 224, 224))
    
            #  Optimization parameters:
            criterion = nn.CrossEntropyLoss()
            opt = optimizer.Adam(net.parameters(), lr=lr, weight_decay=decay)
            LR_sch = optimizer.lr_scheduler.MultiStepLR(opt, [4,8], gamma=0.1, last_epoch=-1)
    
            count = 0
            running_loss = []
            running_top1_acc = []
            running_top2_acc = []
            running_top3_acc = []
            best_accuracy = 0

            for epoch in range(num_epochs):
                print('Epoch No. ' + str(epoch + 1))
                skipped_batches = 0
                for tr_count, (img, label) in enumerate(train_loader):
                    net.train()
                    x = img.cuda()
                    opt.zero_grad()
                    label = label.cuda()
                    _, out = net.forward(x)
                    L = criterion(out, label)
                    L.backward()
                    opt.step()
                    batch_loss = L.item()
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
                gt_beam = []
                
                
                for val_count, (imgs, labels) in enumerate(val_loader):
                    net.eval()
                    x = imgs.cuda()
                    labels = labels.cuda()
                    opt.zero_grad()
                    

                    gt_beam.append(labels.detach().cpu().numpy()[0].tolist())
                    total_count += labels.size(0)
                    _, out = net.forward(x)
                    _, top_1_pred = t.max(out, dim=1)
                    top1_pred_out.append(top_1_pred.detach().cpu().numpy()[0].tolist())
                    sorted_out = t.argsort(out, dim=1, descending=True)
                    
                    top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
                    top2_pred_out.append(top_2_pred.detach().cpu().numpy()[0].tolist())
 
                    top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:3])
                    top3_pred_out.append(top_3_pred.detach().cpu().numpy()[0].tolist() )
                        
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
            
                
     
                cur_accuracy  = running_top1_acc[-1]
   
                print("current acc", cur_accuracy)
                print("best acc", best_accuracy)
                if cur_accuracy > best_accuracy:
                    print("Saving the best model")
                    net_name = checkpoint_directory  + '//' +  'CNN_beam_pred'
                    t.save(net.state_dict(), net_name)  
                    best_accuracy =  cur_accuracy  
                print("updated best acc", best_accuracy)
                
                
                
                print("Saving the predicted value in a csv file")
                file_to_save = f'{save_directory}//top1_pred_beam_val_after_{epoch+1}th_epoch.csv'
                indx = np.arange(1, len(top1_pred_out)+1, 1)
                df1 = pd.DataFrame()
                df1['index'] = indx                
                df1['link_status'] = gt_beam
                df1['top1_pred'] = top1_pred_out
                df1['top2_pred'] = top2_pred_out
                df1['top3_pred'] = top3_pred_out
                df1.to_csv(file_to_save, index=False)  
                                                  
    
                LR_sch.step()
            top_1[0,idx] = running_top1_acc[-1]
            top_2[0,idx] = running_top2_acc[-1]
            top_3[0,idx] = running_top3_acc[-1]

    ########################################################################
    ########################################################################
    ################### Load the model checkpoint ##########################    
    test_dir = './scenario6_img_beam_test.csv'
    checkpoint_path = f'{checkpoint_directory}/CNN_beam_pred'   
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval() 
    net = net.cuda()   
    
    test_loader = DataLoader(DataFeed(test_dir, transform=proc_pipe),
                            batch_size=val_batch_size,
                            #num_workers=8,
                            shuffle=False)
    
    print('Start validation')
    ave_top1_acc = 0
    ave_top2_acc = 0
    ave_top3_acc = 0
    ind_ten = t.as_tensor([0, 1, 2, 3, 4], device='cuda:0')
    top1_pred_out = []
    top2_pred_out = []
    top3_pred_out = []
    total_count = 0

    gt_beam = []
    for val_count, (imgs, labels) in enumerate(test_loader):
        net.eval()
        x = imgs.cuda()                   
        opt.zero_grad()
        labels = labels.cuda()
        gt_beam.append(labels.detach().cpu().numpy()[0].tolist())
        total_count += labels.size(0)
        _, out = net.forward(x)
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
   
    print('Training_size {}--No. of skipped batchess {}'.format(n,skipped_batches))
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