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
    import csv
    import os
    
    isExists = os.path.exists('./checkpoint')
    if not isExists:
        os.makedirs('./checkpoint') 
        
    isExists = os.path.exists('./saved_analysis_file')
    if not isExists:
        os.makedirs('./saved_analysis_file')        
    
    # Hyper-parameters
    batch_size = 32
    val_batch_size = 1
    lr = 1e-4
    decay = 1e-4
    image_grab = False
    num_epochs = 15
    train_size = [1]
    
    # Data pre-processing:
    img_resize = transf.Resize((224, 224))
    img_norm = transf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    proc_pipe = transf.Compose(
        [transf.ToPILImage(),
         img_resize,
         transf.ToTensor(),
         img_norm]
    )
    
    train_dir = 'example_train_data.csv'
    val_dir = 'example_test_data.csv'
    
    
    train_loader = DataLoader(DataFeed(train_dir, transform=proc_pipe),
                              batch_size=batch_size,
                              num_workers=8,
                              shuffle=False)
    val_loader = DataLoader(DataFeed(val_dir, transform=proc_pipe),
                            batch_size=val_batch_size,
                            num_workers=8,
                            shuffle=False)
    now = datetime.datetime.now()
    
    
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
            net = resnet50(pretrained=True, progress=True, num_classes=64)
            net = net.cuda()
            layers = list(net.children())
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
            
            #allowed_batches = np.floor((n*3500)/batch_size)
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
                for val_count, (imgs, labels) in enumerate(val_loader):
                    net.eval()
                    x = imgs.cuda()
                    opt.zero_grad()
                    labels = labels.cuda()
                    
                    total_count += labels.size(0)
                    
                    _, out = net.forward(x)
                    
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
                
                print('Average Top-1 accuracy {}'.format( running_top1_acc[-1]))
                print('Average Top-2 accuracy {}'.format( running_top2_acc[-1]))
                print('Average Top-3 accuracy {}'.format( running_top3_acc[-1]))
                print("Saving the predicted value in a csv file")
                tmp = [running_top1_acc[-1], running_top2_acc[-1], running_top3_acc[-1]]
                val_acc.append(tmp)
                
                with open("./saved_analysis_file/top1_pred_beam_val_after_%sth_epoch.csv"%(epoch+1), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(top1_pred_out))
                with open("./saved_analysis_file/top2_pred_beam_val_after_%sth_epoch.csv"%(epoch+1), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(top2_pred_out))  
                with open("./saved_analysis_file/top3_pred_beam_val_after_%sth_epoch.csv"%(epoch+1), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(top3_pred_out))               
    
                LR_sch.step()
            top_1[0,idx] = running_top1_acc[-1]
            top_2[0,idx] = running_top2_acc[-1]
            top_3[0,idx] = running_top3_acc[-1]
    with open("./val_acc_32_beams.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(val_acc)      
    net_name = 'checkpoint/img_beam_pred_ckpt'
    t.save(net.state_dict(), net_name)

    
if __name__ == "__main__":
    #run()
    main()