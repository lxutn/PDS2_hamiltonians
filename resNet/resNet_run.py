# -*- coding: utf-8 -*-
"""
Load or train and then evaluates a ResNet on CIFAR10 like defined in the paper [1]
The training can be done with a grid search along learning rate (lr_lst) and epochs (nb_epochs_lst)
To train just a network do agrid search along 1 parameter in each

The ResNet structure and the pretrained models come from https://github.com/akamaster/pytorch_resnet_cifar10
 
Similar accuracy obtained w.r.t. the paper in this online version
Additionnaly, a resNet110 has been retrained from scratch and obtained similar performance than paper
However, the main difference is that here we train with 50k on train and 10 on test where as on the 
paper they also used a validation dataset. See other small differences on the GitHub
         
Based on:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

# ----- Libraries ----- #
#Pytorch
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
#ResNet
import resnet
#Utils
import time
import numpy as np

# ----- Constants & Parameters ----- #
#Pretrain or train network
download_pretrained_net = True  #use pretrained model or start from scratch
skip_train = True               #skip the training (directly test the network)

#Network
resnet_size = 20                #select the size of the ResNet
NET_PATH = "pretrained_models/"
saved_net_name = {  '20': 'resnet20-12fca82f.th', 
                    '32': 'resnet32-d509ac18.th',
                    '44': 'resnet44-014dd654.th',
                    '56': 'resnet56-4bfd9763.th',
                    '110': 'resnet110-1d1ed7c2.th',
                    '1202': 'resnet1202-f3b1deed.th'}

#Preprocessing
crop_values = [32, 4]           #crop: 4 pixels padded and then crop to 32x32 (from [1])

#Training infos
batch_size = 128                #batch size from [1]
lr_lst = [1e-1]                 #lr used in the [1] (for 20,32,44 !SPECIAL CARE FOR 110&1202)
nb_epochs_lst = [165]           #deduced nb of epochs from [1]
momentum = 0.9                  #SGD momentum from [1] 
wd = 0.0001                     #weight decay from [1]
gamma = 0.1                     #lr decay [1]
lr_milestones = [80, 120]       #milestones for lr decay deduced from [1]
#Only for big networks
lr_decay_big_net = 0.1          #for big net the lr starts lower (from [1])
train_perf_lr_up_big_net = 0.8  #performance to reincrease the lr of big net (from [1])
lr_upgrade_done = False         #keep in track if the lr upgrade has been done (from [1])

#Datasets stats
cifar_mean = [0.485, 0.456, 0.406]
cifar_std = [0.229, 0.224, 0.225]

#GPU infos
use_cuda = torch.cuda.is_available()  # not no_cuda and
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
torch.set_num_threads(1)

#Utils
loss_dsp_rate = 500 
SAVE_PATH = "resNet"+str(resnet_size)+"/"

# ----- Functions ----- #
def nb_param_model(model):
    #Return the nb of trainable parameters of a model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----- Loading and preprocessing the data ----- #
#Transformations
preprocess_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(crop_values[0], crop_values[1]),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])
preprocess_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

#Load data
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=True, download=True, 
                                    transform=preprocess_train), 
                                    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=False, 
                                    transform=preprocess_test),
                                    batch_size=batch_size, shuffle=False, **kwargs)

# ----- Grid search ----- #
print("Resnet size: "+str(resnet_size))
print("Pretrained: "+str(download_pretrained_net))
print("Will be trained: "+str(not skip_train))

#Create variables for saving
train_acc_lst = np.zeros((len(lr_lst),len(nb_epochs_lst)))
test_acc_lst = np.zeros((len(lr_lst),len(nb_epochs_lst)))
train_loss_lst = np.zeros((len(lr_lst),len(nb_epochs_lst)))
test_loss_lst = np.zeros((len(lr_lst),len(nb_epochs_lst)))
if len(lr_lst) == 1 and len(nb_epochs_lst) == 1:
    train_loss_evolv = np.zeros(int(nb_epochs_lst[0]))

for lr_i, lr in enumerate(lr_lst):
    for e_i, nb_epochs in enumerate(nb_epochs_lst):
        print("Learning rate at "+str(lr)+" and number of epochs at "+str(nb_epochs))
        # ----- Define the problem (model, optimiser, ...)
        #Create the model
        resNet = torch.nn.DataParallel(resnet.__dict__["resnet"+str(resnet_size)](),device_ids=[0])
        resNet.to(device)
        print("This ResNet has "+str(nb_param_model(resNet))+" parameters")
        
        #Load the pretrained model  
        if download_pretrained_net:
            saved_data = torch.load(NET_PATH+saved_net_name[str(resnet_size)],
                                    map_location=torch.device(device))
            resNet.load_state_dict(saved_data['state_dict'])
        
        #Loss fct
        criterion = nn.CrossEntropyLoss().to(device)
        
        #Optimiser
        optimizer = optim.SGD(resNet.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        
        #LR modification
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones,gamma=gamma)
        
        #Big resnet have a different lr
        if resnet_size >=110:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr*lr_decay_big_net
        
        if not skip_train:
            # ----- Train the model ----- #
            print("\nProblem with "+str(nb_epochs)+" epochs and "+str(lr)+" as lr")
            print("TRAINING START")
            start_train = time.time()
            resNet.train()
            for epoch in range(int(nb_epochs)):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.requires_grad_().to(device), target.to(device)
                    optimizer.zero_grad()
                    output = resNet(data)
                    loss = criterion(output ,target)
                    loss.backward()
                    optimizer.step()
                    
                    if(batch_idx%loss_dsp_rate)==0:
                        print("At epoch "+str(epoch)+" and batch_idx "+str(batch_idx)+" loss is "+str(loss.item()))
                
                #for big network if good performance has been obtained go to normal lr
                if resnet_size >=110 and not lr_upgrade_done:
                    correct = 0
                    with torch.no_grad():
                        for data, target in train_loader:
                            data, target = data.to(device), target.to(device)
                            output = resNet(data)   
                            pred = output.argmax(dim=1, keepdim=True) 
                            correct += pred.eq(target.view_as(pred)).sum().item()
                    train_acc = correct/len(train_loader.dataset)
                    
                    if train_acc > train_perf_lr_up_big_net:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        lr_upgrade_done = True
                        print("Upgrade of lr done")
                    else:
                        print("Upgrade of lr not done for now, acc is "+str(train_acc))
                        
                lr_scheduler.step()
                        
                #save value of the loss if not grid search
                if len(lr_lst) == 1 and len(nb_epochs_lst) == 1:
                    train_loss_evolv[epoch] = loss
            
            finish_train = time.time()
            print("The training lasted "+str(finish_train-start_train)+" s")
        
        # ----- Test the model ----- #
        print("TESTING START")
        #On the training dataset
        resNet.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = resNet(data)        
                train_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = correct/len(train_loader.dataset)
        
        #On the testing dataset
        resNet.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = resNet(data)          
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc = correct/len(test_loader.dataset)
        
        #Stock the results
        train_acc_lst[lr_i][e_i] = train_acc
        test_acc_lst[lr_i][e_i] = test_acc
        train_loss_lst[lr_i][e_i] = train_loss
        test_loss_lst[lr_i][e_i] = test_loss
        
        #Display the result
        print("Train accuracy is "+str(train_acc))
        print("Train loss is "+str(train_loss))
        print("Test accuracy is "+str(test_acc))
        print("Test loss is "+str(test_loss))
        
#Save results
np.save(SAVE_PATH+"nb_epochs_lst.npy", nb_epochs_lst)
np.save(SAVE_PATH+"lr_lst.npy", lr_lst)
np.save(SAVE_PATH+"train_loss.npy", train_loss_lst)
np.save(SAVE_PATH+"test_loss.npy", test_acc_lst)
np.save(SAVE_PATH+"train_acc.npy", train_acc_lst)
np.save(SAVE_PATH+"test_acc.npy", test_acc_lst)
if len(lr_lst) == 1 and len(nb_epochs_lst) == 1:
    np.save(SAVE_PATH+"train_loss_evolv.npy", train_loss_evolv)