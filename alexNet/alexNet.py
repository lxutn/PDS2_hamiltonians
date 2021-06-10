# -*- coding: utf-8 -*-
"""
Load the pretrained AlexNet [1] from PyTorch and retrains it for CIFAR-10
Does a grid search along learning rate (lr_lst) and number of epochs (nb_epochs_lst)
To train just a network do agrid search along 1 parameter in each

Based on:
    [1] KRIZHEVSKY, Alex, SUTSKEVER, Ilya, et HINTON, Geoffrey E. 
        Imagenet classification with deep convolutional neural networks. 
        Advances in neural information processing systems, 2012, vol. 25, p. 1097-1105.
"""

# ----- Libraries ----- #
#Pytorch
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
#Utils
import time
import numpy as np

# ----- Constants & Parameters ----- #
#Training infos
batch_size = 128        #batch size
lr_lst = [1e-2]         #list of lr to test
nb_epochs_lst = [25]    #list of nb of epochs to test
momentum = 0.9          #SGD momentum 
gamma = 0.1             #lr decay
ep_lr_red = 15          #epoch at which the lr decay will happen

#Datasets/Preprocessing stats
NB_CLASSES = 10
CIFAR_MEAN = [0.485, 0.456, 0.406]
CIFAR_STD = [0.229, 0.224, 0.225]
RESIZE_W = 256          #size to reach by padding before cropping (from PyTorch doc)
MIN_INPUT_W = 224       #minimum input for the pretrained AlexNet dataset

#GPU infos
use_cuda = torch.cuda.is_available() 
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
torch.set_num_threads(1)

#Utils
LOSS_DISPLAY_RATE = 128*2-1 #frequency (in term of batch) at which the loss is printed
SAVE_PATH = "save_perf/"

# ----- Functions ----- #
def nb_param_model(model):
    #Return the nb of trainable parameters of a model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----- Loading and preprocessing the data ----- #
#Define transformations (no crop for test)
preprocess_train = transforms.Compose([
    transforms.Resize(RESIZE_W),         
    transforms.CenterCrop(MIN_INPUT_W),     
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
])
preprocess_test = transforms.Compose([
    transforms.Resize(MIN_INPUT_W),          
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
])

#Load data
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=True, download=True, 
                                    transform=preprocess_train), 
                                    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=False, 
                                    transform=preprocess_test),
                                    batch_size=batch_size, shuffle=False, **kwargs)

# ----- Grid search ----- #
#Define the save table
train_acc_lst = np.zeros((len(lr_lst),len(nb_epochs_lst)))
test_acc_lst = np.zeros((len(lr_lst),len(nb_epochs_lst)))
train_loss_lst = np.zeros((len(lr_lst),len(nb_epochs_lst)))
test_loss_lst = np.zeros((len(lr_lst),len(nb_epochs_lst)))
if len(lr_lst) == 1 and len(nb_epochs_lst) == 1:
    train_loss_evolv = np.zeros(nb_epochs_lst[0])
    
for lr_i, lr in enumerate(lr_lst):
    for e_i, nb_epochs in enumerate(nb_epochs_lst):
        # ----- Define the problem (model, optimiser, ...) ----- #
        print("\nProblem with "+str(nb_epochs)+" epochs and "+str(lr)+" as lr")
        #Load the pretrained model (force redownload to avoid infering with the grid search)
        AlexNet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True, force_reload =True)
        
        #Updating the last layer to fit the number of classes
        AlexNet.classifier[6] = nn.Linear(4096,NB_CLASSES)
        print(nb_param_model(AlexNet))
        
        #Bring the model to the GPU
        AlexNet.to(device)
        
        #Loss fct
        criterion = nn.CrossEntropyLoss().to(device)
        
        #Optimiser
        optimizer = optim.SGD(AlexNet.parameters(), lr=lr, momentum=momentum)
        scheduler = StepLR(optimizer, step_size=ep_lr_red, gamma=gamma)
        
        # ----- Train the model ----- #
        print("TRAINING START")
        start_train = time.time()
        AlexNet.train()
        for epoch in range(int(nb_epochs)):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.requires_grad_().to(device), target.to(device)
                optimizer.zero_grad()
                output = AlexNet(data)
                loss = criterion(output ,target)
                loss.backward()
                optimizer.step()
                
                if(batch_idx%LOSS_DISPLAY_RATE)==0:
                    print("At epoch "+str(epoch)+" and batch_idx "+str(batch_idx)+" loss is " \
                          +str(loss.item()))
                    
            if len(lr_lst) == 1 and len(nb_epochs_lst) == 1:
                train_loss_evolv[epoch] = loss
            scheduler.step()
                    
        finish_train = time.time()
        print("The training lasted "+str(finish_train-start_train)+" s")
        
        
        # ----- Test the model ----- #
        print("TESTING START")
        #On the training dataset
        AlexNet.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = AlexNet(data)        
                train_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = correct/len(train_loader.dataset)
        
        #On the testing dataset
        AlexNet.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = AlexNet(data)          
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc = correct/len(test_loader.dataset)
        
        #Stock the results in the grid save structures
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
np.save(SAVE_PATH+"test_loss.npy", test_loss_lst)
np.save(SAVE_PATH+"train_acc.npy", train_acc_lst)
np.save(SAVE_PATH+"test_acc.npy", test_acc_lst)
if len(lr_lst) == 1 and len(nb_epochs_lst) == 1:
    np.save(SAVE_PATH+"train_loss_evolv.npy", train_loss_evolv)