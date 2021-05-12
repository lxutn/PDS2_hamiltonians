# -*- coding: utf-8 -*-
"""
Implement training and testing of Hamiltonians ([1], [2] or [3]) defined inside of the structure defined 
on [2], and some enhancement.

Based on different papers:
    [1] Stable architectures for deep neural networks. 
        Eldab Haber and Lars Ruthotto, 2017
    [2] Reversible Architectures for Arbitrarily Deep Residual Neural Networks. 
        Bo Chang, Lili Meng et al., 2018
    [3] A unified framework for Hamiltonian deep neural networks. 
        Clara Galimberti, Liang Xu and Giancarlo Ferrari Trecate, 2021
"""

# ----- Libraries ----- #
#Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
#Ours
from networks import Network
from datasets import CIFAR_dataset
from utils import nb_param_model
from regularisation import compute_reg
#Utils
import numpy as np
import time
import sys

# ----- Constants ----- #
CIFAR_SIZE = 32                     #size of a side of a CIFAR image (assumed square)
CIFAR_NB_TRAIN = 50000              #nb of training images in the dataset
PRINT_FREQ = 100                    #print frequency (in nb of batch)

# ----- Functions ----- #
def train(epoch, model, optimizer, criterion, train_loader, learn_param, net_params, device):
    #train the given model with the given data
    #Parameters
    #   epoch is the # of the current epoch
    #   model is the model to train
    #   optimiser is the optimiser which will modify the weight
    #   criterion is the criterion used to compute the loss
    #   train_loader is the dataset on which the net will be trained
    #   learn_param are the learning parameters (lr, ...)
    #   net_param are the parameters of the network (which hamiltonian, ...)
    #   deivce is the device on which the operation will be done
    
    model.train()
    #Iterate through the batches
    for batch_idx, (data, target) in enumerate(train_loader):
        #load data
        data, target = data.requires_grad_().to(device), target.to(device)
        
        #apply the model and compute loss
        optimizer.zero_grad()
        output = model(data).to(device)
        loss = criterion(output ,target).to(device)

        #add regularisation term
        loss += compute_reg(model, net_params, learn_param, device)
        
        #update the network
        loss.backward()
        optimizer.step()
        
        #print perf
        if batch_idx % PRINT_FREQ == 0:   
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct, len(data)))

def test(model, criterion, device, loader, test_type):
    #test the given model with the given data and returns the performance
    #Parameters:
    #   model is the model to train
    #   criterion is the criterion used to compute the loss
    #   loader is the dataset on which the net will be tested
    #   deivce is the device on which the operation will be done
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        #iterate in the dataset and test if correct
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data).to(device)           
            test_loss += criterion(output, target).item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    #normalize
    test_loss /= len(loader.dataset)
    correct_perc = correct / len(loader.dataset)
    
    print('\n'+test_type+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    
    return test_loss, correct_perc


def train_and_test(DATASET, net_params, learn_params, device, kwargs):
    #train and test on a selected dataset and return its perfromance
    #Parameters:
    #   DATASET indicates which dataset will be used for training ("CIFAR10" or "CIFAR100")
    #   net_params are the parameters that define the network
    #   learn_params are the parameters that define the learning problem (lr, ...)
    #   device is the device on which the operation will take place
    #   kwargs indicates GPU information, like the nb of workers
    
    #Randomise the seed
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    #Dataset selection -> load the data and deduce the nb of labels
    print(DATASET+" selected")      #nb of labels for classification
    train_loader, test_loader = CIFAR_dataset(DATASET, '../data', learn_params.batch_size, kwargs)
    if DATASET == "CIFAR10":
        n_labels = 10               
    elif DATASET == "CIFAR100":
        n_labels = 100            
    else:
        sys.exit("The dataset is not availible") 
        
    #Compute the nb of epochs to do
    step_per_epoch = CIFAR_NB_TRAIN/learn_params.batch_size
    epochs = int(learn_params.training_steps_max/step_per_epoch)
    
    #create the arrays to stock the performance
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    train_acc = np.zeros(epochs)
    test_acc = np.zeros(epochs)
    final_train_loss = 0
    final_test_loss = 0
    final_train_acc = 0
    final_test_acc = 0
        
    #Define the net model
    model = Network(net_params, n_labels, CIFAR_SIZE, device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learn_params.lr, 
                          weight_decay=learn_params.wd, momentum=learn_params.momentum)
    scheduler = MultiStepLR(optimizer, milestones=learn_params.lr_decay_at, 
                            gamma=learn_params.lr_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    
    #Train the model
    print("TRAINING START")
    print("There will be "+str(epochs)+" epochs")
    print("Model has "+str(nb_param_model(model))+" parameters")
    for epoch in range(1, epochs + 1):
        start_ep = time.time()
        train(epoch, model, optimizer, criterion, train_loader, learn_params, net_params, device)
        test_loss[epoch-1], test_acc[epoch-1] = test(model, criterion, device, test_loader, "Test")
        train_loss[epoch-1], train_acc[epoch-1] = test(model, criterion, device, train_loader, "Train")
        scheduler.step()
        finish_ep = time.time()
        print("Epoch "+str(epoch)+" lasted "+str(finish_ep-start_ep)+" s\n")
    
    #Test the model
    print("TESTING START")
    final_train_loss, final_train_acc = test(model,criterion, device, train_loader,"Train")
    final_test_loss, final_test_acc = test(model,criterion, device, test_loader,"Test")
    
    #Return the final performance and the performance along epochs
    return train_loss, test_loss, train_acc, test_acc, \
           final_train_loss, final_test_loss, final_train_acc, final_test_acc
        
        