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
from datasets import CIFAR_dataset, STL10_dataset
from utils import nb_param_model
from regularisation import compute_reg
#Utils
import numpy as np
import time
import sys

# ----- Constants ----- #
PRINT_FREQ = 100            #print frequency (in nb of batch)
ERR_ACC = 0                 #error accuracy to set if crahsed
ERR_LOSS = 1                #error loss to use if crahsed

# ----- Functions ----- #
def train(epoch, model, optimizer, criterion, train_loader, learn_param, net_params, device):
    #Train the given model during one epoch, returns True if the network crashed
    #Parameters
    #   epoch           current epoch number
    #   model           model to train
    #   optimiser       Adam or SGD
    #   criterion       Crossentropy / other
    #   train_loader    data to train on
    #   learn_param     set of information about the learning configuration (see params.py) 
    #   net_param       set of information about the network strucutre (see params.py) 
    #   device          device on which computation have to be handled (GPU/CPU)
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.requires_grad_().to(device), target.to(device)
        
        #apply the model and compute loss
        optimizer.zero_grad()
        output = model(data).to(device)
        loss = criterion(output ,target).to(device)

        #add regularisation term
        loss += compute_reg(model, net_params, learn_param, device)
        
        #verify that there is no NaN or infinte loss -> if it is the case quit and return True 
        if int((output != output).sum()) > 0:
            print("Output of the model had NaN or Infinite values (train_and_test.py)\n") 
            return True
        if loss > 10**15:
            print("Loss is much too big (train_and_test.py)\n")
            return True
        
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
            
    #If reached this line the network did not crahsed -> output false
    return False

def test(model, criterion, device, loader, test_type):
    #Test the given model and return the achieved performance (accuracy and loss)
    #Parameters:
    #   model           model to train
    #   criterion       Crossentropy / other
    #   train_loader    data to test on
    #   test_type       indicates if training or testing dataset ("Test" or "Train")
    #   device          device on which computation have to be handled (GPU/CPU)
    
    model.eval()
    test_loss = 0
    correct = 0
    
    #Test the network
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data).to(device)           
            test_loss += criterion(output, target).item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    #Normalise
    test_loss /= len(loader.dataset)
    correct_perc = correct / len(loader.dataset)
    
    #Print
    print('\n'+test_type+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    
    return test_loss, correct_perc


def train_and_test(DATASET, net_params, learn_params, device, kwargs):
    #Train and test a network on the specified condition and returns the performance
    #Parameters:
    #   DATASET         which dataset to use ("CIFAR10", "CIFAR100" or "STL10")
    #   net_params      set of information about the network strucutre (see params.py) 
    #   learn_params    set of information about the learning configuration (see params.py) 
    #   device          device on which computation have to be handled (GPU/CPU)
    #   kwargs          GPU information (nb of workers, ...)
    
    has_crashed = False
    
    #Randomise the seed
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    #Dataset loading 
    print(DATASET+" selected")
    if DATASET == "CIFAR10":
        train_loader, test_loader, img_size, n_labels, n_train_img = CIFAR_dataset(DATASET, 
                    '../data', learn_params.batch_size, learn_params.crop_and_flip, kwargs)       
    elif DATASET == "CIFAR100":
        train_loader, test_loader, img_size, n_labels, n_train_img = CIFAR_dataset(DATASET, 
                    '../data', learn_params.batch_size, learn_params.crop_and_flip, kwargs)
    elif DATASET == "STL10":
        train_loader, test_loader, img_size, n_labels, n_train_img = STL10_dataset('../data',
                    learn_params.batch_size, kwargs, learn_params.crop_and_flip)     
    else:
        sys.exit("The dataset is not availible (train_and_test.py)\n") 
        
    #Compute the nb of epochs will be done
    step_per_epoch = n_train_img/learn_params.batch_size
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
        
    #Define the problem (network, optimsier, ...)
    model = Network(net_params, n_labels, img_size, device).to(device)
    if(learn_params.optimiser == "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=learn_params.lr, weight_decay=learn_params.wd, 
                              momentum=learn_params.momentum)
    elif(learn_params.optimiser == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=learn_params.lr, weight_decay=learn_params.wd,
                               betas=learn_params.momentum)
    else:
        sys.exit("The dataset is not optimiser (train_and_test.py)\n") 
        
    scheduler = MultiStepLR(optimizer, milestones=learn_params.lr_decay_at, 
                            gamma=learn_params.lr_decay)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    #Train the model
    print("TRAINING START")
    print("There will be "+str(epochs)+" epochs")
    print("Model has "+str(nb_param_model(model))+" parameters")
    for epoch in range(1, epochs + 1):
        start_ep = time.time()
        
        #If it has not crashed train the network
        if not has_crashed:
            has_crashed = train(epoch, model, optimizer, criterion, train_loader, 
                                learn_params, net_params, device)
        
        #If it has not crashed test the network and update lr if needed
        if not has_crashed:
            test_loss[epoch-1], test_acc[epoch-1] = test(model, criterion, device, test_loader, "Test")
            train_loss[epoch-1], train_acc[epoch-1] = test(model, criterion, device, train_loader, "Train")
            scheduler.step()
            finish_ep = time.time()
            print("Epoch "+str(epoch)+" lasted "+str(finish_ep-start_ep)+" s\n")
        
        #if it has crashed just put extreme values in loss and accuracy
        else: 
            test_loss[epoch-1], test_acc[epoch-1] = ERR_LOSS, ERR_ACC
            train_loss[epoch-1], train_acc[epoch-1] = ERR_LOSS, ERR_ACC
        
    #Test the model
    if not has_crashed:
        print("TESTING START")
        final_train_loss, final_train_acc = test(model,criterion, device, train_loader,"Train")
        final_test_loss, final_test_acc = test(model,criterion, device, test_loader,"Test")
    #if it has crashed just put extreme values in loss and accuracy
    else: 
        final_train_loss, final_train_acc = ERR_LOSS, ERR_ACC
        final_test_loss, final_test_acc = ERR_LOSS, ERR_ACC
    
    #save the model
    torch.save(model.state_dict(),"../NET.pt")
    
    #Return the final performance and the performance along epochs
    return train_loss, test_loss, train_acc, test_acc, \
           final_train_loss, final_test_loss, final_train_acc, final_test_acc
        
        