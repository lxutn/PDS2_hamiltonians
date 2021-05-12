# -*- coding: utf-8 -*-
"""
Generates CIFAR10 and CIFAR100 datasets with the preprocessing steps presented in [2]

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
from torchvision import datasets, transforms
#Utils
import sys

# ----- Constants ----- #
#Dataset proprieties
CIFAR_SIZE = 32         #size of a side of the image (assumed sqaure)
CIFAR_NB_TRAIN = 50000  #nb of images in the training dataset
CIFAR_MEAN_VALS = [0.49139968, 0.48215841, 0.44653091]  #mean value of the image for each color
CIFAR_STD_VALS = [0.24703223, 0.24348513, 0.26158784]   #std value of the image for each color

#Preprocessing constants
NB_PIXEL_PADDING = 4    #nb pixel to pad around the image
PADDED_VALUE = 0        #which value to pad 
PROBA_FLIP = 0.5        #proba to flip the image

# ----- Functions ----- #

def CIFAR_dataset(dataset, path, batch_size, kwargs):
    #Return the loader for the selected CIFAR dataset with the preprocessing steps
    #Parameters:
    #   dataset is the dataset to load (choose btw "CIFAR10" and "CIFAR100")
    #   path is the path where to load the data
    #   batch_size is the batch_size
    #   kwargs contains infos about gpu and num workers
    
    #Create the transformation preprocessing steps
    preprocess_train = transforms.Compose([
        transforms.ToTensor(),                                     
        transforms.RandomCrop(CIFAR_SIZE, padding=NB_PIXEL_PADDING, fill=PADDED_VALUE),
        transforms.RandomHorizontalFlip(p=PROBA_FLIP),
        transforms.Normalize(CIFAR_MEAN_VALS, CIFAR_STD_VALS)       
    ])
    #the croping and flip is not applied to the testing dataset
    preprocess_test = transforms.Compose([
        transforms.ToTensor(),                                    
        transforms.Normalize(CIFAR_MEAN_VALS, CIFAR_STD_VALS)        
    ])
    
    #Select the dataset
    if dataset == "CIFAR10":
        #Load train data
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=preprocess_train),
                             batch_size=batch_size, shuffle=True, **kwargs)
        
        #Load test data
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=preprocess_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        
    elif dataset == "CIFAR100":
        #Load train data
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=True, download=True, transform=preprocess_train),
                             batch_size=batch_size, shuffle=True, **kwargs)
        
        #Load test data
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=False, transform=preprocess_test),
            batch_size=batch_size, shuffle=True, **kwargs)
    else:
        sys.exit("This dataset is unavailible (datasets.py)\n")
        
    return train_loader, test_loader