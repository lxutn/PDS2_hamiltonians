# -*- coding: utf-8 -*-
"""
Generates CIFAR10, CIFAR100 and STL10 datasets with the preprocessing steps presented in [2]

Based on different papers:
    [1] Stable architectures for deep neural networks. 
        Eldab Haber and Lars Ruthotto, 2017
    [2] Reversible Architectures for Arbitrarily Deep Residual Neural Networks. 
        Bo Chang, Lili Meng et al., 2018
    [3] A unified framework for Hamiltonian deep neural networks. 
        Clara Galimberti, Liang Xu and Giancarlo Ferrari Trecate, 2021
"""

# ----- Libraries ----- #
# Pytorch
import torch
from torchvision import datasets, transforms
# Utils
import sys

# ----- Constants ----- #
# CIFAR proprieties
CIFAR_SIZE = 32  # size of a side of the image (assumed sqaure)
CIFAR10_n_labels = 10  # nb of labels in CIFAR10
CIFAR100_n_labels = 100  # nb of labels in CIFAR100
CIFAR_NB_TRAIN = 50000  # nb of images in the training dataset
# mean value of the image for each color
CIFAR_MEAN_VALS = [0.49139968, 0.48215841, 0.44653091]
# std value of the image for each color
CIFAR_STD_VALS = [0.24703223, 0.24348513, 0.26158784]

# STL proprieties
STL10_SIZE = 96  # size of a side of the image (assumed sqaure)
STL10_n_labesl = 10  # nb of labels in STL10
STL10_NB_TRAIN = 5000  # nb of images in the training dataset
# mean value of the image for each color
STL10_MEAN_VALS = [0.4467, 0.4398, 0.4066]
# std value of the image for each color
STL10_STD_VALS = [0.2603, 0.2566, 0.2713]

# Preprocessing constants
NB_PIXEL_PADDING_CIFAR = 4  # nb pixel to pad around the image for CIFAR
NB_PIXEL_PADDING_STL10 = 12  # nb pixel to pad around the image for STL
PADDED_VALUE = 0  # which value to pad
PROBA_FLIP = 0.5  # proba to flip the image

# ----- Functions ----- #


def CIFAR_dataset(dataset, path, batch_size, crop_and_flip, kwargs):
    # Return the loader for the selected preprocessed CIFAR dataset and informations about the dataset
    # Parameters:
    #   dataset         dataset to load ("CIFAR10"/"CIFAR100")
    #   path            where to load/save data
    #   batch_size      size of a batch
    #   crop_and_flip   enable cropping and flipping of the training images
    #   kwargs          GPU information
    # Output:
    #   train_loader    loader for the train data
    #   test_loader     loader for the test data
    #   CIFAR_SIZE      the size of the images (one side)
    #   n_labels        the number of different labels in the dataset
    #   CIFAR_NB_TRAIN  the number of train images in the dataset

    # Preprocessing transformation
    preprocess_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(
            CIFAR_SIZE, padding=NB_PIXEL_PADDING_CIFAR, fill=PADDED_VALUE),
        transforms.RandomHorizontalFlip(p=PROBA_FLIP),
        transforms.Normalize(CIFAR_MEAN_VALS, CIFAR_STD_VALS)
    ])
    # No crop nor flip for testing
    preprocess_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN_VALS, CIFAR_STD_VALS)
    ])
    # if cropping and flipping disabled -> use the test transformation for the training
    if crop_and_flip == False:
        preprocess_train = preprocess_test

    # Create the data loaders
    if dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True,
                             download=True, transform=preprocess_train),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False,
                             transform=preprocess_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        n_labels = CIFAR10_n_labels

    elif dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=True,
                              download=True, transform=preprocess_train),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=False,
                              transform=preprocess_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        n_labels = CIFAR100_n_labels

    else:
        sys.exit("This dataset is unavailible (datasets.py)\n")

    return train_loader, test_loader, CIFAR_SIZE, n_labels, CIFAR_NB_TRAIN


def STL10_dataset(path, batch_size, crop_and_flip, kwargs):
    # Return the loader for the preprocessed STL dataset and informations about the dataset
    # Parameters:
    #   path            where to laod/save data
    #   batch_size      size of a batch
    #   crop_and_flip   enable cropping and flipping of the training images
    #   kwargs          GPU information
    # Output:
    #   train_loader    loader for the train data
    #   test_loader     loader for the test data
    #   STL10_SIZE      the size of the images (one side)
    #   n_labels        the number of different labels in the dataset
    #   STL10_NB_TRAIN  the number of train images in the dataset

    # Preprocessing transformation
    preprocess_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(
            STL10_SIZE, padding=NB_PIXEL_PADDING_STL10, fill=PADDED_VALUE),
        transforms.RandomHorizontalFlip(p=PROBA_FLIP),
        transforms.Normalize(STL10_MEAN_VALS, STL10_STD_VALS)
    ])
    # Transformation preprocessing steps for testing (no crop nor flip)
    preprocess_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN_VALS, STL10_STD_VALS)
    ])
    # if cropping and flipping disabled -> use the test transformation for the training
    if crop_and_flip == False:
        preprocess_train = preprocess_test

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.STL10('../data', split="train", download=True,
                       transform=preprocess_train),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.STL10('../data', split="test", transform=preprocess_test),
        batch_size=batch_size, shuffle=True)

    n_labels = STL10_n_labesl

    return train_loader, test_loader, STL10_SIZE, n_labels, STL10_NB_TRAIN
