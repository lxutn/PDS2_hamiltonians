# -*- coding: utf-8 -*-
"""
Plot the results obtained by AlexNet [1] 
Plot either grid searched performance or either a one run performance

Based on:
    [1] KRIZHEVSKY, Alex, SUTSKEVER, Ilya, et HINTON, Geoffrey E. 
        Imagenet classification with deep convolutional neural networks. 
        Advances in neural information processing systems, 2012, vol. 25, p. 1097-1105.
"""
# ----- Libraries ----- #
#Utils
import numpy as np
#Plot
import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')

# ----- Constants ----- #
TRAIN_SIZE = 50000
TEST_SIZE = 10000

# ----- Parameters ----- #
LOAD_FOLDER = "best_net/"

# ----- Load data ----- #
epochs = np.load(LOAD_FOLDER+"nb_epochs_lst.npy")
lrs = np.load(LOAD_FOLDER+"lr_lst.npy")
train_loss = np.load(LOAD_FOLDER+"train_loss.npy")
test_loss = np.load(LOAD_FOLDER+"test_loss.npy")
train_acc = np.load(LOAD_FOLDER+"train_acc.npy")
test_acc = np.load(LOAD_FOLDER+"test_acc.npy")
epochs = np.load(LOAD_FOLDER+"nb_epochs_lst.npy")
lrs = np.load(LOAD_FOLDER+"lr_lst.npy")

# ----- Print the performance ----- #
print(train_acc)
print(test_acc)

if len(lrs) != 1 or len(epochs) != 1:
    # ----- Print the heathmap of the grid search ----- #
    plt.figure(0)
    sns.heatmap(train_loss, xticklabels=epochs, yticklabels=lrs, annot=True)
    plt.title("Train loss")
    plt.xlabel("Nb epochs")
    plt.ylabel("Learning rate")
    plt.savefig(LOAD_FOLDER+"train_loss.jpg")
    
    plt.figure(1)
    sns.heatmap(test_loss, xticklabels=epochs, yticklabels=lrs, annot=True)
    plt.title("Test loss")
    plt.xlabel("Nb epochs")
    plt.ylabel("Learning rate")
    plt.savefig(LOAD_FOLDER+"test_loss.jpg")
    
    plt.figure(2)
    sns.heatmap(test_acc, xticklabels=epochs, yticklabels=lrs, annot=True )
    plt.title("Test accuracy")
    plt.xlabel("Nb epochs")
    plt.ylabel("Learning rate")
    plt.savefig(LOAD_FOLDER+"test_acc.jpg")
    
    plt.figure(3)
    sns.heatmap(train_acc, xticklabels=epochs, yticklabels=lrs, annot=True)
    plt.title("Train accuracy")
    plt.xlabel("Nb epochs")
    plt.ylabel("Learning rate")
    plt.savefig(LOAD_FOLDER+"train_acc.jpg")

if len(lrs) == 1 and len(epochs) == 1:
# ----- Print the evolution of the loss along epochs ----- #
    train_loss_evolv = np.load(LOAD_FOLDER+"train_loss_evolv.npy")
    epochs_val = np.arange(epochs[0])
    plt.figure(4)
    plt.title("Loss wrt the epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs_val)
    plt.plot(epochs_val,train_loss_evolv)
    plt.savefig(LOAD_FOLDER+"train_loss_evolv.jpg")