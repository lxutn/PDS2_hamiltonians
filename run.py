# -*- coding: utf-8 -*-
"""
Run the training and testing of Hamiltonians defined by parameters structures.
Save the results and implement grid search.

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
#Data
import pickle
#Ours
import params as p
from train_and_test import train_and_test
#Utils
import numpy as np
import os

# ----- Define the base problem through parameters ----- #
#Dataset
DATASET = "CIFAR10"                 #choose btw "CIFAR100" and "CIFAR10"
CIFAR_NB_TRAIN = 50000              #nb of training images in the dataset

#Utils
SAVE_PATH = "data"+"/"              #save folder
if not os.path.exists(SAVE_PATH):   #if the folder doesn't exist, create it
    os.makedirs(SAVE_PATH)
STATS_REPETS = 3                    #nb of repetition for statisitcs measures (mean, std)

#Define base network and problem from scratch:
#Learning parameters (see params.py)
learn_params = p.LearningParameters(training_steps_max=80000,
                                    batch_size=100,
                                    lr=0.1,
                                    wd=2e-4,
                                    alpha=2e-4,
                                    b_in_reg=False,
                                    h_div=True,
                                    lr_decay_at=[80,120,160],
                                    lr_decay=0.1,
                                    momentum=0.9)

#Hamiltonian parameters (see params.py)
ham_params = p.HamiltonianParameters(hamiltonianFunction="J1",
                                   n_blocks=6,
                                   ks=3,
                                   h=0.08,
                                   act="ReLU",
                                   init="Xavier")

#Network parameters (see params.py)
net_params = p.NetowrkParameters(hamParams=ham_params,
                                   nb_units=3,
                                   nb_features=list(map(int,[3, 32, 64, 112])),
                                   ks_conv=3,
                                   strd_conv=1,
                                   pd_conv=1,
                                   pooling="Avg",
                                   ks_pool=2,
                                   strd_pool=2,
                                   init="Xavier",
                                   second_final_FC=None,
                                   batchnorm_bool=False,
                                   both_border_pad=False,
                                   dropout=None)

#Or import and modify prexisting problem
# =============================================================================
# learn_params = p.best_paper_learn_params
# ham_params = p.best_paper_ham_params
# net_params = p.best_paper_net_params
# #learn_params.lr = 0.01 #change init lr for example
# =============================================================================

# ----- Save the settings of the problem ----- #
pickle.dump(learn_params, open(SAVE_PATH+"learn_params.pkl", "wb"))
pickle.dump(ham_params, open(SAVE_PATH+"ham_params.pkl", "wb"))
pickle.dump(net_params, open(SAVE_PATH+"net_params.pkl", "wb"))
pickle.dump(STATS_REPETS, open(SAVE_PATH+"STATS_REPETS.pkl", "wb"))
pickle.dump(DATASET, open(SAVE_PATH+"DATASET.pkl", "wb"))

# ----- Define GPU utilisation ----- #
use_cuda = torch.cuda.is_available()  # not no_cuda and
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
torch.set_num_threads(1)

# ----- 2D grid search ----- #
#Define the parameters along which apply a grid search
#If no grid search just let intial parameters
params_names = ["learning rate","learning rate"]
param1_values = [learn_params.lr]
param2_values = [learn_params.lr]

#Save the grid search values
pickle.dump(params_names, open(SAVE_PATH+"params_names.pkl", "wb"))
pickle.dump(param2_values, open(SAVE_PATH+"param2_values.pkl", "wb"))
pickle.dump(param1_values, open(SAVE_PATH+"param1_values.pkl", "wb"))

#Prepare the save arrays
train_acc = []
test_acc = []
train_loss = []
test_loss = []
train_acc_e = []
test_acc_e = []
train_loss_e = []
test_loss_e = []

#Apply the grid search
for i, param1 in enumerate(param1_values):
    #Add a line
    train_acc.append([])
    test_acc.append([])
    train_loss.append([])
    test_loss.append([])
    train_acc_e.append([])
    test_acc_e.append([])
    train_loss_e.append([])
    test_loss_e.append([])
    for j, param2 in enumerate(param2_values):
        #Update the parameters with the values
        print("New run with "+params_names[0]+" at "+str(param1_values[i])+" and with "\
              +params_names[1]+" at "+str(param2_values[j]))
        learn_params.lr = param1_values[i]
        learn_params.lr = param2_values[j]
        
        #Compute the nb of epochs (may change during the grid search)
        step_per_epoch = CIFAR_NB_TRAIN/learn_params.batch_size
        epochs = int(learn_params.training_steps_max/step_per_epoch)
        
        #Temporary arrays
        train_loss_e_tmp = np.zeros((STATS_REPETS, epochs))
        test_loss_e_tmp = np.zeros((STATS_REPETS, epochs))
        train_acc_e_tmp = np.zeros((STATS_REPETS, epochs))
        test_acc_e_tmp = np.zeros((STATS_REPETS, epochs))
        train_loss_tmp = np.zeros((STATS_REPETS))
        test_loss_tmp = np.zeros((STATS_REPETS))
        train_acc_tmp = np.zeros((STATS_REPETS))
        test_acc_tmp = np.zeros((STATS_REPETS))
        
        #Repeat for stats performance
        for it in range(STATS_REPETS):
            print("Iteration "+str(it+1)+" over "+str(STATS_REPETS))
            #Train and test the network 
            train_loss_e_tmp[it,:], test_loss_e_tmp[it,:], train_acc_e_tmp[it,:], test_acc_e_tmp[it,:], \
                train_loss_tmp[it], test_loss_tmp[it], train_acc_tmp[it], test_acc_tmp[it] = \
                train_and_test(DATASET, net_params, learn_params, device, kwargs)
            
        #Add to tables
        train_acc[i].append(train_acc_tmp)
        test_acc[i].append(test_acc_tmp)
        train_loss[i].append(train_loss_tmp)
        test_loss[i].append(test_loss_tmp)
        train_acc_e[i].append(train_acc_e_tmp)
        test_acc_e[i].append(test_acc_e_tmp)
        train_loss_e[i].append(train_loss_e_tmp)
        test_loss_e[i].append(test_loss_e_tmp)
        
        #Save (each time to keep values even if it crashes at one point)
        pickle.dump(train_acc, open(SAVE_PATH+"train_acc.pkl", "wb" ))
        pickle.dump(test_acc, open(SAVE_PATH+"test_acc.pkl", "wb" ))
        pickle.dump(train_loss, open(SAVE_PATH+"train_loss.pkl", "wb" ))
        pickle.dump(test_loss, open(SAVE_PATH+"test_loss.pkl", "wb" ))
        pickle.dump(train_acc_e, open(SAVE_PATH+"train_acc_e.pkl", "wb" ))
        pickle.dump(test_acc_e, open(SAVE_PATH+"test_acc_e.pkl", "wb" ))
        pickle.dump(train_loss_e, open(SAVE_PATH+"train_loss_e.pkl", "wb" ))
        pickle.dump(test_loss_e, open(SAVE_PATH+"test_loss_e.pkl", "wb" ))
            
    
