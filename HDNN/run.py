# -*- coding: utf-8 -*-
"""
Run the training and testing of a Hamiltonians network and save the performance
Note that the saving format may look strange, but it is the same format as in grid search (where it makes sense)

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
#Ours
import params as p
from train_and_test import train_and_test
#Utils
import numpy as np
import os
import pickle

# ----- Define the base problem through parameters ----- #
#Dataset
DATASET = "CIFAR10"                  #choose btw "CIFAR100", "CIFAR10" and "STL10"
NB_TRAIN_IMGS = 50000                #nb of images in the dataset
#Utils
STATS_REPETS = 1                    #nb of repetition for statisitcs measures (mean, std)
SAVE_PATH = "data"+"/"              #save folder
if not os.path.exists(SAVE_PATH):   #if the folder doesn't exist, create it
    os.makedirs(SAVE_PATH)

#Load the network, Hamiltonian and problem to run (from params.py)
learn_params = p.J1ReLU_learn_params
net_params = p.J1ReLU_net_params
ham_params = net_params.hamParams

#Or one can also create here a new one
# =============================================================================
# best_paper_ham_params = p.HamiltonianParameters(hamiltonianFunction="TwoLayersHam",
#                                    n_blocks=6,
#                                    ks=3,
#                                    h=0.2,
#                                    act="ReLU",
#                                    init="Xavier")
# 
# best_paper_net_params = p.NetowrkParameters(hamParams=best_paper_ham_params,
#                                    nb_units=3,
#                                    nb_features=list(map(int,[3, 32, 64, 112])),
#                                    ks_conv=3,
#                                    strd_conv=1,
#                                    pd_conv=1,
#                                    pooling="Avg",
#                                    ks_pool=2,
#                                    strd_pool=2,
#                                    init="Xavier",
#                                    second_final_FC=None,
#                                    batchnorm_bool=True,
#                                    both_border_pad=True,
#                                    dropout=[0.1,0.2])
# 
# best_paper_learn_params = p.LearningParameters(training_steps_max=80000,
#                                     batch_size=100,
#                                     lr=0.1,
#                                     lr_decay_at=[80,120,150],
#                                     lr_decay=0.1,
#                                     wd=2e-4,
#                                     alpha=2e-4,
#                                     b_in_reg=True,
#                                     h_div=False,
#                                     momentum=0.9,
#                                     crop_and_flip=True,
#                                     optimiser="SGD")
# =============================================================================

# ----- Save the settings of the problem ----- #
pickle.dump(learn_params, open(SAVE_PATH+"learn_params.pkl", "wb"))
pickle.dump(ham_params, open(SAVE_PATH+"ham_params.pkl", "wb"))
pickle.dump(net_params, open(SAVE_PATH+"net_params.pkl", "wb"))
pickle.dump(STATS_REPETS, open(SAVE_PATH+"STATS_REPETS.pkl", "wb"))
pickle.dump(DATASET, open(SAVE_PATH+"DATASET.pkl", "wb"))

# ----- Define GPU utilisation ----- #
use_cuda = torch.cuda.is_available() 
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
torch.set_num_threads(1)

# ----- Define save tables ----- #
#Save void information just to have the same format as grid search
pickle.dump(["No grid"], open(SAVE_PATH+"params_names.pkl", "wb"))
pickle.dump([0], open(SAVE_PATH+"param2_values.pkl", "wb"))
pickle.dump([0], open(SAVE_PATH+"param1_values.pkl", "wb"))

#Compute the nb of epochs (may change during the grid search)
step_per_epoch = NB_TRAIN_IMGS/learn_params.batch_size
epochs = int(learn_params.training_steps_max/step_per_epoch)

#Prepare the save arrays
train_acc = []
test_acc = []
train_loss = []
test_loss = []
train_acc_e = []
test_acc_e = []
train_loss_e = []
test_loss_e = []

#Add the first line
train_acc.append([])
test_acc.append([])
train_loss.append([])
test_loss.append([])
train_acc_e.append([])
test_acc_e.append([])
train_loss_e.append([])
test_loss_e.append([])

#Temporary arrays
train_loss_e_tmp = np.zeros((STATS_REPETS, epochs))
test_loss_e_tmp = np.zeros((STATS_REPETS, epochs))
train_acc_e_tmp = np.zeros((STATS_REPETS, epochs))
test_acc_e_tmp = np.zeros((STATS_REPETS, epochs))
train_loss_tmp = np.zeros((STATS_REPETS))
test_loss_tmp = np.zeros((STATS_REPETS))
train_acc_tmp = np.zeros((STATS_REPETS))
test_acc_tmp = np.zeros((STATS_REPETS))
      
# ----- Train the network and test it ----- #
    
#Repeat for stats performance
for it in range(STATS_REPETS):
    print("Iteration "+str(it+1)+" over "+str(STATS_REPETS))
    #Train and test the network 
    train_loss_e_tmp[it,:], test_loss_e_tmp[it,:], train_acc_e_tmp[it,:], test_acc_e_tmp[it,:], \
        train_loss_tmp[it], test_loss_tmp[it], train_acc_tmp[it], test_acc_tmp[it] = \
        train_and_test(DATASET, net_params, learn_params, device, kwargs)
   
# ----- Save the results ----- # 
   
#Add results to tables
train_acc[0].append(train_acc_tmp)
test_acc[0].append(test_acc_tmp)
train_loss[0].append(train_loss_tmp)
test_loss[0].append(test_loss_tmp)
train_acc_e[0].append(train_acc_e_tmp)
test_acc_e[0].append(test_acc_e_tmp)
train_loss_e[0].append(train_loss_e_tmp)
test_loss_e[0].append(test_loss_e_tmp)

#Save
pickle.dump(train_acc, open(SAVE_PATH+"train_acc.pkl", "wb" ))
pickle.dump(test_acc, open(SAVE_PATH+"test_acc.pkl", "wb" ))
pickle.dump(train_loss, open(SAVE_PATH+"train_loss.pkl", "wb" ))
pickle.dump(test_loss, open(SAVE_PATH+"test_loss.pkl", "wb" ))
pickle.dump(train_acc_e, open(SAVE_PATH+"train_acc_e.pkl", "wb" ))
pickle.dump(test_acc_e, open(SAVE_PATH+"test_acc_e.pkl", "wb" ))
pickle.dump(train_loss_e, open(SAVE_PATH+"train_loss_e.pkl", "wb" ))
pickle.dump(test_loss_e, open(SAVE_PATH+"test_loss_e.pkl", "wb" ))
            
    
