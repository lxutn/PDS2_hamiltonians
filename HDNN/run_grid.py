# -*- coding: utf-8 -*-
"""
Run a grid search over a Hamiltonian network and save the performance. 2D grid search
In lines 97-99 one can define the values of the parameters that will be grid searched
Change in lines 135-136 the parameters that has to be modified

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
# Ours
import params as p
# Utils
import numpy as np
import os
import pickle

# ----- Define the base problem through parameters ----- #
# Dataset
DATASET = "CIFAR10"  # choose btw "CIFAR100", "CIFAR10" and "STL10"
NB_TRAIN_IMGS = 50000  # nb of images in the dataset
# Utils
STATS_REPETS = 1  # nb of repetition for statisitcs measures (mean, std)
SAVE_PATH = "data"+"/"  # save folder
if not os.path.exists(SAVE_PATH):  # if the folder doesn't exist, create it
    os.makedirs(SAVE_PATH)

# Load the network, Hamiltonian and problem to run (from params.py)
learn_params = p.best_paper_learn_params
net_params = p.best_paper_net_params
ham_params = net_params.hamParams

# Or one can also create here a new one
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
use_cuda = torch.cuda.is_available()  # not no_cuda and
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
torch.set_num_threads(1)

# ----- Grid search Parameters ----- #
# Define the parameters along which apply a grid search
# for a 1D grid search just iterate the second dimension through a fixed parameter
# for example param2_values = [learn_params.lr]
params_names = ["batch size", "learning rate"]
param1_values = [learn_params.batch_size]
param2_values = [learn_params.lr]

# Save the grid search values
pickle.dump(params_names, open(SAVE_PATH+"params_names.pkl", "wb"))
pickle.dump(param2_values, open(SAVE_PATH+"param2_values.pkl", "wb"))
pickle.dump(param1_values, open(SAVE_PATH+"param1_values.pkl", "wb"))

# ----- Save arrays ----- #
train_acc = []
test_acc = []
train_loss = []
test_loss = []
train_acc_e = []
test_acc_e = []
train_loss_e = []
test_loss_e = []

# ----- Grid search ----- #
print("Grid over "+params_names[0]+" and "+params_names[1])
print("Values of 1st param: "+str(param1_values))
print("Values of 2nd param: "+str(param2_values))

for i, param1 in enumerate(param1_values):
    # Add a line
    train_acc.append([])
    test_acc.append([])
    train_loss.append([])
    test_loss.append([])
    train_acc_e.append([])
    test_acc_e.append([])
    train_loss_e.append([])
    test_loss_e.append([])
    for j, param2 in enumerate(param2_values):
        # ----- Update the network's parameters ----- #
        print("New run with "+params_names[0]+" at "+str(param1_values[i])+" and with "
              + params_names[1]+" at "+str(param2_values[j]))
        learn_params.batch_size = param1_values[i]
        learn_params.lr = param2_values[j]

        # Update hamiltonian in the network (in case it has been modified by the grid)
        net_params.hamParams = ham_params

        # Compute the nb of epochs (may change during the grid search)
        step_per_epoch = NB_TRAIN_IMGS/learn_params.batch_size
        epochs = int(learn_params.training_steps_max/step_per_epoch)

        # Temporary arrays
        train_loss_e_tmp = np.zeros((STATS_REPETS, epochs))
        test_loss_e_tmp = np.zeros((STATS_REPETS, epochs))
        train_acc_e_tmp = np.zeros((STATS_REPETS, epochs))
        test_acc_e_tmp = np.zeros((STATS_REPETS, epochs))
        train_loss_tmp = np.zeros((STATS_REPETS))
        test_loss_tmp = np.zeros((STATS_REPETS))
        train_acc_tmp = np.zeros((STATS_REPETS))
        test_acc_tmp = np.zeros((STATS_REPETS))

        # Repeat for stats performance
        for it in range(STATS_REPETS):
            print("Iteration "+str(it+1)+" over "+str(STATS_REPETS))
            # Train and test the network
            train_loss_e_tmp[it, :], test_loss_e_tmp[it, :], train_acc_e_tmp[it, :], test_acc_e_tmp[it, :], \
                train_loss_tmp[it], test_loss_tmp[it], train_acc_tmp[it], test_acc_tmp[it] = \
                train_and_test(DATASET, net_params,
                               learn_params, device, kwargs)

        # Add to tables
        train_acc[i].append(train_acc_tmp)
        test_acc[i].append(test_acc_tmp)
        train_loss[i].append(train_loss_tmp)
        test_loss[i].append(test_loss_tmp)
        train_acc_e[i].append(train_acc_e_tmp)
        test_acc_e[i].append(test_acc_e_tmp)
        train_loss_e[i].append(train_loss_e_tmp)
        test_loss_e[i].append(test_loss_e_tmp)

        # Save after each combination for security in case it shut downs btw the end
        pickle.dump(train_acc, open(SAVE_PATH+"train_acc.pkl", "wb"))
        pickle.dump(test_acc, open(SAVE_PATH+"test_acc.pkl", "wb"))
        pickle.dump(train_loss, open(SAVE_PATH+"train_loss.pkl", "wb"))
        pickle.dump(test_loss, open(SAVE_PATH+"test_loss.pkl", "wb"))
        pickle.dump(train_acc_e, open(SAVE_PATH+"train_acc_e.pkl", "wb"))
        pickle.dump(test_acc_e, open(SAVE_PATH+"test_acc_e.pkl", "wb"))
        pickle.dump(train_loss_e, open(SAVE_PATH+"train_loss_e.pkl", "wb"))
        pickle.dump(test_loss_e, open(SAVE_PATH+"test_loss_e.pkl", "wb"))
