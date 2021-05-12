# -*- coding: utf-8 -*-
"""
Define the regularisation computation of the Hamiltonian units as defined in the papers.

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

# ----- Functions ----- #
def compute_reg(model, net_params, learn_param, device):
    #add the regularisation loss
    #Parameters:
    #   model is the model that is trained
    #   net_params are the parameters of this model
    #   learn_params are the learning parameters
    #   device is the device on which the operation have to be handled
    reg_loss = 0
    #iterate through the units
    for unit in model.units:
        #do not consider the batchnorm layer
        if not isinstance(unit, nn.BatchNorm2d):
            #Get the weights
            K = unit.baseUnit.getK().to(device)
            b = unit.baseUnit.getb().to(device)
            
            #Choose if h will devide or multiply the added loss
            h_reg = net_params.hamParams.h
            if learn_param.h_div:
                h_reg = 1/h_reg
            
            #Iterate through the blocks of the unit
            for j in range(unit.ham_params.n_blocks-1):
                #Add the regularization from K
                reg_loss += learn_param.alpha * (torch.norm((K[:, :, :, :, j+1] - K[:, :, :, :, j]))**2 * h_reg)
                #Add the regularisation from b (if asked)
                if learn_param.b_in_reg:
                    #eHam and TwoLayersHam have not the same shape for b
                    if net_params.hamParams.hamiltonianFunction == "TwoLayersHam":
                        reg_loss += learn_param.alpha * (torch.norm((b[:, :, j+1] - b[:, :, j]))**2 * h_reg) 
                    else:
                        reg_loss += learn_param.alpha * (torch.norm((b[:, j+1] - b[:, j]))**2 * h_reg) 
    return reg_loss
        