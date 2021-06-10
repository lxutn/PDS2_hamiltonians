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
    #Return the regularisation loss corresponding to the smoothness of K and b as defined on [1] [2] [3]
    #Parameters:
    #   model           model that is being trained
    #   net_params      set of information about the network strucutre (see params.py) 
    #   learn_params    set of information about the learning configuration (see params.py) 
    #   device          device on which computation have to be handled (GPU/CPU)
    
    reg_loss = 0
    
    for unit in model.units:
        #batchnorm layer do not have K or b
        if not isinstance(unit, nn.BatchNorm2d):
            K = unit.baseUnit.getK().to(device)
            b = unit.baseUnit.getb().to(device)
            
            #Choose if h will divide ([1] & [2] or multiply [3] the regularisation)
            h_reg = net_params.hamParams.h
            if learn_param.h_div:
                h_reg = 1/h_reg
            
            for j in range(unit.ham_params.n_blocks-1):
                #Add the smoothness reg of K
                reg_loss += learn_param.alpha * (torch.norm((K[:, :, :, :, j+1] - K[:, :, :, :, j]))**2 * h_reg)
                #If enabled, add the smoothness reg of b
                if learn_param.b_in_reg:
                    #eHam and TwoLayersHam have not the same shape for b -> two cases but same idea
                    if net_params.hamParams.hamiltonianFunction == "TwoLayersHam":
                        reg_loss += learn_param.alpha * (torch.norm((b[:, :, j+1] - b[:, :, j]))**2 * h_reg) 
                    else:
                        reg_loss += learn_param.alpha * (torch.norm((b[:, j+1] - b[:, j]))**2 * h_reg) 
    return reg_loss
        