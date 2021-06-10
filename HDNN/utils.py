# -*- coding: utf-8 -*-
"""
Utils functions for the NN

Based on different papers:
    [1] Stable architectures for deep neural networks. 
        Eldab Haber and Lars Ruthotto, 2017
    [2] Reversible Architectures for Arbitrarily Deep Residual Neural Networks. 
        Bo Chang, Lili Meng et al., 2018
    [3] A unified framework for Hamiltonian deep neural networks. 
        Clara Galimberti, Liang Xu and Giancarlo Ferrari Trecate, 2021
"""

# ----- Libraries ----- #
import torch
import sys
import numpy as np

# ----- Utils functions ----- #
def nb_param_model(model):
    #Return the nb of trainable parameters of a model (model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initWeights(layer, init):
    #Initialise the weights of a standard Pytorch layer in the desired way
    #Parameters:
    #   layer           layer to initialise
    #   init            intialisation method to use ("Xavier", "Zeros", "Ones" or "Normal")
    
    if init == "Xavier":
        torch.nn.init.xavier_uniform_(layer.weight)
    elif init == "Zeros":
        layer.weight.data.fill_(0.0)
    elif init == "Ones":
        layer.weight.data.fill_(1.0)
    elif init == "Normal":
        layer.weight.data.normal_(0,0.01) #default would be mean=0 and std=1
    else:
        sys.exit("This initialisation for the layers in unknown (utils.py)")
        
def init_K_and_b(init, nf, ks, n_blocks, hamiltonian):
    #Initialise the the K and b matrix of the Hamiltoninan networks in the desired way
    #Warning: does not output the same number of elements if TwoLayers or J1 or J2
    #Parameters:
    #   init            intialisation method to use ("Xavier", "Zeros", "Ones" or "Normal")
    #   nf              number of channels in input
    #   ks              kernel size of the K conv operator
    #   n_blocks        number of hamiltonian blocks
    #   hamiltonian     indicates which Hamiltonian is used
        
    #Two layers Hamiltonian need to receive K1 and K2 to create a sym K, not the case for J1 and J2
    if hamiltonian == "TwoLayersHam": 
        if init == "Xavier":
            #Not really a Xavier here but similar idea 
            #This come from a similar project done by the authors of [2] (see report for more information)
            std = np.sqrt(2/(nf*ks*ks)) 
            K1 = torch.normal(mean=0, std=std ,size=(nf // 2, nf // 2,ks,ks, n_blocks))
            K2 = torch.normal(mean=0, std=std ,size=(nf // 2, nf // 2, ks, ks, n_blocks))
            b1 = torch.normal(mean=0, std=std ,size=(nf // 2, 1, n_blocks))
            b2 = torch.normal(mean=0, std=std ,size=(nf // 2, 1, n_blocks))
        elif init == "Ones":
            K1 = torch.ones(nf // 2, nf // 2, ks, ks, n_blocks)
            K2 = torch.ones(nf // 2, nf // 2, ks, ks, n_blocks)
            b1 = torch.ones(nf // 2, 1, n_blocks)
            b2 = torch.ones(nf // 2, 1, n_blocks)
        elif init == "Zeros":
            K1 = torch.zeros(nf // 2, nf // 2, ks, ks, n_blocks)
            K2 = torch.zeros(nf // 2, nf // 2, ks, ks, n_blocks)
            b1 = torch.zeros(nf // 2, 1, n_blocks)
            b2 = torch.zeros(nf // 2, 1, n_blocks)
        elif init == "Normal":
            K1 = torch.normal(mean=0,std=0.01,size=(nf // 2, nf // 2, ks, ks, n_blocks))
            K2 = torch.normal(mean=0,std=0.01,size=(nf // 2, nf // 2, ks, ks, n_blocks))
            b1 = torch.normal(mean=0,std=0.01,size=(nf // 2, 1, n_blocks))
            b2 = torch.normal(mean=0,std=0.01,size=(nf // 2, 1, n_blocks))
        else:
            sys.exit("Unknown Hamitonian initialisation (utils.py)\n")
            
        return K1, K2, b1, b2
    
    #If J1 or J2 no condition on K
    else:
   
        if init == "Xavier":
            #Not really a Xavier here but similar idea 
            #This come from a similar project done by the authors of [2]
            std = np.sqrt(2/(nf*ks*ks)) 
            K = torch.normal(mean=0, std=std ,size=(nf, nf, ks, ks, n_blocks))
            b = torch.randn(nf, n_blocks)
        elif init == "Ones":
            K = torch.ones(nf, nf, ks, ks, n_blocks)
            b = torch.ones(nf, n_blocks)
        elif init == "Zeros":
            K = torch.zeros(nf, nf, ks, ks, n_blocks)
            b = torch.zeros(nf, n_blocks)
        elif init == "Normal":
            K = torch.normal(mean=0,std=0.01,size=(nf, nf, ks, ks, n_blocks))
            b = torch.normal(mean=0,std=0.01,size=(nf, n_blocks))
        else:
            sys.exit("Unknown Hamitonian initialisation (utils.py)\n")
            
        return K, b
         
def createJacobian(jacobian_name, nf):
    #Return the selected Jacobian from [3]
    #Parameters:
    #   jacobian_name   the name of the Jacobian to create (see [3])
    #   nf              number of channels in input
    
    if jacobian_name == 'J1':
        j_identity = torch.eye(nf//2)
        j_zeros = torch.zeros(nf//2, nf//2)
        J = torch.cat((torch.cat((j_zeros, j_identity), 0), torch.cat((- j_identity, j_zeros), 0)), 1)
    elif jacobian_name == 'J2':
        j_aux = np.hstack((np.zeros(1), np.ones(nf-1)))
        J = j_aux
        for j in range(nf-1):
            j_aux = np.hstack((-1 * np.ones(1), j_aux[:-1]))
            J = np.vstack((J, j_aux))
        J = torch.tensor(J, dtype=torch.float32)
    else:
        sys.exit("Unknown eHamiltonian function (utils.py)\n")
        
    return J
            
        
            
            