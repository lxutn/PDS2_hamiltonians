# -*- coding: utf-8 -*-
"""
Define different Hamiltonians basic blocks proposed in [1], [2] and [3]; and some enhancement.

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
import torch.nn.functional as F
#Utils
import numpy as np
import sys

# ----- Hamiltonians definition ----- #
class TwoLayerHamiltonian(nn.Module):
    # Two-layer Hamiltonian based on the Chang and Meng paper [2]
    # General ODE: \dot{y} = J(y,t) K(t) \act( K^T(t) y(t) + b(t) )
    # Constraints:
    #   K(t) = [ 0 K_1(t) ; K_2(t) 0 ],
    #   J(y,t) = [ 0 I ; -I 0 ].
    # Discretization method: Verlet
    # Parameters:
    #   ham_params is a structure with the hamiltonian parameters (see params.py)
    #   nf is the number of features (nb of channels) of the input
    #   img_size is the size of the side of the image (image assumed square)
    #   device indicated if it will be defined on GPU or CPU

    def __init__(self, ham_params, nf, img_size, device):
        super().__init__()

        #Add parameters to the class        
        self.n_blocks = ham_params.n_blocks
        self.h = ham_params.h
        self.nf = nf
        self.device = device
        if self.nf%2 == 1:
            print("Possible errors due to extended dimension being odd\n")
        
        #Select the correct activaiton function
        if  ham_params.act == "ReLU":
            self.act = nn.ReLU().to(device)
        elif ham_params.act == "Tanh":
            self.act = nn.Tanh().to(device)
        else: 
            sys.exit("Unknown Activation Function (hamiltonians.py)\n")
        
        #Intialise the layers
        if ham_params.init == "Xavier":
            #Here it is not exactly a Xavier but it is similar and comes from the git of the same authors
            # than [2]
            std = np.sqrt(2/(self.nf*ham_params.ks*ham_params.ks)) 
            K1 = torch.normal(mean=0, std=std ,size=(self.nf // 2, self.nf // 2,
                              ham_params.ks, ham_params.ks, self.n_blocks))
            K2 = torch.normal(mean=0, std=std ,size=(self.nf // 2, self.nf // 2, 
                              ham_params.ks, ham_params.ks, self.n_blocks))
            b1 = torch.normal(mean=0, std=std ,size=(self.nf // 2, 1, self.n_blocks))
            b2 = torch.normal(mean=0, std=std ,size=(self.nf // 2, 1, self.n_blocks))
        elif ham_params.init == "Ones":
            K1 = torch.ones(self.nf // 2, self.nf // 2, ham_params.ks, ham_params.ks, self.n_blocks)
            K2 = torch.ones(self.nf // 2, self.nf // 2, ham_params.ks, ham_params.ks, self.n_blocks)
            b1 = torch.ones(self.nf // 2, 1, self.n_blocks)
            b2 = torch.ones(self.nf // 2, 1, self.n_blocks)
        elif ham_params.init == "Zeros":
            K1 = torch.zeros(self.nf // 2, self.nf // 2, ham_params.ks, ham_params.ks, self.n_blocks)
            K2 = torch.zeros(self.nf // 2, self.nf // 2, ham_params.ks, ham_params.ks, self.n_blocks)
            b1 = torch.zeros(self.nf // 2, 1, self.n_blocks)
            b2 = torch.zeros(self.nf // 2, 1, self.n_blocks)
        else:
            sys.exit("Unknown Hamitonian initialisation (hamiltonians.py)\n")
        
        #Define them as parameters
        self.K1 = nn.Parameter(K1, True)
        self.K2 = nn.Parameter(K2, True)
        self.b1 = nn.Parameter(b1, True)
        self.b2 = nn.Parameter(b2, True)

    def getK(self):
        K = torch.cat((torch.cat((torch.zeros(self.K1.size()).to(self.device), self.K1), 1),
                      torch.cat((self.K2, torch.zeros(self.K1.size()).to(self.device)), 1)), 
                      0).to(self.device)
        return K

    def getb(self):
        b = torch.cat((self.b1, self.b2), 0).to(self.device)
        return b

    def forward(self, Y0):
        # Separate the channels
        Y = Y0[:, :self.nf//2, :, :]
        Z = Y0[:, self.nf//2:, :, :]
        
        # Apply the Hamiltonians functions to the two groups at each block
        for j in range(self.n_blocks):    
            Y = Y + self.h * F.conv2d(
                self.act(F.conv2d(Z, self.K1[:, :, :, :, j], self.b1[:, 0, j],padding=1)),
                self.K1[:, :, :, :, j].transpose(0, 1), padding=1)
            Z = Z - self.h * F.conv2d(
                self.act(F.conv2d(Y, self.K2[:, :, :, :, j], self.b2[:, 0, j],padding=1)),
                self.K2[:, :, :, :, j].transpose(0, 1),padding=1)   
        #Regroup them
        YF = torch.cat((Y, Z), 1)
        
        return YF

class eHamiltonian(nn.Module):
    # eHamiltonians J1 and J2 based on the Galimberti paper [3]
    # General ODE: \dot{y} = J(y,t) K^T(t) \tanh( K(t) y(t) + b(t) )
    # Constraints:
    #   J(y,t) = J_1 = [ 0 I ; -I 0 ]  or  J(y,t) = J_2 = [ 0 1 .. 1 ; -1 0 .. 1 ; .. ; -1 -1 .. 0 ].
    # Discretization method: Forward Euler
    # Parameters:
    #   ham_params is a structure with the hamiltonian parameters (see params.py)
    #   nf is the number of features (nb of channels) of the input
    #   img_size is the size of the side of the image (image assumed square)
    #   device indicated if it will be defined on GPU or CPU
    def __init__(self, ham_params, nf, img_size, device):
        super().__init__()

        #Add parameters to the class        
        self.n_blocks = ham_params.n_blocks
        self.h = ham_params.h
        self.nf = nf
        self.device = device
        if self.nf%2 == 1:
            print("Possible errors due to extended dimension being odd\n")
            
        #Select the correct activaiton function
        if  ham_params.act == "ReLU":
            self.act = nn.ReLU().to(device)
        elif ham_params.act == "Tanh":
            self.act = nn.Tanh().to(device)
        else: 
            sys.exit("Unknown Activation Function (hamiltonians.py)\n")
        
        #Select J shape
        if ham_params.hamiltonianFunction == 'J1':
            j_identity = torch.eye(self.nf//2)
            j_zeros = torch.zeros(self.nf//2, self.nf//2)
            self.J = torch.cat((torch.cat((j_zeros, j_identity), 0), 
                                torch.cat((- j_identity, j_zeros), 0)), 1)
        elif ham_params.hamiltonianFunction == 'J2':
            j_aux = np.hstack((np.zeros(1), np.ones(self.nf-1)))
            J = j_aux
            for j in range(self.nf-1):
                j_aux = np.hstack((-1 * np.ones(1), j_aux[:-1]))
                J = np.vstack((J, j_aux))
            self.J = torch.tensor(J, dtype=torch.float32)
        else:
            sys.exit("Unknown eHamiltonian function (hamiltonians.py)\n")

        #Initialise the layers
        if ham_params.init == "Xavier":
            std = np.sqrt(2/(ham_params.nf*ham_params.ks*ham_params.ks)) 
            K = torch.normal(mean=0, std=std ,size=(self.nf, self.nf, 
                             ham_params.ks, ham_params.ks, self.n_blocks))
            b = torch.randn(self.nf, self.n_blocks)
        elif ham_params.init == "Ones":
            K = torch.ones(self.nf, self.nf, ham_params.ks, ham_params.ks, self.n_blocks)
            b = torch.ones(self.nf, self.n_blocks)
        elif ham_params.init == "Zeros":
            K = torch.zeros(self.nf, self.nf, ham_params.ks, ham_params.ks, self.n_blocks)
            b = torch.zeros(self.nf, self.n_blocks)
        else:
            sys.exit("Unknown Hamitonian initialisation (hamiltonians.py)\n")
        
        #Define them as parameters 
        self.K = nn.Parameter(K, True)
        self.b = nn.Parameter(b, True)

    def getK(self):
        return self.K

    def getb(self):
        return self.b

    def forward(self, Y0):

        Y = Y0
        
        #Iterate through the blocks
        for j in range(self.n_blocks):
            fY = self.act(F.conv2d(Y, self.K[:, :, :, :, j], self.b[:, j]))
            JKt = F.linear(self.K[:, :, :, :, j].transpose(1, 3), self.J.to(self.device)).transpose(1, 3)
            Y = Y + self.h * F.conv_transpose2d(fY, JKt)
            
        NNoutput = Y

        return NNoutput