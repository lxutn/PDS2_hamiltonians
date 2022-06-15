# -*- coding: utf-8 -*-
"""
Define different Hamiltonians basic blocks proposed in [1], [2] and [3]; and some enhancements.

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
import torch.nn as nn
import torch.nn.functional as F
# Utils
import sys
# Ours
from utils import init_K_and_b

# ----- Hamiltonians definition ----- #


class eHamiltonian(nn.Module):
    # General ODE: y = y + self.h*sigma(conv(y))
    # Parameters:
    #   ham_params      set of information about the Hamiltonian network (see params.py)
    #   nf              number of channels in input
    #   img_size        side length of an image (assumed square)
    #   device          device on which computation have to be handled (GPU/CPU)
    def __init__(self, ham_params, nf, img_size, device):
        super().__init__()

        self.ham_params = ham_params
        self.n_blocks = ham_params.n_blocks
        self.h = ham_params.h
        self.nf = nf
        self.device = device

        if self.nf % 2 == 1:
            sys.exit(
                "Possible errors due to extended dimension being odd (hamitlonians.py).\n")

        # Select the correct activaiton function
        if ham_params.act == "ReLU":
            self.act = nn.LeakyReLU().to(device)
        elif ham_params.act == "Tanh":
            self.act = nn.Tanh().to(device)
        else:
            sys.exit("Unknown Activation Function (hamiltonians.py)\n")

        # Define K and b
        K, b = init_K_and_b(ham_params.init, nf, ham_params.ks, ham_params.n_blocks,
                            ham_params.hamiltonianFunction)
        self.K = nn.Parameter(K, True)
        self.b = nn.Parameter(b, True)

    def getK(self):
        return self.K

    def getb(self):
        return self.b

    def forward(self, Y):
        # Iterate through the blocks
        for j in range(self.n_blocks):
            # fY = self.act(F.conv2d(Y, self.K[:, :, :, :, j], self.b[:, j]))
            # JKt = F.linear(self.K[:, :, :, :, j].transpose(
            #     1, 3), self.J.to(self.device)).transpose(1, 3)
            # Y = Y + self.h * F.conv_transpose2d(fY, JKt)

            fY = self.act(
                F.conv2d(Y, self.K[:, :, :, :, j], self.b[:, j], padding=1))
            Y = Y + self.h * fY

        return Y
