# -*- coding: utf-8 -*-
"""
Define the network structure: input->conv->batch_norm->NODE_block->\ldots->NODE_block->FC
the Unit is given by: NODE_iterations->pooling->pading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Utils
import sys


class Network(nn.Module):
    # Conv -> (BatchNorm) -> (Dropout) -> n Units -> (dropout) -> (FC) -> FC
    # Parameters
    #   net_params      set of information about the network strucutre (see params.py)
    #   img_size        side length of an image (assumed square)
    #   device          device on which computation have to be handled (GPU/CPU)

    def __init__(self, net_params, n_labels, img_size, device):
        super(Network, self).__init__()

        self.nfs = net_params.nb_features
        self.final_img_size = int(img_size/(2**net_params.nb_units))
        self.batchnorm_bool = net_params.batchnorm_bool
        self.dropout = net_params.dropout

        # Input convolution
        self.conv_init = nn.Conv2d(in_channels=self.nfs[0], out_channels=self.nfs[1],
                                   kernel_size=net_params.ks_conv, stride=net_params.strd_conv,
                                   padding=net_params.pd_conv)

        #Batchnorm and dropout
        if self.batchnorm_bool:
            self.batch = nn.BatchNorm2d(self.nfs[1])
        if self.dropout != None:
            self.init_dropout = nn.Dropout2d(p=self.dropout[0])

        # Units
        self.units = nn.ModuleList()
        for i in range(net_params.nb_units-1):
            self.units.append(
                Unit(net_params, img_size, device, self.nfs[i+1], self.nfs[i+2]))
        # last layer doesn't have zero padding -> None for n_f_out (-> n_f_in=n_f_out)
        self.units.append(
            Unit(net_params, img_size, device, self.nfs[-1], None))

        # Second dropout
        if self.dropout != None:
            self.end_dropout = nn.Dropout(self.dropout[1])

        # FC layer(s)
        self.fc = nn.Linear(
            self.nfs[-1]*(self.final_img_size)**2, n_labels)

    def forward(self, x):
        x = self.conv_init(x)

        if self.batchnorm_bool:
            x = self.batch(x)

        if self.dropout != None:
            x = self.init_dropout(x)

        # All the Hamiltonian units
        for l in self.units:
            x = l(x)

        x = x.reshape(-1, self.nfs[-1]*((self.final_img_size)**2))

        if self.dropout != None:
            x = self.end_dropout(x)

        x = self.fc(x)

        return x


class Unit(nn.Module):
    # Basic unit as presented in [2]: Hamiltonian blocks -> pooling (-> padding)
    # Parameters
    #   net_params      set of information about the network strucutre (see params.py)
    #   img_size        side length of an image (assumed square)
    #   device          device on which computation have to be handled (GPU/CPU)
    #   n_f_in          number of channles in input of the unit
    #   n_f_out         number of channles in output of the unit (if None->keep the same number of channels)

    def __init__(self, net_params, img_size, device, n_f_in, n_f_out):
        super().__init__()

        self.net_params = net_params
        self.node_params = net_params.NODEParams
        self.both_border_pad = net_params.both_border_pad
        self.n_f_in = n_f_in
        self.n_f_out = n_f_out

        # NODE
        if self.node_params.act == "ReLU":
            self.act = nn.LeakyReLU().to(device)
        elif self.node_params.act == "Tanh":
            self.act = nn.Tanh().to(device)
        else:
            sys.exit("Unknown Activation Function\n")

        # Pooling
        if net_params.pooling == "Avg":
            self.pool = nn.AvgPool2d(
                kernel_size=net_params.ks_pool, stride=net_params.strd_pool)
        elif net_params.pooling == "Max":
            self.pool = nn.MaxPool2d(
                kernel_size=net_params.ks_pool, stride=net_params.strd_pool)
        else:
            sys.exit("This pooling function is not available (networks.py)")

        # Define conv list for NODE iterations
        self.convList = nn.ModuleList()
        for i in range(self.node_params.n_blocks):
            self.convList.append(
                nn.Conv2d(n_f_in, n_f_in, self.node_params.ks, padding=1))

        my_init_weight = -torch.tensor([[0, 0, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]])
        with torch.no_grad():
            for i in range(self.node_params.n_blocks):
                self.convList[i].weight.copy_(my_init_weight)

    def NODE_iterations(self, Y):
        for j in range(self.node_params.n_blocks):
            Y = Y+self.node_params.h*self.act(self.convList[j](Y))
        return Y

    def forward(self, x):
        # Apply Hamiltonian
        x = self.NODE_iterations(x)

        # Apply Pooling
        x = self.pool(x)

        # If enabled, apply padding on the channels
        if self.n_f_out is not None:
            # pad around the already existing channels ("sandwich")
            if self.both_border_pad:
                pad_val = int((self.n_f_out-self.n_f_in)/2)
                x = F.pad(x, (0, 0, 0, 0, pad_val, pad_val))
            else:  # append the padding channels to the already existing channels
                pad_val = int((self.n_f_out-self.n_f_in))
                x = F.pad(x, (0, 0, 0, 0, 0, pad_val))

        return x
