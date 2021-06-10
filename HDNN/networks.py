# -*- coding: utf-8 -*-
"""
Define the structure for Hamiltonian similar to what is proposed in [2]; with some enhancements

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
import torch.nn as nn
import torch.nn.functional as F
#Utils
import sys
#Ours
from hamiltonians import TwoLayerHamiltonian, eHamiltonian
from utils import initWeights

# ----- Network units declaration ----- #
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
        
        self.ham_params = net_params.hamParams 
        self.n_f_out = n_f_out
        self.n_f_in = n_f_in
        self.both_border_pad = net_params.both_border_pad
        ham_params = net_params.hamParams
        
        #Hamiltonian
        if ham_params.hamiltonianFunction == "J2" or ham_params.hamiltonianFunction == "J1":
            self.baseUnit = eHamiltonian(ham_params, n_f_in, img_size, device)
        elif ham_params.hamiltonianFunction == "TwoLayersHam":
            self.baseUnit = TwoLayerHamiltonian(ham_params, n_f_in, img_size, device)
        else:
            sys.exit("This hamiltonian is not available (networks.py)")
            
        #Pooling
        if net_params.pooling == "Avg":
            self.pool = nn.AvgPool2d(kernel_size=net_params.ks_pool, stride=net_params.strd_pool)
        elif net_params.pooling == "Max":
            self.pool = nn.MaxPool2d(kernel_size=net_params.ks_pool, stride=net_params.strd_pool)
        else:
            sys.exit("This pooling function is not available (networks.py)")

    def forward(self, x):
        #Apply Hamiltonian
        x = self.baseUnit(x)
        
        #Apply Pooling
        x = self.pool(x)
        
        #If enabled, apply padding on the channels
        if self.n_f_out is not None:
            if self.both_border_pad: #pad around the already existing channels ("sandwich")
                pad_val = int((self.n_f_out-self.n_f_in)/2)
                x = F.pad(x, (0,0,0,0,pad_val,pad_val))
            else: #append the padding channels to the already existing channels
                pad_val = int((self.n_f_out-self.n_f_in))
                x = F.pad(x, (0,0,0,0,0,pad_val))
        
        return x

class Network(nn.Module):
    # Full network as presented in [2] (and some extensions) 
    # Conv -> (BatchNorm) -> (Dropout) -> n Units -> (dropout) -> (FC) -> FC
    # Parameters
    #   net_params      set of information about the network strucutre (see params.py) 
    #   img_size        side length of an image (assumed square)     
    #   device          device on which computation have to be handled (GPU/CPU)            
    
    def __init__(self, net_params, n_labels, img_size, device):
        super(Network, self).__init__()
        
        self.nfs = net_params.nb_features
        self.final_img_size = int(img_size/(2**net_params.nb_units))
        self.second_final_FC = net_params.second_final_FC
        self.batchnorm_bool = net_params.batchnorm_bool
        self.dropout = net_params.dropout
        
        #Input convolution
        self.conv_init = nn.Conv2d(in_channels=self.nfs[0], out_channels=self.nfs[1], 
                                   kernel_size=net_params.ks_conv, stride=net_params.strd_conv, 
                                   padding=net_params.pd_conv)
        initWeights(self.conv_init, net_params.init)

        #Batchnorm and dropout
        if self.batchnorm_bool:
            self.batch = nn.BatchNorm2d(self.nfs[1])
        if self.dropout != None:
            self.init_dropout = nn.Dropout2d(p=self.dropout[0])
        
        #Units
        self.units = nn.ModuleList()
        for i in range(net_params.nb_units-1):
            self.units.append(Unit(net_params, img_size, device, self.nfs[i+1], self.nfs[i+2]))
        #last layer doesn't have zero padding -> None for n_f_out (-> n_f_in=n_f_out)
        self.units.append(Unit(net_params, img_size, device, self.nfs[-1], None))
        
        #Second dropout
        if self.dropout != None:
            self.end_dropout = nn.Dropout(self.dropout[1])
        
        #FC layer(s)
        if self.second_final_FC==None:
            self.fc = nn.Linear(self.nfs[-1]*(self.final_img_size)**2, n_labels)
            initWeights(self.fc, net_params.init)
        else:
            self.fc1 = nn.Linear(self.nfs[-1]*(self.final_img_size)**2, self.second_final_FC)
            initWeights(self.fc1, net_params.init)
            self.fc2 = nn.Linear(self.second_final_FC, n_labels)
            initWeights(self.fc2, net_params.init)

    def forward(self, x):
        x = self.conv_init(x)
        
        if self.batchnorm_bool:
            x = self.batch(x)
            
        if self.dropout != None:
            x = self.init_dropout(x)
        
        #All the Hamiltonian units
        for l in self.units:
            x = l(x)
        
        x = x.reshape(-1, self.nfs[-1]*((self.final_img_size)**2))
        
        if self.dropout != None:
            x = self.end_dropout(x)
            
        if self.second_final_FC==None:
            x = self.fc(x)
        else:
            x = self.fc1(x)
            x = self.fc2(x)

        return x
