# -*- coding: utf-8 -*-
"""
Define the structure for Hamiltonian similar to what is proposed in [2]

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
    # Basic unit as presented in [2]: ham function -> pooling (-> padding)
    # Parameters
    #   net_params are the parameters of the network (see params.py)
    #   img_size is the size of the received image/data
    #   device indicates to which device the data and operation have to be done
    #   n_f_in are the nb of features (channels) that come to the unit
    #   n_f_out are the nb of features (channels) that have to go out of the unit
    #           note that if n_f_out is None no padding is applied
    def __init__(self, net_params, img_size, device, n_f_in, n_f_out):
        super().__init__()
        
        #Add to the class
        self.n_f_out = n_f_out
        self.n_f_in = n_f_in
        self.ham_params = net_params.hamParams
        self.n_blocks = self.ham_params.n_blocks
        self.both_border_pad = net_params.both_border_pad
        self.pooling = net_params.pooling
        
        #Add layers
        #Hamiltonian
        if self.ham_params.hamiltonianFunction == "eHam_J2":
            self.baseUnit = eHamiltonian(self.ham_params, n_f_in, img_size, device)
        elif self.ham_params.hamiltonianFunction == "eHam_J1":
            self.baseUnit = eHamiltonian(self.ham_params, n_f_in, img_size, device)
        elif self.ham_params.hamiltonianFunction == "TwoLayersHam":
            self.baseUnit = TwoLayerHamiltonian(self.ham_params, n_f_in, img_size, device)
        else:
            sys.exit("This hamiltonian is not available (networks.py)")
        #Pooling
        if self.pooling == "Avg":
            self.pool = nn.AvgPool2d(kernel_size=net_params.ks_pool, stride=net_params.strd_pool)
        elif self.pooling == "Max":
            self.pool = nn.MaxPool2d(kernel_size=net_params.ks_pool, stride=net_params.strd_pool)
        else:
            sys.exit("This pooling function is not available (networks.py)")

    def forward(self, x):
        x = self.baseUnit(x)
        x = self.pool(x)
        
        if self.n_f_out is not None: #zero padding (of the channels)
            if self.both_border_pad:
                pad_val = int((self.n_f_out-self.n_f_in)/2)
                x = F.pad(x, (0,0,0,0,pad_val,pad_val))
            else:
                pad_val = int((self.n_f_out-self.n_f_in))
                x = F.pad(x, (0,0,0,0,0,pad_val))
        
        return x

class Network(nn.Module):
    # Full network as presented in [2]: Conv -> n Units -> FC
    # Parameters
    #   net_params are the parameters of the network (see params.py)
    #   img_size is the size of the received image/data (side of the image, assumed square)
    #   device indicates to which device the data and operation have to be done
    # Warning : Assumes that the convolutional layer doesn't reduce dimension and that
    #           the average pooling divide by two the size each time               
    
    def __init__(self, net_params, n_labels, img_size, device):
        super(Network, self).__init__()
        
        #Add to the class
        self.nfs = net_params.nb_features
        self.nb_units = net_params.nb_units
        self.final_img_size = int(img_size/(2**self.nb_units))
        self.second_final_FC = net_params.second_final_FC
        self.batchnorm_bool = net_params.batchnorm_bool
        self.dropout = net_params.dropout
        
        #Define the layers
        self.conv_init = nn.Conv2d(in_channels=self.nfs[0], out_channels=self.nfs[1], 
                                   kernel_size=net_params.ks_conv, stride=net_params.strd_conv, 
                                   padding=net_params.pd_conv)
        initWeights(self.conv_init, net_params.init)

        if self.batchnorm_bool:
            self.batch = nn.BatchNorm2d(self.nfs[1])
        if self.dropout != None:
            self.init_dropout = nn.Dropout2d(p=self.dropout[0])
        
        self.units = nn.ModuleList()
        for i in range(self.nb_units-1):
            self.units.append(Unit(net_params, img_size, device, self.nfs[i+1], self.nfs[i+2]))
        #last layer doesn't have zero padding -> out nb of channels is None (will be the same as input)
        self.units.append(Unit(net_params, img_size, device, self.nfs[-1], None))
        
        if self.dropout != None:
            self.end_dropout = nn.Dropout(self.dropout[1])
        
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
