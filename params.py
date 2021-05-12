# -*- coding: utf-8 -*-
"""
Define a structure for the parameters needed to construct a Hamiltoinan block, the parameters needed
to construct a full network as presented in [2] and the parameters tht define the learning problem

Based on different papers:
    [1] Stable architectures for deep neural networks. 
        Eldab Haber and Lars Ruthotto, 2017
    [2] Reversible Architectures for Arbitrarily Deep Residual Neural Networks. 
        Bo Chang, Lili Meng et al., 2018
    [3] A unified framework for Hamiltonian deep neural networks. 
        Clara Galimberti, Liang Xu and Giancarlo Ferrari Trecate, 2021
"""

# ----- Libraries ----- #
from dataclasses import dataclass

# ----- Parameters structure definition ----- #
#Hamiltonian structure
@dataclass
class HamiltonianParameters:
    hamiltonianFunction: str                    #selects which hamiltonian fct will be used, 
                                                #choose btw "TwoLayersHam" ([2]), "J1" and "J2" ([3])
    n_blocks: int                               #how many times the hamiltonian will be applied to the input
    ks: int                                     #kernel size of the hamiltonian
    h: float                                    #discretisation time step
    act: str                                    #activation fct to use, choose btw "ReLU" and "Tanh"
    init: str                                   #initialisation to use
                                                #choose btw "Xavier", "Zeros" and "Ones"     

#Network structure
@dataclass
class NetowrkParameters:
    #Global structure
    hamParams: HamiltonianParameters            #the Hamiltonian to use and it parameters 
    nb_units: int                               #how many units to put in the network
    nb_features: int                            #list of the nb of channels/features along the network
                                                #starting from the input to the output
    #First convolutional layer parameters
    ks_conv: int                                #kernel size of the input conv layer
    strd_conv: int                              #stride of the input conv layer
    pd_conv: int                                #padding of the input conv layer
    #Pooling layers parameters
    pooling: str                                #define the type of pooling to use ("Avg" or "Max")
    ks_pool: int                                #kernel size of the pooling layer
    strd_pool: int                              #stride of the pooling layer
    #Other addons
    init: str                                   #choose btw "Xavier", "Zeros" and "Ones" 
    second_final_FC: int                        #None to keep only one FC output layer, put an int value to
                                                #have a second one which will have the int val nb of neurons
    batchnorm_bool: bool                        #add or not a batchnorm layer after the conv layer
    both_border_pad: bool                       #apply padding only at the end (False) or divide the pad
                                                #by putting half channels at the start and half at the end
    dropout: list                               #list of 2 values, first one will define the dropout proba
                                                #after the conv layer, second the dropout proba befor the FC
                                                #layer, put a None value to deactivate it
                                                
#Learning parameters structure
@dataclass
class LearningParameters:
    #Training duration
    training_steps_max: int                     #nb of training step
    batch_size: int                             #size of a batch
    #Learning rate
    lr: float                                   #learning rate
    lr_decay_at: list                           #list of when a learning decay has to happen (in epoch)
    lr_decay: float                             #value of the deca
    #Regularisation
    wd: float                                   #weight decay in the optimiser
    alpha: float                                #regression weight of the norm of K
    b_in_reg: bool                              #add b in the regression term
    h_div: bool                                 #divide by b in the regression instead of multipling by b
    #Optimiser
    momentum: float                             #momentul for SGD
                                                
# ----- Already built structures ----- #
#Classic paper implementation
paper_ham_params = HamiltonianParameters(hamiltonianFunction="TwoLayersHam",
                                   n_blocks=6,
                                   ks=3,
                                   h=0.05,
                                   act="ReLU",
                                   init="Xavier")

paper_net_params = NetowrkParameters(hamParams=paper_ham_params,
                                   nb_units=3,
                                   nb_features=list(map(int,[3, 32, 64, 112])),
                                   ks_conv=3,
                                   strd_conv=1,
                                   pd_conv=1,
                                   pooling="Avg",
                                   ks_pool=2,
                                   strd_pool=2,
                                   init="Xavier",
                                   second_final_FC=None,
                                   batchnorm_bool=False,
                                   both_border_pad=False,
                                   dropout=None)

paper_learn_params = LearningParameters(training_steps_max=80000,
                                    batch_size=100,
                                    lr=0.1,
                                    lr_decay_at=[80,120,160],
                                    lr_decay=0.1,
                                    wd=2e-4,
                                    alpha=2e-4,
                                    b_in_reg=False,
                                    h_div=True,
                                    momentum=0.9)

#Optimised paper implementation
best_paper_ham_params = HamiltonianParameters(hamiltonianFunction="TwoLayersHam",
                                   n_blocks=6,
                                   ks=3,
                                   h=0.2,
                                   act="ReLU",
                                   init="Xavier")

best_paper_net_params = NetowrkParameters(hamParams=paper_ham_params,
                                   nb_units=3,
                                   nb_features=list(map(int,[3, 32, 64, 112])),
                                   ks_conv=3,
                                   strd_conv=1,
                                   pd_conv=1,
                                   pooling="Avg",
                                   ks_pool=2,
                                   strd_pool=2,
                                   init="Xavier",
                                   second_final_FC=None,
                                   batchnorm_bool=True,
                                   both_border_pad=True,
                                   dropout=[0.1,0.2])

best_paper_learn_params = LearningParameters(training_steps_max=80000,
                                    batch_size=100,
                                    lr=0.1,
                                    lr_decay_at=[80,120,150],
                                    lr_decay=0.1,
                                    wd=2e-4,
                                    alpha=2e-4,
                                    b_in_reg=True,
                                    h_div=False,
                                    momentum=0.9)