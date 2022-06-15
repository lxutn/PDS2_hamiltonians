# -*- coding: utf-8 -*-
"""
Define a structure for the parameters needed to construct a Hamiltonian block, the parameters needed
to construct a full network as presented in [2] and the parameters tht define the learning problem
See the report for more information

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
# Hamiltonian structure


@dataclass
class HamiltonianParameters:
    # selects which hamiltonian fct will be used,
    hamiltonianFunction: str = "TwoLayersHam"
    # choose btw "TwoLayersHam" ([2]), "J1" and "J2" ([3])
    n_blocks: int = 6  # how many Hamiltonians side by side in each unit
    ks: int = 3  # kernel size of the Hamiltonians
    h: float = 0.1  # discretisation time step
    act: str = "ReLU"  # activation fct to use, choose btw "ReLU" and "Tanh"
    init: str = "Xavier"  # initialisation to use
    # choose btw "Xavier", "Zeros", "Normal" and "Ones"

# Network structure


@dataclass
class NetowrkParameters:
    # Global structure
    hamParams: HamiltonianParameters  # the Hamiltonian to use and its parameters
    nb_units: int = 3  # how many units to put in the network
    nb_features: list = None  # list of the nb of channels/features along the network
    # first the input, then the input of each unit

    # First convolutional layer parameters
    ks_conv: int = 3  # kernel size of the input conv layer
    strd_conv: int = 1  # stride of the input conv layer
    pd_conv: int = 1  # padding of the input conv layer

    # Pooling layers parameters
    pooling: str = "Avg"  # define the type of pooling to use ("Avg" or "Max")
    ks_pool: int = 2  # kernel size of the pooling layer
    strd_pool: int = 2  # stride of the pooling layer

    # Other addons
    init: str = "Xavier"  # initialisation to use for the FC and conv layers
    # choose btw "Xavier", "Zeros", "Normal" and "Ones"
    # None to keep only one FC output layer, if an int is set the
    second_final_FC: int = None
    # additional FC layer will have this nb of neurons as input
    # add or not a batchnormalisation layer after the conv layer
    batchnorm_bool: bool = True
    # apply padding around the channels (True) or append the
    both_border_pad: bool = True
    # channels to the end of the already existing ones (False)
    dropout: list = None  # None to disable dropout, else list of two values
    # the first is the dropout after the first conv layer
    # the second is the dropout after the last unit

# Learning parameters structure


@dataclass
class LearningParameters:
    # Training duration
    training_steps_max: int = 80000  # nb of training step
    batch_size: int = 100  # size of a batch

    # Learning rate
    lr: float = 0.1  # learning rate
    # list of when a learning decay has to happen (in epoch)
    lr_decay_at: list = None
    lr_decay: float = 0.1  # value of the decay (new lr = lr * lr_decay)

    # Regularisation
    wd: float = 2e-4  # weight decay factor
    alpha: float = 2e-4  # weight smoothness factor
    b_in_reg: bool = True  # add b in the regularisation term
    h_div: bool = False  # divide by b in the reg instead of multipling by b

    # Optimiser
    optimiser: str = "SGD"  # select the correct optimiser "SGD" or "Adam"
    momentum: float = 0.9  # if SGD: one value -> SGD momentum
    # if Adam: give a tuple of two values (beta1, beta2)

    # Data modification
    crop_and_flip: bool = True  # activate or not crop and flip for the training data


# J1-ReLU
J1ReLU_ham_params = HamiltonianParameters(hamiltonianFunction="J1",
                                          n_blocks=6,
                                          ks=3,
                                          h=0.1,
                                          act="ReLU",
                                          init="NegativeDominant")

J1ReLU_net_params = NetowrkParameters(hamParams=J1ReLU_ham_params,
                                      nb_units=3,
                                      nb_features=list(
                                          map(int, [3, 32, 64, 112])),
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
                                      dropout=[0.1, 0.2])

J1ReLU_learn_params = LearningParameters(training_steps_max=20000,
                                         batch_size=100,
                                         lr=0.01,
                                         lr_decay_at=[10, 15],
                                         lr_decay=0.1,
                                         wd=2e-4,
                                         alpha=2e-4,
                                         b_in_reg=True,
                                         h_div=False,
                                         momentum=0.9,
                                         crop_and_flip=True,
                                         optimiser="SGD")
