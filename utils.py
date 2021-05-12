# -*- coding: utf-8 -*-
"""
Utils functions for the NN
"""

# ----- Libraries ----- #
import torch
import sys

# ----- Utils functions ----- #
def nb_param_model(model):
    #Return the nb of trainable parameters of a model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initWeights(layer, init):
    #init the weights of the layer in the desired fashion (depending on the init str)
    if init == "Xavier":
        torch.nn.init.xavier_uniform_(layer.weight)
    elif init == "Zeros":
        layer.weight.data.fill_(0.0)
    elif init == "Ones":
        layer.weight.data.fill_(1.0)
    else:
        sys.exit("This initialisation for the layers in unknown (utils.py)")