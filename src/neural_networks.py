import os, sys
sys.path.append("..")

import torch
from torch import nn


def make_net(n_inputs, n_outputs, n_layers=3, n_hiddens=100):
    layers = [nn.Linear(n_inputs, n_hiddens), nn.ReLU()]
    
    for i in range(n_layers - 1):
        layers.extend([nn.Linear(n_hiddens, n_hiddens), nn.ReLU()])
        
    layers.append(nn.Linear(n_hiddens, n_outputs))
    
    return nn.Sequential(*layers)
