import os, sys
sys.path.append("..")

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.utils import make_grid

from src.model import Glow
from src.train import calc_z_shapes

import gc


def tensor_list_to_vector(input_list):
    batch_size = input_list[0].shape[0]
    reshaped_input_list = [elem.reshape(batch_size, -1) for elem in input_list]
    return torch.cat(reshaped_input_list, dim=-1)
    

def vector_to_tensor_list(input_vector, shapes):
    batch_size = input_vector.shape[0]
    result_list = []
    
    start_ind = 0
    end_ind = 0
    for shape in shapes:
        channels, x_size, y_size = shape
        end_ind += channels*x_size*y_size
        
        result_list.append(input_vector[:, start_ind:end_ind].reshape(batch_size, channels, x_size, y_size))
        
        start_ind = end_ind
        
    return result_list


def sample_z_list(batch_size, z_shapes, device, temp=0.7):
    z_sample = []
    for z in z_shapes:
        z_new = torch.randn(batch_size, *z) * temp
        z_sample.append(z_new.to(device))
    
    return z_sample


@torch.no_grad()
def z_list_to_image(model, z_list):
    return 2*model.module.reverse(z_list)


def latent_langevien_step(z_vector, condition_images, glow_model, 
                          log_f_x_y, shapes, temperatue=0.7, delta_time=1e-2):
    unconditional_grad = -1/(temperatue**2)*z_vector
    
    z_vector.requires_grad = True
    z_list = vector_to_tensor_list(z_vector, shapes)
    images = 2*glow_model.module.reverse(z_list)
    log_f = log_f_x_y(images, condition_images)
    log_f.backward(torch.ones_like(log_f))
    conditional_grad = z_vector.grad
    
    with torch.no_grad():
        grad = conditional_grad + unconditional_grad
        noise = torch.randn_like(z_vector)
        new_z_vector = torch.clone(z_vector).detach() + grad*delta_time + np.sqrt(2*delta_time)*noise
        
    return new_z_vector
