import os, sys
sys.path.append("..")

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import grad as torch_grad
from torchvision.utils import make_grid

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.independent import Independent
from torch.nn.functional import softmax

from .glow_model import Glow
from .glow_train import calc_z_shapes, sample_data
from typing import Callable
import pdb

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


def log_p_x_y(z_vector: torch.Tensor,
              condition_images: torch.Tensor,
              glow_model: torch.nn.Module,
              temperature: float,
              shapes,
              log_f_x_y) -> torch.Tensor:
    """
    Calculate log of the unnormalized density p(x|y)=p(y|x)p(x)/p(y) for a given entropic OT plan p(x, y) = p(y|x)p(x) given by a Glow normalizing model.
    
    Parameters:
            current_state: A batch of current state vectors [B x D].
            log_p: A function to calculate logarithm of unnormalized pdf from which samples are drawn. Maps tensor of size [B x D] to the tensor of size [B].
            delta_time: Time discritization.
    Returns:
            new_state: A batch of state vectors after one step of the algorithm [B x D].
    """
    
    log_p = -1/(temperature**2)*torch.norm(z_vector, dim=-1).square()
    z_list = vector_to_tensor_list(z_vector, shapes)
    images = 2*glow_model.module.reverse(z_list)
    log_p += log_f_x_y(images, condition_images)
    
    return log_p


def log_f_x_y_image(x, y, potential_probs, potential_mus, potential_sigmas, eps):
    log_probs = []
    
    potential_probs = potential_probs.to(x.device)
    potential_mus = potential_mus.to(x.device)
    potential_sigmas = potential_sigmas.to(x.device)

    plan_sigmas = 1/((1/eps) + 1/potential_sigmas)
    plan_mu_biases = plan_sigmas*(1/potential_sigmas)*potential_mus
    plan_mu_weights = plan_sigmas/eps
    
    cat_distr = Independent(Normal(loc=potential_mus, scale=torch.sqrt(potential_sigmas + eps)), 3)
    current_y_probs = potential_probs*softmax(cat_distr.log_prob(y[0]))
    mix = Categorical(current_y_probs)

    current_y_mu = plan_mu_biases + plan_mu_weights*y
    comp = Independent(Normal(loc=current_y_mu, scale=torch.sqrt(plan_sigmas)), 3)
    gmm = MixtureSameFamily(mix, comp)
    
    for elem in x:
        log_probs.append(gmm.log_prob(elem))

    log_probs = torch.stack(log_probs, dim=0)
    
    return log_probs


def metropolis_adjusted_latent_langevien_step(current_state: torch.Tensor,
                                              log_p: Callable[[torch.Tensor], torch.Tensor],
                                              delta_time: float=1e-2) -> torch.Tensor:
    """
    Make one step of Metropolis-adjusted Langevin algorithm.
    
    Parameters:
            current_state: A batch of current state vectors [B x D].
            log_p: A function to calculate logarithm of unnormalized pdf from which samples are drawn. Maps tensor of size [B x D] to the tensor of size [B].
            delta_time: Time discritization.
    Returns:
            new_state: A batch of state vectors after one step of the algorithm [B x D].
    """
    
    current_state.requires_grad = True
    current_log_p = log_p(current_state)
    current_log_p.backward(torch.ones_like(current_log_p))
    current_grad = current_state.grad
    
    with torch.no_grad():
        noise = torch.randn_like(current_state).to(current_state.device)
        proposal_state = torch.clone(current_state).detach() + current_grad*delta_time + np.sqrt(2*delta_time)*noise
    
    proposal_state.requires_grad = True
    proposal_log_p = log_p(proposal_state)
    proposal_log_p.backward(torch.ones_like(proposal_log_p))
    proposal_grad = proposal_state.grad
    
    with torch.no_grad():        
        current_log_q = -(1/(4*delta_time))*torch.norm(current_state - proposal_state - delta_time*proposal_grad, p=2, dim=-1).square()
        proposal_log_q = -(1/(4*delta_time))*torch.norm(proposal_state - current_state - delta_time*current_grad, p=2, dim=-1).square()
        
        rejection_value = torch.exp(proposal_log_p - current_log_p + current_log_q - proposal_log_q)
        alpha = torch.minimum(torch.ones_like(rejection_value).to(rejection_value.device), rejection_value)
        
        is_accepted = torch.rand(alpha.shape[0]).to(alpha.device) < alpha
        is_accepted = is_accepted[:, None]
        bool_index = torch.cat((is_accepted, ~is_accepted), dim=1)
        
        states = torch.cat((proposal_state[:, None], current_state[:, None]), dim=1)
        new_state = states[bool_index, :, ...]
    
    return new_state, is_accepted


def glow_log_p(inp, model, temperature=0.7):
    z_list = model(inp)[2]
    z_vector = tensor_list_to_vector(z_list)
    return -1/(2*temperature**2)*(z_vector.square().sum(dim=(1,)))
