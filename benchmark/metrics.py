import os

import torch
from torch.nn.functional import softmax
import numpy as np
# from .tools import freeze
from scipy.linalg import sqrtm
from .auxiliary import get_data_home

from .gaussian_mixture_benchmark import (
    get_guassian_mixture_benchmark_ground_truth_sampler, 
)

import gc


def symmetrize(X):
    return np.real((X + X.T) / 2)


def compute_BW_with_gt_stats(model_samples, true_samples_mu, true_samples_covariance):
    model_samples_covariance = np.cov(model_samples.T)
    model_samples_mu = model_samples.mean(axis=0)
    model_samples_covariance_sqrt = symmetrize(sqrtm(model_samples_covariance))
    
    true_samples_covariance_sqrt = symmetrize(sqrtm(true_samples_covariance))

    mu_term = 0.5*((true_samples_mu - model_samples_mu)**2).sum()
    covariance_term = (
        0.5*np.trace(model_samples_covariance) + 
        0.5*np.trace(true_samples_covariance) -
        np.trace(symmetrize(sqrtm(model_samples_covariance_sqrt@true_samples_covariance@model_samples_covariance_sqrt)))
    )

    BW = mu_term + covariance_term
        
    return BW


def compute_BW_by_gt_samples(model_samples, true_samples):
    true_samples_covariance = np.cov(true_samples.T)
    true_samples_mu = true_samples.mean(axis=0)
        
    return compute_BW_with_gt_stats(model_samples, true_samples_mu, true_samples_covariance)


def calculate_cond_bw(input_samples, predictions, eps, dim):
    ground_truth_plan_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=dim, eps=eps,
                                                                                batch_size=32, 
                                                                                device="cpu",
                                                                                download=False)
    input_samples = input_samples.detach().cpu()
    predictions = predictions.detach().cpu()
    
    conditional_plan = ground_truth_plan_sampler.conditional_plan
    cov_matrices = torch.stack([distr.covariance_matrix for distr in conditional_plan.gaussians_distributions], dim=0)
    locs_static = torch.stack([distr.loc for distr in conditional_plan.gaussians_distributions], dim=0)
    
    cond_bws = []
    
    for input_sample, predict in zip(input_samples, predictions):
        X = input_sample[None, :]
        locs_x = torch.stack([(weight@X.T).T[0] for weight in conditional_plan.plan_mu_weights], dim=0)
        locs = locs_static + locs_x

        probs = softmax(conditional_plan.components_distirubtion.calculate_log_probs(X), dim=1)[0]

        mean_covariance = (probs[:, None, None]*cov_matrices).sum(dim=0)
        total_mean = (probs[:, None]*locs).sum(dim=0)
        covariance_of_means = ((locs - total_mean[None, :]).T)@(probs[:, None]*(locs - total_mean[None, :]))

        total_covariance = mean_covariance + covariance_of_means
        
        cond_bws.append(compute_BW_with_gt_stats(predict.numpy(), total_mean.numpy(), total_covariance.numpy()))
        
    cond_bws = np.mean(cond_bws)
    norm = 0.5*np.load(os.path.join(get_data_home(), "y_vars", f"y_var_dim_{dim}_eps_{eps}.npy"))
        
    return 100*cond_bws/norm


def compute_BW_UVP_with_gt_stats(model_samples, true_samples_mu, true_samples_covariance):
    model_samples_covariance = np.cov(model_samples.T)
    model_samples_mu = model_samples.mean(axis=0)
    model_samples_covariance_sqrt = symmetrize(sqrtm(model_samples_covariance))
    
    true_samples_covariance_sqrt = symmetrize(sqrtm(true_samples_covariance))

    mu_term = 0.5*((true_samples_mu - model_samples_mu)**2).sum()
    covariance_term = (
        0.5*np.trace(model_samples_covariance) + 
        0.5*np.trace(true_samples_covariance) -
        np.trace(symmetrize(sqrtm(model_samples_covariance_sqrt@true_samples_covariance@model_samples_covariance_sqrt)))
    )

    BW = mu_term + covariance_term
    BW_UVP = 100*(BW/(0.5*np.trace(true_samples_covariance)))
        
    return BW_UVP


def compute_BW_UVP_by_gt_samples(model_samples, true_samples):
    true_samples_covariance = np.cov(true_samples.T)
    true_samples_mu = true_samples.mean(axis=0)
        
    return compute_BW_UVP_with_gt_stats(model_samples, true_samples_mu, true_samples_covariance)

