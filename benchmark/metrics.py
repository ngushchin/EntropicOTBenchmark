import os

import torch
import numpy as np
# from .tools import freeze
from scipy.linalg import sqrtm
from .auxiliary import get_data_home

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


def calculate_rbf_mmd_kernels_for_mmd(x, y, kernel_width=1):
    dxx = torch.cdist(x, x).square()
    dyy = torch.cdist(y, y).square()
    dxy = torch.cdist(x, y).square()

    XX = torch.exp(-0.5*dxx/kernel_width)
    YY = torch.exp(-0.5*dyy/kernel_width)
    XY = torch.exp(-0.5*dxy/kernel_width)
    
    return XX, YY, XY


def calculate_distance_mmd_kernels_for_mmd(x, y):
    x_norm = torch.norm(x, dim=-1)[:, None]
    y_norm = torch.norm(y, dim=-1)[None, :]
    
    dxx = torch.cdist(x, x)
    dyy = torch.cdist(y, y)
    dxy = torch.cdist(x, y)
    
    XX = 0.5*x_norm + 0.5*x_norm - 0.5*dxx
    YY = 0.5*y_norm + 0.5*y_norm - 0.5*dyy
    XY = 0.5*x_norm + 0.5*y_norm - 0.5*dxy
    
    return XX, YY, XY


def calculate_mmd(x, y, batch_size=1000, kernel_type="rbf", kernel_width=1):
    assert kernel_type in ["rbf", "distance"]
    
    same_sampler_pairs = 0
    cross_sampler_pairs = 0
    
    xx_kernel = 0
    yy_kernel = 0
    xy_kernel = 0
    
    x = x[:(x.shape[0]//batch_size)*batch_size]
    y = y[:(y.shape[0]//batch_size)*batch_size]
    
    first_batches = x.reshape(-1, batch_size, x.shape[-1])
    second_batches = y.reshape(-1, batch_size, y.shape[-1])
    
    iterations = len(first_batches)
        
    for i in range(iterations):
        for j in range(iterations):
            first_batch = first_batches[i]
            second_batch = second_batches[j]
            
            if kernel_type == "rbf":
                xx, yy, xy = calculate_rbf_mmd_kernels_for_mmd(first_batch, second_batch, kernel_width)
            else:
                xx, yy, xy = calculate_distance_mmd_kernels_for_mmd(first_batch, second_batch)
                
            xx_kernel = xx_kernel + (xx.sum()) - (xx.diag().sum())
            yy_kernel = yy_kernel + (yy.sum()) - (yy.diag().sum())
            xy_kernel = xy_kernel + (xy.sum()) - (xy.diag().sum())
            
            batch_size = xx.shape[0]
            same_sampler_pairs += batch_size*(batch_size-1)
            cross_sampler_pairs += batch_size*(batch_size-1)
        
    return xx_kernel/same_sampler_pairs + yy_kernel/same_sampler_pairs - 2*xy_kernel/cross_sampler_pairs


def calculate_gm_mmd(x, y, dim, eps, normalization_type, batch_size=1000):
    assert normalization_type in ["indep_plan_rbf_kernel",
                                  "indep_plan_rbf_kernel_dim_norm",
                                  "indep_plan_rbf_distance_kernel",
                                  "identity_rbf_kernel", 
                                  "indentity_rbf_kernel_norm", 
                                  "identity_distance_kernel", 
                                  "identity_plan_rbf_kernel",
                                  "indentity_plan_rbf_kernel_norm",
                                  "identity_plan_distance_kernel",]
    assert dim in [2, 4, 8, 16, 32, 64, 128]
    assert eps in [0.01, 0.1, 1, 10]
    
    kernel_width = dim if "norm" in normalization_type else 1
    kernel_type = "distance" if "distance_kernel" in normalization_type else "rbf"
#     path = f"{normalization_type}_scale_dim_{dim}_eps_{eps}.torch"
    path = os.path.join(get_data_home(),
                    "gaussian_mixture_benchmark_data",
                    f"{normalization_type}_scale_dim_{dim}_eps_{eps}.torch")
    
    scale_factor = torch.load(path).item()
    return (calculate_mmd(x, y, batch_size=batch_size, kernel_type=kernel_type, kernel_width=kernel_width)/scale_factor)*100

