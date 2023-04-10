import os

import torch
import numpy as np
# from .tools import freeze
from scipy.linalg import sqrtm
from .auxiliary import get_data_home

import gc


def symmetrize(X):
    return np.real((X + X.T) / 2)


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
                                  "identity_distance_kernel"]
    assert dim in [2, 4, 8, 16, 32, 64, 128]
    assert eps in [0.1, 1, 10]
    
    kernel_width = dim if "norm" in normalization_type else 1
    kernel_type = "distance" if "distance_kernel" in normalization_type else "rbf"
#     path = f"{normalization_type}_scale_dim_{dim}_eps_{eps}.torch"
    path = os.path.join(get_data_home(),
                    "gaussian_mixture_benchmark_data",
                    f"{normalization_type}_scale_dim_{dim}_eps_{eps}.torch")
    
    scale_factor = torch.load(path).item()
    return (calculate_mmd(x, y, batch_size=batch_size, kernel_type=kernel_type, kernel_width=kernel_width)/scale_factor)*100
    
#     if normalization_type == "indep_plan_rbf_kernel":
#         kernel_type = "rbf"
#         kernel_width = 1
#         path = os.path.join(get_data_home(),
#                             "gaussian_mixture_benchmark_data",
#                             f"scale_dim_{dim}_eps_{eps}.torch")
        
        
#     elif normalization_type == "indep_plan_rbf_kernel_dim_norm":
#         kernel_type = "rbf
#         kernel_width = dim
#         path = os.path.join(get_data_home(),
#                             "gaussian_mixture_benchmark_data",
#                             f"scale_dim_{dim}_eps_{eps}_dim_normalized.torch")
        
#     elif normalization_type == "indep_plan_rbf_distance_kernel":
#         kernel_type = "distance"
#         path = os.path.join(get_data_home(),
#                             "gaussian_mixture_benchmark_data",
#                             f"scale_dim_{dim}_eps_{eps}_distance_kernel.torch")
#         scale_factor = torch.load(path).item()
        
#     elif normalization_type == "identity_rbf_kernel":
#         kernel_type = "rbf"
#         kernel_width = 1
#         path = os.path.join(get_data_home(),
#                             "gaussian_mixture_benchmark_data",
#                             f"scale_dim_{dim}_eps_{eps}.torch")
    
#     elif normalization_type == "indentity_rbf_kernel_norm":
#         kernel_type = "rbf
#         kernel_width = dim
#         path = os.path.join(get_data_home(),
#                             "gaussian_mixture_benchmark_data",
#                             f"identity_scale_dim_{dim}_eps_{eps}_dim_normalized.torch")
        
#     elif normalization_type == "identity_distance_kernel":
#         kernel_type = "distance"
#         path = os.path.join(get_data_home(),
#                             "gaussian_mixture_benchmark_data",
#                             f"identity_scale_dim_{dim}_eps_{eps}_distance_kernel.torch")
#         scale_factor = torch.load(path).item()
#     kernel_width=1
#     path = os.path.join(get_data_home(),
#                         "gaussian_mixture_benchmark_data",
#                         f"scale_dim_{dim}_eps_{eps}.torch")
#     scale_factor = torch.load(path).item()


# def calculate_gm_mmd(x, y, dim, eps, batch_size=1000):
#     assert dim in [2, 4, 8, 16, 32, 64, 128]
#     assert eps in [0.1, 1, 10]
    
#     kernel_width=1
#     path = os.path.join(get_data_home(),
#                         "gaussian_mixture_benchmark_data",
#                         f"scale_dim_{dim}_eps_{eps}.torch")
#     scale_factor = torch.load(path).item()
    
#     return (calculate_mmd(x, y, batch_size=batch_size, kernel_width=kernel_width)/scale_factor)*100


# def calculate_gm_mmd_dim_normalized(x, y, dim, eps, batch_size=1000):
#     assert dim in [2, 4, 8, 16, 32, 64, 128]
#     assert eps in [0.1, 1, 10]
    
#     kernel_width=dim
#     path = os.path.join(get_data_home(),
#                         "gaussian_mixture_benchmark_data",
#                         f"scale_dim_{dim}_eps_{eps}_dim_normalized.torch")
#     scale_factor = torch.load(path).item()
    
#     return (calculate_mmd(x, y, batch_size=batch_size, kernel_width=kernel_width)/scale_factor)*100


# def calculate_gm_mmd_distance_kernel(x, y, dim, eps, batch_size=1000):
#     assert dim in [2, 4, 8, 16, 32, 64, 128]
#     assert eps in [0.1, 1, 10]
    
#     path = os.path.join(get_data_home(),
#                         "gaussian_mixture_benchmark_data",
#                         f"scale_dim_{dim}_eps_{eps}_distance_kernel.torch")
#     scale_factor = torch.load(path).item()
    
#     return (calculate_mmd(x, y, batch_size=batch_size, kernel_type="distance")/scale_factor)*100



# dims = [2, 4, 8, 16, 32, 64, 128]
# eps_array = [0.1, 1, 10]
# batch_size = 1000
# n_runs = 10
# n_samples = 10000

# independet_results = {}
# independet_results_normilized_by_mmd = {}
# independet_results_with_distance_kernel = {}

# identity_results = {}
# identity_results_normilized_by_mmd = {}
# identity_results_with_distance_kernel = {}

# for dim in dims:
#     independet_results[dim] = {}
#     independet_results_normilized_by_mmd[dim] = {}
#     independet_results_with_distance_kernel[dim] = {}
    
#     identity_results[dim] = {}
#     identity_results_normilized_by_mmd[dim] = {}
#     identity_results_with_distance_kernel[dim] = {}
    
#     for eps in tqdm(eps_array):
#         independet_result = []
#         independet_result_normilized_by_mmd = []
#         independet_result_with_distance_kernel = []
        
#         identity_result = []
#         identity_result_normilized_by_mmd = []
#         identity_result_with_distance_kernel = []
        
#         for run in range(n_runs):
#             input_sampler = get_guassian_mixture_benchmark_sampler("input", dim=dim, eps=eps, batch_size=batch_size)
#             output_sampler = get_guassian_mixture_benchmark_sampler("target", dim=dim, eps=eps, batch_size=batch_size)
#             gt_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=dim, eps=eps, batch_size=batch_size)
            
#             x = input_sampler.sample(n_samples)
#             y = output_sampler.sample(n_samples)

#             independet_samples = torch.cat((x, y), dim=1)
            
#             x_gt, y_gt = gt_sampler.sample(n_samples)
#             plan_gt_samples = torch.cat((x_gt, y_gt), dim=1)
            
#             independet_mmd = calculate_mmd(
#                 independet_samples,
#                 plan_gt_samples,
#                 batch_size=batch_size,
#                 kernel_width=1
#             )
            
#             independet_mmd_normilized_by_mmd = calculate_mmd(
#                 independet_samples,
#                 plan_gt_samples,
#                 batch_size=batch_size,
#                 kernel_width=dim
#             )
            
#             independet_mmd_with_distance_kernel = calculate_mmd(
#                 independet_samples,
#                 plan_gt_samples,
#                 batch_size=batch_size,
#                 kernel_type="distance"
#             )
            
            
#             identity_mmd = calculate_mmd(
#                 x_gt,
#                 y_gt,
#                 batch_size=batch_size,
#                 kernel_width=1
#             )
            
#             identity_mmd_normilized_by_mmd = calculate_mmd(
#                 x_gt,
#                 y_gt,
#                 batch_size=batch_size,
#                 kernel_width=dim
#             )
            
#             identity_mmd_with_distance_kernel = calculate_mmd(
#                 x_gt,
#                 y_gt,
#                 batch_size=batch_size,
#                 kernel_type="distance"
#             )
            
#             independet_result.append(independet_mmd.item())
#             independet_result_normilized_by_mmd.append(independet_mmd_normilized_by_mmd.item())
#             independet_result_with_distance_kernel.append(independet_mmd_with_distance_kernel.item())
            
            
#             identity_result.append(identity_mmd.item())
#             identity_result_normilized_by_mmd.append(identity_mmd_normilized_by_mmd.item())
#             identity_result_with_distance_kernel.append(identity_mmd_with_distance_kernel.item())
            
#         independet_results[dim][eps] = torch.tensor(independet_result)
#         independet_results_normilized_by_mmd[dim][eps] = torch.tensor(independet_result_normilized_by_mmd)
#         independet_results_with_distance_kernel[dim][eps] = torch.tensor(independet_result_with_distance_kernel)
        
        
#         identity_results[dim][eps] = torch.tensor(identity_result)
#         identity_results_normilized_by_mmd[dim][eps] = torch.tensor(identity_result_normilized_by_mmd)
#         identity_results_with_distance_kernel[dim][eps] = torch.tensor(identity_result_with_distance_kernel)
    