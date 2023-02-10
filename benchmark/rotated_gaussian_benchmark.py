import os, sys
sys.path.append("..")

import gdown
import os
import torch
import numpy as np

from zipfile import ZipFile
from scipy.stats import ortho_group
from torch.utils.data import Dataset, DataLoader

from .rotated_gaussian_analytical_solution import get_D_sigma, get_C_sigma, get_optimal_plan_covariance 
from .distributions import LoaderSampler, RotatedGaussisnLoaderSamplerWithDensity

def initialize_random_rotated_gaussian(eigenvalues, seed=42):
    np.random.seed(seed)

    dim = len(eigenvalues)
    
    rotation = ortho_group.rvs(dim)
    weight = rotation @ np.diag(eigenvalues)
    sigma = weight @ weight.T

    return weight, sigma


class RotatedGaussianDataset(Dataset):
    def __init__(self, weight, dataset_len, device):
        self.weight = weight.float()
        self.sigma = weight@weight.T
        self.dataset_len = dataset_len
        self.device = device
        self.dim = self.sigma.shape[0]
        
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        return torch.randn(self.dim)@self.weight.T, torch.zeros(self.dim)
        
    @classmethod
    def make_new_dataset(cls, dim=2, dataset_len=int(1e18), device="cpu", eigenvalues=None, seed=42):
        if eigenvalues is None:
            eigenvalues = np.exp(np.linspace(np.log(0.5), np.log(2), dim))
            
        assert len(eigenvalues) == dim
        
        weight, sigma = initialize_random_rotated_gaussian(eigenvalues, seed)
        weight = torch.tensor(weight, device=device)
                
        return cls(weight, dataset_len, device)
    
    @classmethod
    def load_from_tensor(cls, tensor_path, dataset_len=int(1e18), device="cpu"):
        weight = torch.load(tensor_path, map_location=device)
        
        return cls(weight, dataset_len, device)

    def save_to_tensor(self, path):
        torch.save(self.weight, path)

    
def download_rotated_gaussian_benchmark_files(path):
    urls = {
        "rotated_gaussians.zip": "https://drive.google.com/uc?id=1ZOUXFdkssPbGJb1jPhVK1dh8lkwu0Sx0",
    }
    for name, url in urls.items():
        gdown.download(url, os.path.join(path, f"{name}"), quiet=False)
        
    with ZipFile(os.path.join(path, "rotated_gaussians.zip"), 'r') as zip_ref:
        zip_ref.extractall("..")
        

def get_rotated_gaussian_dataset(input_or_target, dim, benchmark_data_path, device="cpu", download=False):
    assert input_or_target in ["input", "target"]
    assert dim in [2, 4, 8, 16, 32, 64, 128]
        
    if download:
        download_rotated_gaussian_benchmark_files(benchmark_data_path)
    
    file_name = (f"rotated_gaussian_{dim}_weight_X.torch" 
                 if input_or_target == "input" else f"rotated_gaussian_{dim}_weight_Y.torch"
                )
    
    return RotatedGaussianDataset.load_from_tensor(
        os.path.join(benchmark_data_path, "rotated_gaussians", file_name),
        device=device)


def get_rotated_gaussian_sampler(input_or_target, dim, batch_size, with_density, 
                                 benchmark_data_path, device="cpu", download=False):
    assert input_or_target in ["input", "target"]
    assert dim in [2, 4, 8, 16, 32, 64, 128]
    
    dataset = get_rotated_gaussian_dataset(input_or_target, dim, benchmark_data_path, device, download)
    
    if with_density:
        return RotatedGaussisnLoaderSamplerWithDensity(
            DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size), device
        )
    else:
        return LoaderSampler(DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size), device)


def get_rotated_gaussian_benchmark_stats(dim, eps, benchmark_data_path, device="cpu", download=False):
    assert dim in [2, 4, 8, 16, 32, 64, 128]
        
    if download:
        download_rotated_gaussian_benchmark_files(benchmark_data_path)

    X_dataset = RotatedGaussianDataset.load_from_tensor(
        os.path.join(benchmark_data_path, "rotated_gaussians", f"rotated_gaussian_{dim}_weight_X.torch"),
        device=device
    )
    Y_dataset = RotatedGaussianDataset.load_from_tensor(
        os.path.join(benchmark_data_path, "rotated_gaussians", f"rotated_gaussian_{dim}_weight_Y.torch"),
        device=device
    )
    
    covariance_X = X_dataset.sigma.cpu().numpy()
    covariance_Y = Y_dataset.sigma.cpu().numpy()
        
    mu_X = np.zeros(covariance_X.shape[0])
    mu_Y = np.zeros(covariance_X.shape[0])
    
    optimal_plan_mu = np.zeros(covariance_X.shape[0]*2)
    optimal_plan_covariance = get_optimal_plan_covariance(covariance_X, covariance_Y, eps)
    
    return mu_X, mu_Y, covariance_X, covariance_Y, optimal_plan_mu, optimal_plan_covariance
    

class RotatedGaussiansBenchmark:
    def __init__(self, dim, eps, benchmark_data_path, 
                 make_samplers=False, device="cpu", download=True):
        assert dim in [2, 4, 8, 16, 32, 64, 128]
        
        if download:
            download_rotated_gaussian_benchmark_files(benchmark_data_path)
        
        self.X_dataset = RotatedGaussianDataset.load_from_tensor(
            os.path.join(benchmark_data_path, "rotated_gaussians", f"rotated_gaussian_{dim}_weight_X.torch"),
            device=device
        )
        self.Y_dataset = RotatedGaussianDataset.load_from_tensor(
            os.path.join(benchmark_data_path, "rotated_gaussians", f"rotated_gaussian_{dim}_weight_Y.torch"),
            device=device
        )
        
        if make_samplers:
            self.X_sampler = LoaderSampler(
                DataLoader(self.X_dataset, shuffle=False, num_workers=8, batch_size=batch_size), device
            )
            self.Y_sampler = LoaderSampler(
                DataLoader(self.X_dataset, shuffle=False, num_workers=8, batch_size=batch_size), device
            )
        
        # computing stats for BW-UVP metric calculation
        self.covariance_X = self.X_dataset.sigma.cpu().numpy()
        self.covariance_Y = self.Y_dataset.sigma.cpu().numpy()
        
        self.mu_X = np.zeros(self.covariance_X.shape[0])
        self.mu_Y = np.zeros(self.covariance_X.shape[0])
        
        self.optimal_plan_mu = np.zeros(self.covariance_X.shape[0]*2)
        self.optimal_plan_covariance = get_optimal_plan_covariance(self.covariance_X, self.covariance_Y, eps)

#         mu_t = np.stack(
#             [get_mu_t(t, mu_0, mu_T) for t in np.linspace(0, 1, N_STEPS+1)], axis=0
#         )

#         covariance_t = np.stack(
#             [get_covariance_t(t, covariance_0, covariance_T, C_sigma, EPSILON) for t in np.linspace(0, 1, N_STEPS+1)],
#             axis=0
#         )
        
