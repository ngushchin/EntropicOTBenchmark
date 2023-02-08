import torch
import numpy as np

from scipy.stats import ortho_group
from torch.utils.data import Dataset


def initialize_random_rotated_gaussian(eigenvalues, seed=42):
    np.random.seed(seed)

    dim = len(eigenvalues)
    
    rotation_X = ortho_group.rvs(dim)
    weight_X = rotation_X @ np.diag(eigenvalues)
    sigma_X = weight_X @ weight_X.T

    return weight_X, sigma_X


class RotatedGaussianBenchmarkDataset(Dataset):
    def __init__(self, weight_X, dataset_len, device):
        self.weight_X = weight_X.float()
        self.sigma_X = weight_X@weight_X.T
        self.dataset_len = dataset_len
        self.device = device
        self.dim = self.sigma_X.shape[0]
        
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        return torch.randn(self.dim, self.dim)@self.weight_X.T
        
    @classmethod
    def make_new_dataset(cls, dim=2, dataset_len=int(1e18), device="cpu", eigenvalues=None, seed=42):
        if eigenvalues is None:
            eigenvalues = np.exp(np.linspace(np.log(0.5), np.log(2), dim))
            
        assert len(eigenvalues) == dim
        
        weight_X, sigma_X = initialize_random_rotated_gaussian(eigenvalues, seed)
        weight_X = torch.tensor(weight_X, device=device)
                
        return cls(weight_X, dataset_len, device)
    
    @classmethod
    def load_from_tensor(cls, tensor_path, dataset_len=int(1e18), device="cpu"):
        weight_X = torch.load(tensor_path, map_location=device)
        
        return cls(weight_X, dataset_len, device)

    def save_to_tensor(self, path):
        torch.save(self.weight_X, path)
    
    