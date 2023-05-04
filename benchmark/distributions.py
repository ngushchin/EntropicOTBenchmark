import torch
import numpy as np
from scipy.linalg import sqrtm
import sklearn.datasets
import random

from torch.distributions.multivariate_normal import MultivariateNormal


def symmetrize(X):
    return np.real((X + X.T) / 2)


class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
        self.mean, self.var, self.cov = None, None, None
    
    def sample(self, size=5):
        pass
    
    def _estimate_moments(self, size=2**14, mean=True, var=True, cov=True):
        if (not mean) and (not var) and (not cov):
            return
        
        sample = self.sample(size).cpu().detach().numpy().astype(np.float32)
        if mean:
            self.mean = sample.mean(axis=0)
        if var:
            self.var = sample.var(axis=0).sum()
        if cov:
            self.cov = np.cov(sample.T).astype(np.float32)
            
            
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=4):
        samples = []
        n_samples = 0
        
        while n_samples < size:
            try:
                batch, _ = next(self.it)
                samples.append(batch)
                n_samples += batch.shape[0]
            except StopIteration:
                self.it = iter(self.loader)
            
        samples = torch.cat(samples, dim=0)
            
        return samples[:size].to(self.device)
    
    
class RotatedGaussisnLoaderSamplerWithDensity(LoaderSampler):
    def __init__(self, loader, device='cuda'):
        super(RotatedGaussisnLoaderSamplerWithDensity, self).__init__(loader, device)
        covariance_matrix = self.loader.dataset.sigma
        loc = torch.zeros(covariance_matrix.shape[0])
        
        self.distribution = MultivariateNormal(loc, covariance_matrix)
        
    def log_prob(self, samples):
        return self.distribution.log_prob(samples)

