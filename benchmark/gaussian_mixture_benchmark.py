import gdown
import os
import torch
import numpy as np

from matplotlib import pyplot as plt

from torch.nn.functional import softmax

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gumbel import Gumbel

from sklearn.decomposition import PCA
from scipy.stats import ortho_group
from scipy.linalg import sqrtm, inv
from zipfile import ZipFile

from .auxiliary import get_data_home


class GaussianMixture:
    def __init__(self, probs: torch.tensor,
                 mus: torch.tensor,
                 sigmas: torch.tensor,
                 device: str):
#         assert torch.allclose(probs.sum(), torch.ones(1))
        self.device = device
    
        probs = probs.to(device)
        mus = mus.to(device)
        sigmas = sigmas.to(device)
    
        self.probs = probs
        self.components_distirubtion = Categorical(probs)
        
        assert len(probs) == len(mus) and len(mus) == len(sigmas)
        self.gaussians_distributions = [
            MultivariateNormal(loc=mu, covariance_matrix=sigma) for mu, sigma in zip(mus, sigmas)
        ]
        self.dim = mus[0].shape[0]
        
    def sample(self, n_samples:int =1) -> torch.tensor:
        components = self.components_distirubtion.sample(sample_shape=torch.Size((n_samples,))).to(self.device)
        
        gaussian_samples = [
            gaussian_distribution.sample(sample_shape=torch.Size((n_samples,))) for gaussian_distribution in self.gaussians_distributions
        ]
        gaussian_samples = torch.stack(gaussian_samples, dim=1)
            
        gaussian_mixture_samples = gaussian_samples.gather(1, components[:, None, None].expand(components.shape[0], 1, self.dim)).squeeze()
        
        return gaussian_mixture_samples
    
    
class ConditionalCategoricalDistribution:
    def __init__(self):
        self.gumbel_distribution = Gumbel(0, 1)
        
    def sample(self, log_probs: torch.tensor) -> torch.tensor:
        gumbel_samples = self.gumbel_distribution.sample(log_probs.shape).to(log_probs.device)
        return torch.argmax(gumbel_samples + log_probs, dim=1)
    
    
class PotentialCategoricalDistribution:
    def __init__(self, potential_probs: torch.tensor,
                 potential_mus: torch.tensor,
                 potential_sigmas: torch.tensor,
                 eps: float):
        
        device = potential_probs.device
        self.dim = potential_mus[0].shape[0]
        
        self.log_probs = torch.log(potential_probs)
        
        eps = torch.tensor(eps).to(device)
        identity = torch.diag(torch.ones(self.dim)).to(device)
        
        self.potential_gaussians_distributions = [
            MultivariateNormal(loc=mu, covariance_matrix=sigma + eps*identity) for mu, sigma in zip(potential_mus, potential_sigmas)
        ]
        self.categorical_distribution = ConditionalCategoricalDistribution()
    
    
    def calculate_log_probs(self, x):
        log_probs = [
            log_prob + distribution.log_prob(x) for log_prob, distribution in zip(self.log_probs, self.potential_gaussians_distributions)
        ]
        log_probs = torch.stack(log_probs, dim=1)
        
        return log_probs
    
    
    def sample(self, x) -> torch.tensor:
        log_probs = self.calculate_log_probs(x)
        
        return self.categorical_distribution.sample(log_probs)
    

class ConditionalPlan:
    def __init__(self, potential_probs: torch.tensor, 
                 potential_mus: torch.tensor, 
                 potential_sigmas: torch.tensor,
                 eps: float,
                 device: str):
#         assert torch.allclose(potential_probs.sum(), torch.ones(1))
        assert len(potential_probs) == len(potential_mus) and len(potential_mus) == len(potential_sigmas)
        assert eps > 0
        
        self.device = device
        potential_probs = potential_probs.to(device)
        potential_mus = potential_mus.to(device)
        potential_sigmas = potential_sigmas.to(device)
        
        self.potential_probs = potential_probs
        self.potential_mus = potential_mus
        self.potential_sigmas = potential_sigmas
        
        self.dim = potential_mus[0].shape[0]
        
        self.components_distirubtion = PotentialCategoricalDistribution(
            potential_probs, potential_mus, potential_sigmas, eps
        )
        
        eps = torch.tensor(eps).to(device)
        identity = torch.diag(torch.ones(self.dim)).to(device)
        
        plan_sigmas = [torch.linalg.inv((1/eps) * identity + torch.linalg.inv(sigma)) for sigma in potential_sigmas]
        plan_mu_biases = [plan_sigma@torch.linalg.inv(sigma)@mu for mu, sigma, plan_sigma in zip(potential_mus, potential_sigmas, plan_sigmas)]
        self.plan_mu_weights = [plan_sigma/eps for plan_sigma in plan_sigmas]
        
        self.gaussians_distributions = [
            MultivariateNormal(loc=mu, covariance_matrix=sigma) for mu, sigma in zip(plan_mu_biases, plan_sigmas)
        ]
        
        
    def sample(self, x: torch.tensor) -> torch.tensor:
        batch_size = x.shape[0]
        components = self.components_distirubtion.sample(x)
        
        gaussian_samples = [
            gaussian_distribution.sample([batch_size]) + (plan_mu_weight@x.T).T for plan_mu_weight, gaussian_distribution in zip(self.plan_mu_weights, self.gaussians_distributions)
        ]
        gaussian_samples = torch.stack(gaussian_samples, dim=1)
            
        gaussian_mixture_samples = gaussian_samples.gather(1, components[:, None, None].expand(components.shape[0], 1, self.dim)).squeeze()
        
        return gaussian_mixture_samples

    
def download_gaussian_mixture_benchmark_files():    
    path = get_data_home()
    urls = {
        "gaussian_mixture_benchmark_data.zip": "https://drive.google.com/uc?id=1HNXbrkozARbz4r8fdFbjvPw8R74n1oiY",
    }
        
    for name, url in urls.items():
        gdown.download(url, os.path.join(path, f"{name}"), quiet=False)
        
    with ZipFile(os.path.join(path, "gaussian_mixture_benchmark_data.zip"), 'r') as zip_ref:
        zip_ref.extractall(path)
        
        
class OutputSampler:
    def __init__(self, gm, conditional_plan):
        self.gm = gm
        self.conditional_plan = conditional_plan
        
    def sample(self, n_samples: int =1):
        return self.conditional_plan.sample(self.gm.sample(n_samples))
    
    
class PlanSampler:
    def __init__(self, gm, conditional_plan):
        self.gm = gm
        self.conditional_plan = conditional_plan
        
    def sample(self, n_samples: int =1):
        input_samples = self.gm.sample(n_samples)
        output_samples = self.conditional_plan.sample(input_samples)
        return input_samples, output_samples
        
        
def get_guassian_mixture_benchmark_sampler(input_or_target: str, dim: int, eps: float,
                                           batch_size: int, device: str ="cpu", download: bool =False):
    assert input_or_target in ["input", "target"]
    assert dim in [2, 4, 8, 16, 32, 64, 128]
    assert eps in [0.01, 0.1, 1, 10]
    
    if download:
        download_gaussian_mixture_benchmark_files()
        
    benchmark_data_path = get_data_home()
        
    probs = torch.load(os.path.join(benchmark_data_path, f"input_probs_dim_{dim}.torch"))
    mus = torch.load(os.path.join(benchmark_data_path, f"input_mus_dim_{dim}.torch"))
    sigmas = torch.load(os.path.join(benchmark_data_path, f"input_sigmas_dim_{dim}.torch"))
    
    gm = GaussianMixture(probs, mus, sigmas, device=device)
        
    if input_or_target == "input":
        return gm
    else:
        probs = torch.load(os.path.join(benchmark_data_path, f"potential_probs_dim_{dim}_eps_{eps}.torch"))
        mus = torch.load(os.path.join(benchmark_data_path, f"potential_mus_dim_{dim}_eps_{eps}.torch"))
        sigmas = torch.load(os.path.join(benchmark_data_path, f"potential_sigmas_dim_{dim}_eps_{eps}.torch"))
        
        conditional_plan = ConditionalPlan(probs, mus, sigmas, eps, device=device)
        
        return OutputSampler(gm, conditional_plan)
    
    
def get_guassian_mixture_benchmark_ground_truth_sampler(dim: int, eps: float, batch_size: int,
                                                        device: str ="cpu", download: bool =False):
    assert dim in [2, 4, 8, 16, 32, 64, 128]
    assert eps in [0.01, 0.1, 1, 10]
    
    if download:
        download_gaussian_mixture_benchmark_files()
        
    benchmark_data_path = get_data_home()
        
    probs = torch.load(os.path.join(benchmark_data_path, f"input_probs_dim_{dim}.torch"))
    mus = torch.load(os.path.join(benchmark_data_path, f"input_mus_dim_{dim}.torch"))
    sigmas = torch.load(os.path.join(benchmark_data_path, f"input_sigmas_dim_{dim}.torch"))
    
    gm = GaussianMixture(probs, mus, sigmas, device=device)
    
    probs = torch.load(os.path.join(benchmark_data_path, f"potential_probs_dim_{dim}_eps_{eps}.torch"))
    mus = torch.load(os.path.join(benchmark_data_path, f"potential_mus_dim_{dim}_eps_{eps}.torch"))
    sigmas = torch.load(os.path.join(benchmark_data_path, f"potential_sigmas_dim_{dim}_eps_{eps}.torch"))

    conditional_plan = ConditionalPlan(probs, mus, sigmas, eps, device=device)
        
    return PlanSampler(gm, conditional_plan)


def get_test_input_samples(dim, device="cpu"):
    benchmark_data_path = get_data_home()
    return torch.load(os.path.join(benchmark_data_path, "test_inputs", f"test_data_dim_{dim}.torch")).to(device)


class LoaderFromSampler:
    def __init__(self, sampler, batch_size, num_batches):
        self.num_batches = num_batches
        self.sampler = sampler
        self.batch_size = batch_size
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
            
        self.current_batch += 1
        samples = self.sampler.sample(self.batch_size)
        return samples, torch.zeros(self.batch_size)
    