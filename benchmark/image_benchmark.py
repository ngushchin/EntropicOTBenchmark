import gdown
import os
import torch
import numpy as np
import json

from matplotlib import pyplot as plt

from torch.nn.functional import softmax

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.gumbel import Gumbel

from sklearn.decomposition import PCA
from scipy.stats import ortho_group
from scipy.linalg import sqrtm, inv
from zipfile import ZipFile

from .auxiliary import get_data_home
from .glow_model import Glow
from .glow_train import calc_z_shapes, sample_data
import torch
from torch import nn, optim

from .distributions import LoaderSampler
from torch.utils.data import TensorDataset, DataLoader


class GlowSampler:
    def __init__(self, glow_checkpoint_path, glow_device, samples_device):
        self.n_flow = 32
        self.n_block = 4
        self.affine = False
        self.no_lu = False
        self.img_size = 64
        self.n_sample = 20
        self.device = glow_device
        self.samples_device = samples_device
        
        model_single = Glow(
            3, self.n_flow, self.n_block, affine=self.affine, conv_lu=not self.no_lu
        )
        model = nn.DataParallel(model_single)
        self.model = model.to(glow_device)
        
        self.model.load_state_dict(torch.load(glow_checkpoint_path, map_location=glow_device))
    
    @torch.no_grad()
    def sample(self, batch_size, temp=0.7):
        z_sample = []
        z_shapes = calc_z_shapes(3, self.img_size, self.n_flow, self.n_block)
        for z in z_shapes:
            z_new = torch.randn(batch_size, *z) * temp
            z_sample.append(z_new.to(self.device))

        return 2*self.model.module.reverse(z_sample).to(self.samples_device)

    
# class FunctionSampler:
#     def __init__(self, sample_function, device, n_outputs=1):
#         self.sample_function = sample_function
#         self.device = device
#         self.n_outputs = n_outputs
    
#     def sample(self, *args, **kwargs):
#         if self.n_outputs == 1:
#             return self.sample_function(*args, **kwargs).to(self.device)
#         return tuple([tensor.to(self.device) for tensor in self.sample_function(*args, **kwargs)])
    
    
class ImageConditionalCategoricalDistribution:
    def __init__(self):
        self.gumbel_distribution = Gumbel(0, 1)
        
    def sample(self, log_probs: torch.tensor) -> torch.tensor:
        gumbel_samples = self.gumbel_distribution.sample(log_probs.shape).to(log_probs.device)
        return torch.argmax(gumbel_samples + log_probs, dim=1)
    
    
class ImagePotentialCategoricalDistribution:
    def __init__(self, potential_probs: torch.tensor,
                 potential_mus: torch.tensor,
                 potential_sigmas: torch.tensor,
                 eps: float):
        
        device = potential_probs.device
        self.dim = potential_mus[0].shape[0]
        
        self.log_probs = torch.log(potential_probs)
        
        eps = torch.tensor(eps).to(device)

        self.potential_gaussians_distributions = [
            Normal(loc=mu, scale=torch.sqrt(sigma + eps)) for mu, sigma in zip(potential_mus, potential_sigmas)
        ]
        self.categorical_distribution = ImageConditionalCategoricalDistribution()
    
    def sample(self, x) -> torch.tensor:
        log_probs = [
            log_prob + distribution.log_prob(x).sum(dim=(1,2,3)) for log_prob, distribution in zip(self.log_probs, self.potential_gaussians_distributions)
        ]
        log_probs = torch.stack(log_probs, dim=1)
        
        return self.categorical_distribution.sample(log_probs)
    

class ImageConditionalPlan:
    def __init__(self, potential_probs: torch.tensor, 
                 potential_mus: torch.tensor, 
                 potential_sigmas: torch.tensor,
                 eps: float,
                 device: str):
        assert len(potential_probs) == len(potential_mus) and len(potential_mus) == len(potential_sigmas)
        assert eps > 0
        
        self.device = device
        potential_probs, potential_mus, potential_sigmas = (
            potential_probs.to(device), potential_mus.to(device), potential_sigmas.to(device)
        )
        
        self.potential_probs = potential_probs
        self.potential_mus = potential_mus
        self.potential_sigmas = potential_sigmas
        
        self.dim = potential_mus[0].shape[0]
        
        self.components_distirubtion = ImagePotentialCategoricalDistribution(
            potential_probs, potential_mus, potential_sigmas, eps
        )
        
        eps = torch.tensor(eps).to(device)
        
        plan_sigmas = [1/((1/eps) + 1/sigma) for sigma in potential_sigmas]
        plan_mu_biases = [plan_sigma*(1/sigma)*mu for mu, sigma, plan_sigma in zip(potential_mus, potential_sigmas, plan_sigmas)]
        self.plan_mu_weights = [plan_sigma/eps for plan_sigma in plan_sigmas]
        
        self.gaussians_distributions = [
            Normal(loc=mu, scale=torch.sqrt(sigma)) for mu, sigma in zip(plan_mu_biases, plan_sigmas)
        ]
        
        
    def sample(self, x: torch.tensor) -> torch.tensor:
        batch_size = x.shape[0]
        components = self.components_distirubtion.sample(x)
        
        gaussian_samples = [
            gaussian_distribution.sample([batch_size]) + (plan_mu_weight[None, :]*x) for plan_mu_weight, gaussian_distribution in zip(self.plan_mu_weights, self.gaussians_distributions)
        ]
        gaussian_samples = torch.stack(gaussian_samples, dim=1)
            
        gaussian_mixture_samples = gaussian_samples.gather(
            1, components[:, None, None, None, None].expand(components.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
        ).squeeze()
        
        return gaussian_mixture_samples

    
class ImageOutputSampler:
    def __init__(self, glow_sampler, conditional_plan, batch_size=1):
        self.glow_sampler = glow_sampler
        self.conditional_plan = conditional_plan
        self.batch_size = batch_size
        
    def sample(self, n_samples = None):
        if n_samples is None:
            n_samples = self.batch_size
        return self.conditional_plan.sample(self.glow_sampler.sample(n_samples))

    
class ImagePlanSampler:
    def __init__(self, glow_sampler, conditional_plan, batch_size=1):
        self.glow_sampler = glow_sampler
        self.conditional_plan = conditional_plan
        self.batch_size = batch_size
        
    def sample(self, n_samples = None):
        if n_samples is None:
            n_samples = self.batch_size
        input_samples = self.glow_sampler.sample(n_samples)
        output_samples = self.conditional_plan.sample(input_samples)
        return input_samples, output_samples


def download_image_benchmark_files():    
    path = get_data_home()
    urls = {
        "image_benchmark.zip": "https://drive.google.com/uc?id=1QDHxJIYjHnpoJMq1IBwap5SlWJtyMjIi",
    }
        
    for name, url in urls.items():
        gdown.download(url, os.path.join(path, f"{name}"), quiet=False)
        
    with ZipFile(os.path.join(path, "image_benchmark.zip"), 'r') as zip_ref:
        zip_ref.extractall(path)


def get_image_benchmark_sampler(input_or_target: str, eps: float,
                                batch_size: int, glow_device: str ="cpu",
                                samples_device: str ="cpu", download: bool =False):
    assert input_or_target in ["input", "target"]
    
    if download:
        download_image_benchmark_files()
        
    benchmark_data_path = os.path.join(get_data_home(), "image_benchmark")

    glow_checkpoint_path = os.path.join(get_data_home(), "image_benchmark", "glow_model.pt")
    glow_sampler = GlowSampler(glow_checkpoint_path, glow_device, samples_device)
        
    if input_or_target == "target":
        return glow_sampler
    else:
        probs = torch.load(os.path.join(benchmark_data_path, "potentials", f"image_potential_probs_eps_{eps}.torch"))
        mus = torch.load(os.path.join(benchmark_data_path, "potentials", f"image_potential_mus_eps_{eps}.torch"))
        sigmas = torch.load(os.path.join(benchmark_data_path, "potentials", f"image_potential_sigmas_eps_{eps}.torch"))
        
        conditional_plan = ImageConditionalPlan(probs, mus, sigmas, eps, device=samples_device)
        
        return ImageOutputSampler(glow_sampler, conditional_plan, batch_size=batch_size)
    

def get_image_benchmark_ground_truth_sampler(eps: float, batch_size: int, glow_device: str ="cpu",
                                             samples_device: str ="cpu", download: bool =False):
    if download:
        download_image_benchmark_files()
    
    benchmark_data_path = os.path.join(get_data_home(), "image_benchmark")
    glow_checkpoint_path = os.path.join(get_data_home(), "image_benchmark", "glow_model.pt")
    glow_sampler = GlowSampler(glow_checkpoint_path, glow_device, samples_device)
    
    probs = torch.load(os.path.join(benchmark_data_path, "potentials", f"image_potential_probs_eps_{eps}.torch"))
    mus = torch.load(os.path.join(benchmark_data_path, "potentials", f"image_potential_mus_eps_{eps}.torch"))
    sigmas = torch.load(os.path.join(benchmark_data_path, "potentials", f"image_potential_sigmas_eps_{eps}.torch"))

    conditional_plan = ImageConditionalPlan(probs, mus, sigmas, eps, device=samples_device)
        
    return ImagePlanSampler(glow_sampler, conditional_plan, batch_size=batch_size)


def load_input_test_images(eps):
    path = os.path.join(get_data_home(), "image_benchmark", "test_images",
                        f"input_images_eps_{eps}.torch")
    X = torch.load(path)
    return X


def load_output_test_images():
    path = os.path.join(get_data_home(), "image_benchmark", "test_images",
                        f"output_images.torch")
    Y = torch.load(path)
    return Y


def load_input_test_image_sampler(eps, batch_size=64, shuffle=True, device='cuda', num_workers=8):
    X = load_input_test_images(eps)
    dataset = TensorDataset(X, torch.zeros_like(X))
    sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size), device)
    
    return sampler


def load_output_test_image_sampler(eps, batch_size=64, shuffle=True, device='cuda', num_workers=8):
    Y = load_output_test_images()
    dataset = TensorDataset(Y, torch.zeros_like(Y))
    sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size), device)
    
    return sampler


class ImageBenchmark:
    def __init__(self, batch_size, eps, glow_device, samples_device, download=False, num_workers=8):   
        if download:
            download_image_benchmark_files()
        
        self.Y_test_sampler = load_input_test_image_sampler(eps=eps, batch_size=batch_size, device=samples_device, num_workers=num_workers)
        self.X_test_sampler = load_output_test_image_sampler(eps=eps, batch_size=batch_size, device=samples_device, num_workers=num_workers)

        glow_checkpoint_path = os.path.join(get_data_home(), "image_benchmark", "glow_model.pt")
        glow_sampler = GlowSampler(glow_checkpoint_path, glow_device, samples_device)

        self.Y_sampler = get_image_benchmark_sampler("input", eps=eps, batch_size=batch_size,
                                                     samples_device=samples_device, glow_device=glow_device)
        self.X_sampler = get_image_benchmark_sampler("target", eps=eps, batch_size=batch_size,
                                                 samples_device=samples_device, glow_device=glow_device)
        
        self.GT_sampler = get_image_benchmark_ground_truth_sampler(eps=eps, batch_size=batch_size, glow_device=glow_device,
                                                                   samples_device=samples_device, download=download)
        
        stats_filename = os.path.join(get_data_home(), "image_benchmark", "Image_Y_test_stats.json")
        with open(stats_filename, 'r') as fp:
            data_stats = json.load(fp)
            Y_test_inception_mu, Y_test_inception_sigma = data_stats['mu'], data_stats['sigma']
        
        self.Y_test_inception_mu = Y_test_inception_mu
        self.Y_test_inception_sigma = Y_test_inception_sigma
