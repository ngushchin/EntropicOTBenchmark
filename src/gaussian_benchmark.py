import torch
import numpy as np

from matplotlib import pyplot as plt

from torch.nn.functional import softmax

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

from sklearn.decomposition import PCA
from scipy.stats import ortho_group
from scipy.linalg import sqrtm, inv



def symmetrize(X):
    return np.real((X + X.T) / 2)


class GaussianMixture:
    def __init__(self, probs, mu, sigma, sqrt_sigma, device):
        self.probs = probs.reshape(-1)
        self.mu = mu
        self.sigma = sigma
        self.sqrt_sigma = sqrt_sigma
        self.device = device
        
    def sample(self, N_samples=1):
        dim = self.mu.shape[-1]
        indexes = np.arange(self.probs.shape[0])
        chosen_gaussians = np.random.choice(indexes, size=N_samples, replace=True, p=np.array(self.probs))
        
        chosen_mu = self.mu[chosen_gaussians]
        chosen_sqrt_sigma = self.sqrt_sigma[chosen_gaussians]
        
        sample = (chosen_sqrt_sigma@torch.randn(N_samples, dim, 1))[:, :, 0] + chosen_mu
    
        return sample.to(self.device), chosen_gaussians
    
    def get_params(self):
        return {"probs": torch.clone(self.probs), "mu": torch.clone(self.mu),
                "sigma": torch.clone(self.sigma), "sqrt_sigma": torch.clone(self.sqrt_sigma)}


class ConditionalGaussianMixture:
    def __init__(self, probs, A, sqrt_A, B, potential_distributions, eps, device):
        self.probs = probs
        self.A = A
        self.sqrt_A = sqrt_A
        self.B = B
        self.eps = eps
        self.potential_distributions = potential_distributions
        self.device = device
        
    def sample(self, x, samples_per_each_x=10):
        batch_size = x.shape[0]
        x = torch.repeat_interleave(x, samples_per_each_x, 0)
        x = x.reshape(batch_size, samples_per_each_x, *x.shape[1:])
        
        dim = self.A.shape[-1]
        indexes = np.arange(self.probs.shape[0])
        log_p = [
            torch.log(prob) + distr.log_prob(x[:, 0]) for prob, distr in zip(self.probs, self.potential_distributions)
        ]
        log_p = torch.stack(log_p, dim=1)
        ps = softmax(log_p, dim=-1)
        
        chosen_gaussians = []
        for p in ps:
            chosen_gaussians.append(
                torch.tensor(np.random.choice(indexes, size=samples_per_each_x, replace=True, p=p.numpy()))
            )
        chosen_gaussians = torch.stack(chosen_gaussians, dim=0)
        
        mask = chosen_gaussians == torch.arange(len(self.probs))
        
        chosen_A = self.A.expand(mask.shape[0], mask.shape[1], self.A.shape[1], self.A.shape[1])
        chosen_A = chosen_A[mask].reshape(batch_size, samples_per_each_x, *self.A.shape[1:])
        
        chosen_sqrt_A = self.sqrt_A.expand(mask.shape[0], mask.shape[1], self.sqrt_A.shape[1], self.sqrt_A.shape[1])
        chosen_sqrt_A = chosen_sqrt_A[mask].reshape(batch_size, samples_per_each_x, *self.sqrt_A.shape[1:])
        
        chosen_B = self.B.expand(mask.shape[0], mask.shape[1], self.B.shape[1])
        chosen_B = chosen_B[mask].reshape(batch_size, samples_per_each_x, *self.B.shape[1:])
        
        mu = (chosen_A@x[:, :, :, None])[:, :, :, 0] + chosen_B
        sigma = chosen_sqrt_A*np.sqrt(self.eps)
        
        sample = (sigma@torch.randn(batch_size, samples_per_each_x, dim, 1))[:, :, :, 0] + mu
        
        return sample.to(self.device), chosen_gaussians
    
    def get_params(self):
        return {"probs": torch.clone(self.probs), "A": torch.clone(self.A),
                "sqrt_A": torch.clone(self.sqrt_A), "B": torch.clone(self.B)}
    
    
def initialize_gaussians(n_gaussians=1, DIM=2, mu_prior_scale=10, seed=0xC0EEEF):
    OUTPUT_SEED = seed

    np.random.seed(OUTPUT_SEED)
    torch.manual_seed(OUTPUT_SEED)
    
    mus = []
    sigmas = []
    sqrt_sigmas = []
    
    for i in range(n_gaussians):
        rotation_X = ortho_group.rvs(DIM)
        weight_X = torch.tensor(rotation_X @ np.diag(np.exp(np.linspace(np.log(0.5), np.log(2), DIM))))
        sigma = weight_X@weight_X.T
        mu = mu_prior_scale*torch.randn(DIM)
        
        mus.append(mu)
        sigmas.append(sigma)
        sqrt_sigmas.append(weight_X)
        
    mus = torch.stack(mus)
    sigmas = torch.stack(sigmas)
    sqrt_sigmas = torch.stack(sqrt_sigmas)

    return mus.float(), sigmas.float(), sqrt_sigmas.float()


def create_conditional_distr(probs, DIM=2, eps=1, prior_scale=10, seed=0xC0EEEA, device="cpu"):
    OUTPUT_SEED = seed

    np.random.seed(OUTPUT_SEED)
    torch.manual_seed(OUTPUT_SEED)
    
    n_gaussians = len(probs)
    
    As = []
    Bs = []
    sqrt_As = []
    potential_distributions = []      
    
    for i in range(n_gaussians):
        rotation_X = ortho_group.rvs(DIM)
        sigma_p_potential_inv_sqrt = torch.tensor(rotation_X @ np.diag(np.exp(np.linspace(np.log(0.9), np.log(1.1), DIM)))).float()
        sigma_p_potential_inv = sigma_p_potential_inv_sqrt@sigma_p_potential_inv_sqrt.T
                                      
        mu_potential = prior_scale*torch.randn(DIM).float()
        sigma_conditional = torch.tensor(inv(1/eps + sigma_p_potential_inv))
        
        A = sigma_conditional/eps
        sqrt_A = torch.tensor(symmetrize(sqrtm(A)))
        
        B = sigma_conditional@sigma_p_potential_inv@mu_potential
                                      
        As.append(A)
        sqrt_As.append(sqrt_A)
        Bs.append(B)
        potential_distributions.append(MultivariateNormal(loc=mu_potential, covariance_matrix=sigma_conditional))
        
    As = torch.stack(As)
    sqrt_As = torch.stack(sqrt_As)
    Bs = torch.stack(Bs)

    return ConditionalGaussianMixture(probs, As, sqrt_As, Bs, potential_distributions, eps, device)


class TargetDistribution:
    def __init__(self, original_distr, conditional_distr):
        self.original_distr = original_distr
        self.conditional_distr = conditional_distr
    
    def sample(self, batch_size):
        samples, labels = self.conditional_distr.sample(self.original_distr.sample(batch_size)[0], 1)
        return samples[:, 0], labels


class GaussianBenchmark:
    def __init__(self, dim, eps, device):        
        assert dim in {2, 4, 8, 16, 32, 64, 128}
        
        mu_prior_scales = {2:8, 4:10, 8:4, 16:3, 32:3, 64:2, 128:1.5}
        prior_scales = {2:1, 4:5, 8:5, 16:5, 32:3, 64:3, 128:1.5}
        
        mu_prior_scale = mu_prior_scales[dim]
        prior_scale = prior_scales[dim]
        
        n_gaussians = 3

        mus, sigmas, sqrt_sigmas = initialize_gaussians(n_gaussians=n_gaussians, DIM=dim, mu_prior_scale=mu_prior_scale, seed=1)
        probs = (1/n_gaussians)*torch.ones(n_gaussians)
        original_distr = GaussianMixture(probs, mus, sigmas, sqrt_sigmas, device)

        n_gaussians = 5
        probs = (1/n_gaussians)*torch.ones(n_gaussians)
        conditional_distr = create_conditional_distr(probs=probs, DIM=dim, seed=0, eps=eps, prior_scale=prior_scale, device=device)

#         target_distr = create_target_distribution(original_distr, conditional_distr, eps=eps, device=device)
        target_distr = TargetDistribution(original_distr, conditional_distr)
        self.X_sampler = original_distr
        self.Y_sampler = target_distr