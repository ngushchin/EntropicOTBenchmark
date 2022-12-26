import torch
import numpy as np

from matplotlib import pyplot as plt

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

from sklearn.decomposition import PCA
from scipy.stats import ortho_group
from scipy.linalg import sqrtm, inv


def symmetrize(X):
    return np.real((X + X.T) / 2)

class GaussianMixture:
    def __init__(self, probs, mu, sigma, sqrt_sigma):
        self.probs = probs.reshape(-1)
        self.mu = mu
        self.sigma = sigma
        self.sqrt_sigma = sqrt_sigma
        
    def sample(self, N_samples=1):
        dim = self.mu.shape[-1]
        indexes = np.arange(self.probs.shape[0])
        chosen_gaussians = np.random.choice(indexes, size=N_samples, replace=True, p=np.array(self.probs))
        
        chosen_mu = self.mu[chosen_gaussians]
        chosen_sqrt_sigma = self.sqrt_sigma[chosen_gaussians]
    
        return (chosen_sqrt_sigma@torch.randn(N_samples, dim, 1))[:, :, 0] + chosen_mu, chosen_gaussians
    
    def get_params(self):
        return {"probs": torch.clone(self.probs), "mu": torch.clone(self.mu),
                "sigma": torch.clone(self.sigma), "sqrt_sigma": torch.clone(self.sqrt_sigma)}


class ConditionalGaussianMixture:
    def __init__(self, probs, A, sqrt_A, B, eps):
        self.probs = probs
        self.A = A
        self.sqrt_A = sqrt_A
        self.B = B
        
    def sample(self, x, samples_per_each_x=10):
        batch_size = x.shape[0]
        x = torch.repeat_interleave(x, samples_per_each_x, 0)
        x = x.reshape(batch_size, samples_per_each_x, *x.shape[1:])
        
        dim = self.A.shape[-1]
        indexes = np.arange(self.probs.shape[0])
        chosen_gaussians = np.random.choice(indexes, size=batch_size*samples_per_each_x, replace=True, p=np.array(self.probs))
        
        chosen_A = self.A[chosen_gaussians].reshape(batch_size, samples_per_each_x, *self.A.shape[1:])
        chosen_sqrt_A = self.sqrt_A[chosen_gaussians].reshape(batch_size, samples_per_each_x, *self.sqrt_A.shape[1:])
        chosen_B = self.B[chosen_gaussians].reshape(batch_size, samples_per_each_x, *self.B.shape[1:])
        
        mu = (chosen_A@x[:, :, :, None])[:, :, :, 0] + chosen_B
        sigma = chosen_sqrt_A*np.sqrt(eps)
        
        return (sigma@torch.randn(batch_size, samples_per_each_x, dim, 1))[:, :, :, 0] + mu, chosen_gaussians
    
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


def create_target_distribution(original_distr, conditional_distr, eps=1):    
    original_distr_params = original_distr.get_params()
    p_x_n_gaussians = original_distr_params["probs"].shape[0]
    p_x_probs = original_distr_params["probs"]
    p_x_mu = original_distr_params["mu"]
    p_x_sigma = original_distr_params["sigma"]
    
    conditional_distr_params = conditional_distr.get_params()
    p_y_x_n_gaussians = conditional_distr_params["probs"].shape[0]
    p_y_x_probs = conditional_distr_params["probs"]
    p_y_x_A = conditional_distr_params["A"]
    p_y_x_B = conditional_distr_params["B"]
    
    p_y_probs = []
    p_y_mu = []
    p_y_sigma = []
    p_y_sigma_sqrt = []

    for i in range(p_y_x_n_gaussians):
        for j in range(p_x_n_gaussians):
            prob = p_y_x_probs[i]*p_x_probs[j]
            mu = p_y_x_A[i]@p_x_mu[j] + p_y_x_B[i]
            sigma = eps*p_y_x_A[i] + p_y_x_A[i]@p_x_sigma[j]@p_y_x_A[i].T

            p_y_probs.append(prob)
            p_y_mu.append(mu)
            p_y_sigma.append(sigma)
            p_y_sigma_sqrt.append(torch.tensor(symmetrize(sqrtm(sigma.numpy()))))

    p_y_probs = torch.stack(p_y_probs)
    p_y_mu = torch.stack(p_y_mu)
    p_y_sigma = torch.stack(p_y_sigma)
    p_y_sigma_sqrt = torch.stack(p_y_sigma_sqrt)
    
    return GaussianMixture(p_y_probs, p_y_mu, p_y_sigma, p_y_sigma_sqrt)


def create_conditional_distr(probs, DIM=2, eps=1, prior_scale=10, seed=0xC0EEEA):
    OUTPUT_SEED = seed

    np.random.seed(OUTPUT_SEED)
    torch.manual_seed(OUTPUT_SEED)
    
    n_gaussians = len(probs)
    
    As = []
    Bs = []
    sqrt_As = []
    
    for i in range(n_gaussians):
        rotation_X = ortho_group.rvs(DIM)
        sqrt_A = torch.tensor(rotation_X @ np.diag(np.exp(np.linspace(np.log(0.9), np.log(1.1), DIM)))).float()
        B = prior_scale*torch.randn(DIM).float()
#         pdb.set_trace()
        As.append(sqrt_A@sqrt_A.T)
        sqrt_As.append(sqrt_A)
        Bs.append(B)
        
    As = torch.stack(As)
    sqrt_As = torch.stack(sqrt_As)
    Bs = torch.stack(Bs)

    return ConditionalGaussianMixture(probs, As, sqrt_As, Bs, eps)


class GaussianBenchmark:
    def __init__(self, dim, eps=1):
        n_gaussians = 3
        
        assert dim in {2, 4, 8, 16, 32, 64, 128}
        
        mu_prior_scales = {2:10, 4:10, 8:4, 16:3, 32:3, 64:2, 128:1.5}
        prior_scales = {2:5, 4:5, 8:5, 16:5, 32:3, 64:3, 128:1.5}
        
        mu_prior_scale = mu_prior_scales[dim]
        prior_scale = prior_scales[dim]
        
        n_gaussians = 3

        mus, sigmas, sqrt_sigmas = initialize_gaussians(n_gaussians=n_gaussians, DIM=dim, mu_prior_scale=mu_prior_scale, seed=1)
        probs = (1/n_gaussians)*torch.ones(n_gaussians)
        original_distr = GaussianMixture(probs, mus, sigmas, sqrt_sigmas)

        n_gaussians=3
        probs = (1/n_gaussians)*torch.ones(n_gaussians)
        conditional_distr = create_conditional_distr(probs=probs, DIM=dim, seed=0, eps=eps, prior_scale=prior_scale)

        target_distr = create_target_distribution(original_distr, conditional_distr, eps=eps)
        
        self.X_sampler = original_distr
        self.Y_sampler = target_distr