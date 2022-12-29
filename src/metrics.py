import torch
import numpy as np
from .tools import freeze

import gc

def score_fitted_maps(benchmark, D, D_conj, size=4096):
    '''Estimates L2-UVP and cosine metrics for transport map by the gradients of potentials.'''
    X = benchmark.input_sampler.sample(size); X.requires_grad_(True)
    Y = benchmark.map_fwd(X, nograd=True); Y.requires_grad_(True)
    
    freeze(D); freeze(D_conj)
    X_push = D.push_nograd(X)
    Y_inv = D_conj.push_nograd(Y)
    
    with torch.no_grad():
        L2_UVP_fwd = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()
        L2_UVP_inv = 100 * (((X - Y_inv) ** 2).sum(dim=1).mean() / benchmark.input_sampler.var).item()
        
        cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
        (np.sqrt((2 * benchmark.cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
        cos_inv = (((X - Y) * (Y_inv - Y)).sum(dim=1).mean() / \
        (np.sqrt((2 * benchmark.cost) * ((Y_inv - Y) ** 2).sum(dim=1).mean().item()))).item()
        
    gc.collect(); torch.cuda.empty_cache() 
    return L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv

def score_baseline_maps(benchmark, baseline='linear', size=4096):
    '''Estimates L2-UVP and cosine similarity metrics for the baseline transport map (identity, const or linear).'''
    assert baseline in ['identity', 'linear', 'constant']
    
    X = benchmark.input_sampler.sample(size); X.requires_grad_(True)
    Y = benchmark.map_fwd(X, nograd=True)
    
    with torch.no_grad():
        if baseline == 'linear':  
            X_push = benchmark.linear_map_fwd(X)
            Y_inv = benchmark.linear_map_inv(Y)
        elif baseline == 'constant':
            X_push = torch.tensor(
                benchmark.output_sampler.mean.reshape(1, -1).repeat(size, 0),
                dtype=torch.float32
            ).to(X)
            Y_inv = torch.tensor(
                benchmark.input_sampler.mean.reshape(1, -1).repeat(size, 0),
                dtype=torch.float32
            ).to(Y)
        elif baseline == 'identity':
            X_push = X
            Y_inv = Y

        if baseline == 'constant':
            L2_UVP_fwd, L2_UVP_inv = 100., 100.
        else:
            L2_UVP_fwd = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()
            L2_UVP_inv = 100 * (((X - Y_inv) ** 2).sum(dim=1).mean() / benchmark.input_sampler.var).item()

        if baseline == 'identity':
            cos_fwd, cos_inv = 0., 0.
        else:
            cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
            (np.sqrt(2 * benchmark.cost * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
            cos_inv = (((X - Y) * (Y_inv - Y)).sum(dim=1).mean() / \
            (np.sqrt(2 * benchmark.cost * ((Y_inv - Y) ** 2).sum(dim=1).mean().item()))).item()

    gc.collect(); torch.cuda.empty_cache() 
    return L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv

def metrics_to_dict(L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv):
    return dict(L2_UVP_fwd=L2_UVP_fwd, cos_fwd=cos_fwd, L2_UVP_inv=L2_UVP_inv, cos_inv=cos_inv)


# https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
def MMD(x, y, kernel, device, scale_factor):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [elem*scale_factor for elem in [0.2, 0.5, 0.9, 1.3]]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [elem*scale_factor for elem in [10, 15, 20, 50]]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)