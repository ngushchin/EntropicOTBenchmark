import numpy as np
from scipy.linalg import sqrtm, inv


def symmetrize(X):
    return np.real((X + X.T) / 2)


def get_D_sigma(covariance_0, covariance_T, epsilon):
    shape = covariance_0.shape[0]
    
    covariance_0_sqrt = symmetrize(sqrtm(covariance_0))
    return symmetrize(sqrtm(4*covariance_0_sqrt@covariance_T@covariance_0_sqrt + (epsilon**2)*np.eye(shape)))


def get_C_sigma(covariance_0, D_sigma, epsilon):
    shape = covariance_0.shape[0]
    
    covariance_0_sqrt = symmetrize(sqrtm(covariance_0))
    covariance_0_sqrt_inv = inv(covariance_0_sqrt)
    
    return 0.5*(covariance_0_sqrt@D_sigma@covariance_0_sqrt_inv - epsilon*np.eye(shape))


def get_mu_t(t, mu_0, mu_T):
    return (1 - t)*mu_0 + t*mu_T


def get_covariance_t(t, covariance_0, covariance_T, C_sigma, epsilon):
    shape = covariance_0.shape[0]
    
    return (
        ((1-t)**2)*covariance_0 + (t**2)*covariance_T + 
        t*(1-t)*(C_sigma+C_sigma.T) + epsilon*t*(1-t)*np.eye(shape)
    )


def get_conditional_covariance_t(t, covariance_0, covariance_T, C_sigma, epsilon):
    shape = covariance_0.shape[0]
    
    covariance_0_inv = inv(covariance_0)
    
    return (
        (t**2)*(covariance_T - C_sigma.T@covariance_0_inv@C_sigma) + 
        epsilon*t*(1-t)*np.eye(shape)
    )


def get_conditional_mu_t(x0, mu_0, mu_T, t, covariance_0, C_sigma, epsilon):
    shape = covariance_0.shape[0]
    
    covariance_0_inv = inv(covariance_0)
    
    return (
        (1-t)*x0 + t*(mu_T + C_sigma.T@covariance_0_inv@(x0[:, None] - mu_0[:, None]))
    )


def get_optimal_plan_covariance(covariance_0, covariance_T, eps):
    covariance_0 = covariance_0
    covariance_T = covariance_T
    
    D_sigma = get_D_sigma(covariance_0, covariance_T, eps)
    C_sigma = get_C_sigma(covariance_0, D_sigma, eps)
    
    size = covariance_0.shape[0]
    optimal_plan_covariance = np.zeros((2*size, 2*size))
    
    optimal_plan_covariance[:size, :size] = covariance_0
    optimal_plan_covariance[size:, size:] = covariance_T
    
    optimal_plan_covariance[:size, size:] = C_sigma
    optimal_plan_covariance[size:, :size] = C_sigma.T
    
    return optimal_plan_covariance