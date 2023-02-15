import os

from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import matplotlib.pyplot as plt

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torchvision.transforms.functional as F

from .glow_model import Glow
from .glow_train import calc_z_shapes, sample_data

import gc
import gdown

from .distributions import LoaderSampler
from torch.utils.data import TensorDataset, DataLoader

from .auxiliary import get_data_home
from zipfile import ZipFile


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
        
        self.model.load_state_dict(torch.load(glow_checkpoint_path))
    
    @torch.no_grad()
    def sample_images(self, batch_size, temp=0.7):
        z_sample = []
        z_shapes = calc_z_shapes(3, self.img_size, self.n_flow, self.n_block)
        for z in z_shapes:
            z_new = torch.randn(batch_size, *z) * temp
            z_sample.append(z_new.to(self.device))

        return 2*self.model.module.reverse(z_sample).to(self.samples_device)
    
    @torch.no_grad()
    def sample_degraded_images(self, batch_size, temp=0.7, eps=0.01):
        gen_images = self.sample_images(batch_size, temp).to(self.samples_device)
        samples = gen_images + np.sqrt((eps))*torch.randn_like(gen_images).to(self.samples_device)

        return samples.to(self.samples_device)
    
    
    @torch.no_grad()
    def sample_pair_image_and_degraded_image(self, batch_size, temp=0.7, eps=0.01):
        gen_images = self.sample_images(batch_size, temp).to(self.samples_device)
        samples = gen_images + np.sqrt((eps))*torch.randn_like(gen_images).to(self.samples_device)

        return samples.to(self.samples_device), gen_images

    
class FunctionSampler:
    def __init__(self, sample_function, device, n_outputs=1):
        self.sample_function = sample_function
        self.device = device
        self.n_outputs = n_outputs
    
    def sample(self, *args, **kwargs):
        if self.n_outputs == 1:
            return self.sample_function(*args, **kwargs).to(self.device)
        return tuple([tensor.to(self.device) for tensor in self.sample_function(*args, **kwargs)])


class ImageBenchmark:
    def __init__(self, X_sampler, Y_sampler, X_test_sampler, Y_test_sampler, X_Y_sampler, X_Y_test_sampler):
        self.X_sampler = X_sampler
        self.Y_sampler = Y_sampler
        self.X_test_sampler = X_test_sampler
        self.Y_test_sampler = Y_test_sampler
        self.X_Y_sampler = X_Y_sampler
        self.X_Y_test_sampler = X_Y_test_sampler

        
def load_test_image_pairs(eps):
    path = os.path.join(get_data_home(), "image_benchmark", "glow_sampler_generated_data",
                        f"glow_generated_pairs_images_eps_{eps}.torch")
    image_pairs = torch.load(path)
    X, Y = image_pairs[0], image_pairs[1]
    
    return X, Y


def load_input_test_image_sampler(eps, batch_size=64, shuffle=True, device='cuda', num_workers=8):
    X, Y = load_test_image_pairs(eps)
    dataset = TensorDataset(X, torch.zeros_like(X))
    sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size), device)
    
    return sampler


def load_output_test_image_sampler(eps, batch_size=64, shuffle=True, device='cuda', num_workers=8):
    X, Y = load_test_image_pairs(eps)
    dataset = TensorDataset(Y, torch.zeros_like(Y))
    sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size), device)
    
    return sampler


def load_pairs_test_image_sampler(eps, batch_size=64, shuffle=True, device='cuda', num_workers=8):
    X, Y = load_test_image_pairs(eps)
    dataset = TensorDataset(X, Y)
    sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size), device, sample_Y=True)
    
    return sampler


def download_image_benchmark_glow_checkpoint():
    path = os.path.join(get_data_home(), "image_benchmark")
    os.makedirs(path, exist_ok=True)
    urls = {
        "glow_model.pt": "https://drive.google.com/uc?id=14G8LtmF3QH5yYSYG0FzV6xjJxYuec1Ww",
    }
    
    for name, url in urls.items():
        gdown.download(url, os.path.join(path, name), quiet=False)


def download_image_benchmark_files():
    path = os.path.join(get_data_home(), "image_benchmark")
    os.makedirs(path, exist_ok=True)
    
    urls = {
        "glow_model.pt": "https://drive.google.com/uc?id=14G8LtmF3QH5yYSYG0FzV6xjJxYuec1Ww",
        "glow_sampler_generated_data.zip": "https://drive.google.com/uc?id=1GFxRJlnujy8A2AYaMSRxjeDWp748dc9m",
    }
    
    for name, url in urls.items():
        gdown.download(url, os.path.join(path, name), quiet=False)
        
    with ZipFile(os.path.join(path, "glow_sampler_generated_data.zip"), 'r') as zip_ref:
        zip_ref.extractall(path)

    
def get_image_benchmark(batch_size, eps, glow_device, samples_device, download=False, use_pairs=False, num_workers=8):
    if download:
        download_image_benchmark_files()

    X_test_sampler = load_input_test_image_sampler(eps=eps, batch_size=batch_size, device=samples_device, num_workers=num_workers)
    Y_test_sampler = load_output_test_image_sampler(eps=eps, batch_size=batch_size, device=samples_device, num_workers=num_workers)
    if use_pairs:
        X_Y_test_sampler = load_pairs_test_image_sampler(eps=eps, batch_size=batch_size, device=samples_device, num_workers=num_workers)
    
    glow_checkpoint_path = os.path.join(get_data_home(), "image_benchmark", "glow_model.pt")
    glow_sampler = GlowSampler(glow_checkpoint_path, glow_device, samples_device)

    X_sampler = FunctionSampler(lambda batch_size: glow_sampler.sample_degraded_images(batch_size, eps=eps),
                               device=samples_device)
    Y_sampler = FunctionSampler(glow_sampler.sample_images,
                               device=samples_device)
    if use_pairs:
        X_Y_sampler = FunctionSampler(lambda batch_size: glow_sampler.sample_pair_image_and_degraded_image(batch_size, eps=eps),
                                   device=samples_device, n_outputs=2)
    
    if use_pairs:
        return ImageBenchmark(X_sampler, Y_sampler, X_test_sampler, Y_test_sampler, X_Y_sampler, X_Y_test_sampler)
    
    return ImageBenchmark(X_sampler, Y_sampler, X_test_sampler, Y_test_sampler, None, None)
