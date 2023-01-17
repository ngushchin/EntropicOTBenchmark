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

from src.model import Glow
from src.train import calc_z_shapes, sample_data

import gc
import gdown

from .distributions import LoaderSampler
from torch.utils.data import TensorDataset, DataLoader


class BehcnmarkGlowSampler:
    def __init__(self, glow_device, samples_device):
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
        
        self.model.load_state_dict(torch.load("../data/glow_model.pt"))
    
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


class AnimeBenchmark:
    def __init__(self, X_sampler, Y_sampler, X_test_sampler, Y_test_sampler, X_Y_sampler):
        self.X_sampler = X_sampler
        self.Y_sampler = Y_sampler
        self.X_test_sampler = X_test_sampler
        self.Y_test_sampler = Y_test_sampler
        self.X_Y_sampler = X_Y_sampler

        
def load_output_test_anime_dataset(batch_size=64, shuffle=True, device='cuda'):
    path = "../data/glow_generated_images.torch"
    images = torch.load(path)
    dataset = TensorDataset(images, torch.zeros_like(images))
    sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=8, batch_size=batch_size), device)
    
    return sampler


def load_input_test_anime_dataset(eps, batch_size=64, shuffle=True, device='cuda'):
    path = f"../data/glow_generated_degrated_images_eps_{eps}.torch"
    images = torch.load(path)
    dataset = TensorDataset(images, torch.zeros_like(images))
    sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=8, batch_size=batch_size), device)
    
    return sampler


def download_benchmark_files():
    urls = {
        "glow_model.pt": "https://drive.google.com/uc?id=19zI6OH48v0Z5rQuiCJesZQK_hvpJipZb",
        "glow_generated_images.torch": "https://drive.google.com/uc?id=1varC4Zjeff-j9iEP9qmVYygTSc2g5iJJ",
        "glow_generated_degrated_images_eps_0.1.torch": "https://drive.google.com/uc?id=126GVodokBty493Z28fWsTVZOGcmNLZNL",
        "glow_generated_degrated_images_eps_1.torch": "https://drive.google.com/uc?id=11TaagsJ4EBlwz6FVcVNxafPpqdgzL2WF",
        "glow_generated_degrated_images_eps_10.torch": "https://drive.google.com/uc?id=1gpMFQNJ0KBgT-REvr7zA9GZtzyP51HWg",
        "glow_generated_degrated_images_eps_100.torch": "https://drive.google.com/uc?id=1qYdQC8UsDNE2gmOY1G2sxwJSWULKwcQ-",
    }
    for name, url in urls.items():
        gdown.download(url, f"../data/{name}", quiet=False)

    
def get_anime_benchmark(batch_size, eps, glow_device, samples_device, download=False):
    if download:
        if not os.path.exists("../data"):
            os.mkdir("../data")
        download_benchmark_files()

    X_test_sampler = load_input_test_anime_dataset(eps=eps, batch_size=batch_size, device=samples_device)
    Y_test_sampler = load_output_test_anime_dataset(batch_size=batch_size, device=samples_device)

    glow_sampler = BehcnmarkGlowSampler(glow_device, samples_device)

    X_sampler = FunctionSampler(lambda batch_size: glow_sampler.sample_degraded_images(batch_size, eps=eps),
                               device=samples_device)
    Y_sampler = FunctionSampler(glow_sampler.sample_images,
                               device=samples_device)
    X_Y_sampler = FunctionSampler(lambda batch_size: glow_sampler.sample_pair_image_and_degraded_image(batch_size, eps=eps),
                               device=samples_device, n_outputs=2)
    
    return AnimeBenchmark(X_sampler, Y_sampler, X_test_sampler, Y_test_sampler, X_Y_sampler)
