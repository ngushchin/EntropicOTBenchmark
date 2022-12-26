import pandas as pd
import numpy as np

import os
import itertools

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm_notebook
import multiprocessing

from .resnet2 import ResNet_G
from .icnn import View

from PIL import Image
from .inception import InceptionV3
from tqdm import tqdm
from .distributions import LoaderSampler
from torch.utils.data import TensorDataset, DataLoader

import gc


def load_dataset(name, path, img_size=64, batch_size=64, shuffle=True, device='cuda'):
    if name == "glow_generated_test":
        images = torch.load(path)*2 - 1
        dataset = TensorDataset(images, torch.zeros_like(images))
        sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=8, batch_size=batch_size), device)
    elif name == "glow_generated_degraded_test":
        images = torch.load(path)*2 - 1
        dataset = TensorDataset(images, torch.zeros_like(images))
        sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=8, batch_size=batch_size), device)

    return sampler


def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)   
    
def train_identity_map(D, sampler, batch_size=1024, max_iter=5000, lr=1e-3, tol=1e-3, blow=3, convex=True, verbose=False):
    "Trains potential D to satisfy D.push(x)=x by using MSE loss w.r.t. sampler's distribution"
    unfreeze(D)
    opt = torch.optim.Adam(D.parameters(), lr=lr, weight_decay=1e-10)
    if verbose:
        print('Training the potentials to satisfy push(x)=x')
    for iteration in tqdm_notebook(range(max_iter)) if verbose else range(max_iter):
        X = sampler.sample(batch_size)
        with torch.no_grad():
            X *= blow
        X.requires_grad_(True)

        loss = F.mse_loss(D.push(X), X.detach())
        loss.backward()
        opt.step(); opt.zero_grad()
        
        if convex:
            D.convexify()

        if loss.item() < tol:
            break
            
    loss = loss.item()
    gc.collect(); torch.cuda.empty_cache()
    return loss

def load_resnet_G(cpkt_path, device='cuda'):
    resnet = nn.Sequential(
        ResNet_G(128, 64, nfilter=64, nfilter_max=512, res_ratio=0.1),
        View(64*64*3)
    )
    resnet[0].load_state_dict(torch.load(cpkt_path))
    resnet = resnet.to(device)
    freeze(resnet)
    gc.collect(); torch.cuda.empty_cache()
    return resnet

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def _read_images(paths, mode='RGB', verbose=True):
    images = []
    for path in paths:
        try:
            with Image.open(path, 'r') as im:
                images.append(im.convert(mode).copy())
        except:
            if verbose:
                print('Failed to read {}'.format(path))
    return images

class ImagesReader:
    def __init__(self, mode='RGB', verbose=True):
        self.mode = mode
        self.verbose = verbose
        
    def __call__(self, paths):
        return _read_images(paths, mode=self.mode, verbose=self.verbose)

def read_image_folder(path, mode='RGB', verbose=True, n_jobs=1):
    paths = [os.path.join(path, name) for name in os.listdir(path)]
    
    chunk_size = (len(paths) // n_jobs) + 1
    chunks = [paths[x:x+chunk_size] for x in range(0, len(paths), chunk_size)]
    
    pool = multiprocessing.Pool(n_jobs)
    
    chunk_reader = ImagesReader(mode, verbose)
    
    images = list(itertools.chain.from_iterable(
        pool.map(chunk_reader, chunks)
    ))
    pool.close()
    return images

def get_generated_inception_stats(G, Z_sampler, size, batch_size=8, verbose=False):
    freeze(G)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    model.eval()

    if batch_size > size:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = size

    pred_arr = np.empty((size, dims))
    for i in tqdm(range(0, size, batch_size)) if verbose else range(0, size, batch_size):
        start = i
        end = min(i + batch_size, size)

        batch = ((G(Z_sampler.sample(end-start)) + 1) / 2).type(torch.FloatTensor).cuda()
        with torch.no_grad():
            pred = model(batch)[0]

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    
    model = model.cpu()
    del model, pred_arr, pred, batch
    gc.collect()
    torch.cuda.empty_cache()
    
    return mu, sigma

class CompositeGenerator(nn.Module):
    """An auxilary class used to simply computation of FID score
    for the generator composed with the transport map"""
    def __init__(self, G, D):
        super(CompositeGenerator, self).__init__()
        self.G = G
        self.D = D
        
    def forward(self, input):
        G_Z = self.G(input).reshape(-1, 64*64*3)
        return self.D.push_nograd(G_Z).reshape(-1, 3, 64, 64)
    
    
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    

def get_sde_pushed_loader_stats(T, loader, batch_size=8, n_epochs=1, verbose=False,
                              device='cuda',
                              use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = T(
                        X[start:end].type(torch.FloatTensor).to(device)
                    )[0][:, -1].add(1).mul(.5)
                    
                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma


def get_loader_stats(loader, batch_size=8, n_epochs=1, verbose=False, use_Y=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    freeze(model)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, Y) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                                           
                    if not use_Y:
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                    else:
                        batch = ((Y[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                        
                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                        
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma