import pandas as pd
import numpy as np
import os
from datetime import datetime

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()

    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)

    return values


def clip(tensor, min_value=None, max_value=None):
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    elif isinstance(tensor, torch.Tensor):
        return torch.clamp(tensor, min_value, max_value)

    raise ValueError("Tensor format is not valid is not valid - " \
        f"should be numpy array or torch tensor. Got {type(tensor)}.")

def m(v):
    nonnull = v[np.isnan(v) == False]
    return np.mean(nonnull)

def std(v):
    nonnull = v[np.isnan(v) == False]
    return np.std(nonnull)

def infill_null(v):
    v[np.isnan(v)] = 0
    return v

def remove_outliers(lists):
    b = np.ones(lists[0].shape)

    for l in lists:
        q1 = np.nanquantile(l,0.25)
        q3 = np.nanquantile(l,0.75)

        iqr = q3 - q1
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr

        b = np.logical_and(b, np.logical_or(l > lower , np.isnan(l)))
        b = np.logical_and(b, np.logical_or(l < upper , np.isnan(l)))
    
    return b

def norm(v):
    nonnull = v[np.isnan(v) == False]
    max = np.nanmax(nonnull)
    min = np.nanmin(nonnull)

    return 2*((v - min) / (max - min))-1

def get_local_gaussian(ys, numbins=50):
    max = np.nanmax(ys)
    min = np.nanmin(ys)
    bins = np.linspace(min, max, retstep=numbins)[0]
    s_ys = np.array(sorted(ys, reverse=True))

    d, m, s = [], [], []
    for i in range(numbins-1):
        low = bins[i]
        high = bins[i+1]
        tbool = np.logical_and(low<=s_ys, s_ys<=high)
        data = s_ys[tbool]
        d.append(len(data))
        m.append(data.mean() if not np.isnan(data.mean()) else low)
        s.append(data.std() if not np.isnan(data.std()) else 0.01)
    d = np.array(d) / sum(d)

    return d, np.array(m), np.array(s)

def sample_local_gaussian(v, noise_type=None, numbins=50):
    d,m,s = get_local_gaussian(v, numbins=numbins)
    num = sum(np.isnan(v))

    samples = np.random.choice(numbins-1, num, p=d)
    rand_n = np.random.randn(num)

    adjust = m[samples] + 1.2 * rand_n *s[samples]
    
    # override 
    #adjust = np.zeros(adjust.shape)
    
    if noise_type == "zeros":
        adjust = np.zeros(adjust.shape)

    if noise_type == "uniform":
        adjust = np.random.uniform(low=-1.0, high=1.0, size=adjust.shape)

    if noise_type == "gaussian":
        adjust = np.random.normal(size=adjust.shape)

    v[np.isnan(v)] = adjust
    return v, (d,m,s)

def sample_noise(b, dmss, noise_type=None, numbins=50):
    if noise_type == "uniform":
        return np.random.uniform(low=-1.0, high=1.0, size=(b,12))
    
    if noise_type == "gaussian":
        return np.random.normal(size=(b,12))

    vs = []
    for d,m,s in dmss:
        samples = np.random.choice(numbins-1, b, p=d)
        rand_n = np.random.randn(b)
        vs.append(m[samples] + 1.2 * rand_n *s[samples])
    return np.stack(vs, axis=-1)