# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import glob
import dataio, meta_modules, utils, training, loss_functions, modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import random
import copy
import torch
import torch.nn as nn
import copy
import cv2
from pytorch_ssim import SSIM
import torch.optim as optim
import numpy as np
from einops import rearrange
from tqdm import tqdm
import os
import json
import timeit
import csv
from torch.utils.data import DataLoader
# import configargparse
from functools import partial
random.seed(42)


type_dict = {'image': (2,3), 'audio': (1,1), 'sdf': (3,1), 'video': (3,3)}


def calculate_rmse(img1, img2):
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    mse = torch.mean((img1 - img2)**2)
    return torch.sqrt(mse)

def calculate_mae(img1, img2):
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    ae = np.abs(img1 - img2)
    mae = round(np.mean(ae), 4)
    return mae

def calculate_apd(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    apd = np.mean(np.abs(img1 - img2))
    if apd == 0:
        return float('inf')

    return np.mean(apd)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def get_ssim(gt, pred):
    #print(gt.shape)
    #print(pred.shape)
    gt = gt.unsqueeze(dim=0)
    pred = pred.unsqueeze(dim=0)
    ssim_loss = SSIM(window_size = 11)
    return ssim_loss(gt, pred)

def ssim(img1, img2):
    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def inr2img(inr, res=256):
    """inr net to image"""
    mgrid = dataio.get_mgrid(res)
    # m = torch.nn.ZeroPad2d((0,1,0,0))
    # mgrid = m(mgrid)
    coord_dataset = {'idx': 0, 'coords': mgrid}

    with torch.no_grad():
        model_output = inr.forward_with_activations(coord_dataset)
    out = model_output['model_out'][1]
    out = rearrange(out, '(h w) c -> () c h w', h=res, w=res)
    return out

def visualize(img, path, gt_img):
    img = img.squeeze()
    gt_img = gt_img.squeeze()
    psnr = 10*torch.log10(1 / torch.mean((gt_img - img)**2))
    print(psnr)
    img = img.cpu().detach()
    plt.figure()
    plt.imsave(path, img.permute(1,2,0))
    
def cut_ckpt(ckpt, in_type, out_type, num_hidden_layers=6): ##size1 is the number of hidden features for the hidden data, size2 is the padded size/2
    f_in = type_dict[out_type][0] - type_dict[in_type][0]
    f_out = type_dict[out_type][1] - type_dict[in_type][1]
    for i in range(num_hidden_layers+2):
        if i == (num_hidden_layers+1):
            if f_out < 0 :
                weight = ckpt[f'net.net.{i}.0.weight']
                ckpt[f'net.net.{i}.0.weight'] = weight[:type_dict[out_type][1], :]
        elif i == 0:
            if f_in < 0 :
                weight = ckpt[f'net.net.{i}.0.weight']
                ckpt[f'net.net.{i}.0.weight'] = weight[:, :type_dict[out_type][0]]
        if i == (num_hidden_layers+1):
            if f_out < 0:
                bias = ckpt[f'net.net.{i}.0.bias']
                ckpt[f'net.net.{i}.0.bias'] = bias[:type_dict[out_type][1]]
    return ckpt


def recon(cover, hidden_size, in_type='image', out_type='image', num_hidden_layers=6):
    inr = modules.SingleBVPNet(type='sine', mode='mlp', hidden_features=hidden_size, out_features=3, in_features=2, sidelength=256, num_hidden_layers=num_hidden_layers)
    size1 = cover.net.net[0][0].weight.data.shape[0]
    pad = int((size1 - hidden_size)/2)
    for i in range(num_hidden_layers+2):
        weight = cover.net.net[i][0].weight.data
        if i == 0:
            weight = weight[pad:-pad, :]
        elif i == (num_hidden_layers+1):
            weight = weight[:, pad:-pad]
        else:
            weight = weight[pad:-pad, pad:-pad]

        inr.net.net[i][0].weight.data = weight
        bias = cover.net.net[i][0].bias.data
        if i != (num_hidden_layers+1):
            bias = bias[pad:-pad]
        inr.net.net[i][0].bias.data = bias
    return inr


inr_ckpt = torch.load('PATH for cover INR')
inr_ckpt = cut_ckpt(inr_ckpt, in_type="sdf", out_type="image")
inr = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=256, out_features=3, 
                           in_features=2, hidden_features=576, num_hidden_layers=6)
inr.load_state_dict(inr_ckpt)
img_dataset = dataio.ImageFile('PATH for cover image')
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=256)
_, gt_dict = coord_dataset[0]
gt_img = gt_dict['img']
gt_img = rearrange(gt_img, '(h w) c -> () c h w', h=256, w=256)
gt_img = gt_img.squeeze()
plt.figure()
plt.show()
plt.close()
org_img1 = inr2img(inr, 256)
img = org_img1.squeeze()
plt.figure()
plt.show()
plt.close()
psnr = 10*torch.log10(1 / torch.mean((gt_img - img)**2))

rmse = calculate_rmse(gt_img, img)
ssimv = get_ssim(gt_img, img)
mae = calculate_mae(gt_img, img)
print("psnr ", psnr)
print("rmse ", rmse)
print("ssim ", ssimv)
print("mae ", mae)

print("cover PSNR: ", psnr)