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
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import skvideo.datasets
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as SNR
import scipy.io.wavfile as wavfile
from functools import partial
import pdb
random.seed(42)

def get_snr(gt, pred):
    snr = SNR()
    return snr(gt, pred)

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
  
def calculate_apd(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    apd = np.mean(np.abs(img1 - img2))
    if apd == 0:
        return float('inf')

    return np.mean(apd)

def visualize(img, path, gt_img):
    img = img.squeeze()
    gt_img = gt_img.squeeze()
    psnr = 10*torch.log10(1 / torch.mean((gt_img - img)**2))
    print(psnr)
    img = img.cpu().detach()
    plt.figure()
    plt.imsave(path, img.permute(1,2,0))
    
def calculate_video_psnr(model):

    video_path = 'PATH for video file'
    vid_dataset = dataio.Video(video_path)

    resolution = vid_dataset.shape

    frames = [0, 4, 8, 12]
    Nslice = 10

    with torch.no_grad():
        coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None,...].cuda() for f in frames]
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
        coords = torch.cat(coords, dim=0)

        output = torch.zeros(coords.shape)
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            model = model.cuda()
            coords = coords.cuda()
            pred = model({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()

    pred_vid = output.view(len(frames), resolution[1], resolution[2], 3) / 2 + 0.5
    pred_vid = torch.clamp(pred_vid, 0, 1)
    gt_vid = torch.from_numpy(vid_dataset.vid[frames, :, :, :])
    psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))
    apd = calculate_apd(gt_vid, pred_vid)
    pred_vid = pred_vid.permute(0, 3, 1, 2)

    gt_vid = gt_vid.permute(0, 3, 1, 2)

    output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
    # pdb.set_trace()
    # output_vs_gt = pred_vid
    out = make_grid(output_vs_gt, scale_each=False, normalize=True)
    plt.figure()
    plt.show()
    plt.close()
    return psnr, apd

def recon(cover, out_type, num_hidden_layers=6):
    # if in_type == 'image':
    #     inr = modules.SingleBVPNet(type='sine', mode='mlp', hidden_features=hidden_size, out_features=3, in_features=2, sidelength=256, num_hidden_layers=num_hidden_layers)
    # elif in_type == 'audio':
    #     inr = modules.SingleBVPNet(type='sine', mode='mlp', out_features=1, hidden_features=hidden_size, in_features=1, num_hidden_layers=num_hidden_layers)
    # elif in_type == 'sdf':
    #     inr = modules.SingleBVPNet(type='sine', mode='mlp', out_features=1, hidden_features=hidden_size, in_features=3, num_hidden_layers=num_hidden_layers)
    # elif in_type == 'video':
    #     inr = modules.SingleBVPNet(type='sine', mode='mlp', out_features=3, hidden_features=hidden_size, in_features=3, num_hidden_layers=num_hidden_layers)
    inr1 = modules.SingleBVPNet(type='sine', mode='mlp', out_features=3, hidden_features=128, in_features=3, num_hidden_layers=num_hidden_layers-1)
    inr2 = modules.SingleBVPNet(type='sine', mode='mlp', out_features=1, hidden_features=128, in_features=1, num_hidden_layers=num_hidden_layers)
    size1 = inr1.net.net[0][0].weight.data.shape[0]
    # pad = int((size1 - hidden_size)/2)
    # f_in = type_dict[out_type][0] - type_dict[in_type][0]
    # f_out = type_dict[out_type][1] - type_dict[in_type][1]
    for i in range(num_hidden_layers+2):
        weight = cover.net.net[i][0].weight.data
        bias = cover.net.net[i][0].bias.data
        if i == 0:
            inr2.net.net[i][0].weight.data = weight[:size1,:1]
            inr2.net.net[i][0].bias.data = bias[:size1]
        elif i == 1:
            inr1.net.net[i-1][0].weight.data = weight[:size1,:3]
            inr1.net.net[i-1][0].bias.data = bias[:size1]
            inr2.net.net[i][0].weight.data = weight[size1:,size1:]
            inr2.net.net[i][0].bias.data = bias[size1:]
        elif i == (num_hidden_layers+1):
            inr1.net.net[i-1][0].weight.data = weight[:3,:size1]
            inr1.net.net[i-1][0].bias.data = bias
            # pdb.set_trace()
            inr2.net.net[i][0].weight.data = weight[:1,size1:]
            inr2.net.net[i][0].bias.data = bias[:1]
            # pdb.set_trace()
        else:
            inr1.net.net[i-1][0].weight.data = weight[:size1,:size1]
            inr1.net.net[i-1][0].bias.data = bias[:size1]
            inr2.net.net[i][0].weight.data = weight[size1:,size1:]
            inr2.net.net[i][0].bias.data = bias[size1:]
    return inr1, inr2


inr_ckpt = torch.load('PATH for cover INR')
inr = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=256, out_features=3, 
                           in_features=2, hidden_features=256, num_hidden_layers=6)
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
print('cover psnr', psnr)


secret_inr1, secret_inr2 = recon(inr, out_type='image')
psnr, apd = calculate_video_psnr(secret_inr1)
print("secret1 psnr", psnr)
print("secret1 apd", apd)


audio_dataset = dataio.AudioFile(filename='PATH for audio file')
coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

secret_inr2.cuda()

model_input, gt = next(iter(dataloader))
model_input = {key: value.cuda() for key, value in model_input.items()}
gt = {key: value.cuda() for key, value in gt.items()}

with torch.no_grad():
    model_output = secret_inr2(model_input)

waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
gt_wf = torch.squeeze(gt['func']).detach().cpu().numpy()
ae = np.abs(gt_wf - waveform)
mae = round(np.mean(ae), 5)
snr = get_snr(torch.Tensor(gt_wf), torch.Tensor(waveform))
print("secret2 mae ", mae)
print("secret2 snr", snr)
wavfile.write('PATH for gt waveform', rate, gt_wf)
wavfile.write('PATH for recon waveform', rate, waveform)
