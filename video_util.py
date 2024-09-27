import argparse
import pathlib
import numpy as np
import cv2
from tensorboard.backend.event_processing import event_accumulator
import skvideo.datasets
import dataio, meta_modules, utils, training, loss_functions, modules
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

def calculate_apd(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    apd = np.mean(np.abs(img1 - img2))
    if apd == 0:
        return float('inf')

    return np.mean(apd)

def calculate_video_psnr(model):
    video_path = skvideo.datasets.bigbuckbunny()
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

    return psnr, apd

    