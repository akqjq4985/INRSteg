import argparse
import pathlib
import numpy as np
import cv2
from tensorboard.backend.event_processing import event_accumulator
import skvideo.datasets
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training_new, loss_functions, modules
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


model_path = "PATH for model"
inr_ckpt = torch.load(model_path)

video_path = skvideo.datasets.bigbuckbunny()

vid_dataset = dataio.Video(video_path)

resolution = vid_dataset.shape

frames = [0, 4, 8, 12]
Nslice = 10

model = modules.SingleBVPNet(type='sine', in_features=3, out_features=vid_dataset.channels,
                            mode='mlp', hidden_features=128, num_hidden_layers=6)
model.load_state_dict(inr_ckpt)


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
print(psnr)
apd = calculate_apd(gt_vid, pred_vid)
print(apd)
pred_vid = pred_vid.permute(0, 3, 1, 2)

gt_vid = gt_vid.permute(0, 3, 1, 2)

output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
# pdb.set_trace()
# output_vs_gt = pred_vid
out = make_grid(output_vs_gt, scale_each=False, normalize=True)
plt.figure()
plt.show()