# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
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
import torch.optim as optim
import numpy as np
from einops import rearrange
from tqdm import tqdm
import os
import json
import timeit
import csv
from torch.utils.data import DataLoader
import configargparse
from functools import partial
random.seed(42)

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')
p.add_argument('--num_hidden_data', type=int, default=2, help='Number of hidden data', choices=[1,2])
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

def inr2img(inr, res=256):
    """inr net to image"""
    mgrid = dataio.get_mgrid(res)
    coord_dataset = {'idx': 0, 'coords': mgrid}

    with torch.no_grad():
        model_output = inr.forward_with_activations(coord_dataset)
    out = model_output['model_out'][1]
    out = rearrange(out, '(h w) c -> () c h w', h=res, w=res)
    return out

def visualize(img, path):
    img = img.squeeze()
    img = img.cpu().detach()
    plt.figure()
    plt.imsave(path, img.permute(1,2,0))

def insert_inr(inr, secret_ckpt, hidden_features=256, num_hidden_layers=4):
    # inr.net
    pad = int(hidden_features/2)
    for i in range(num_hidden_layers+1):
        # print(i)
        org_weight = inr.net.net[i][0].weight.data
        secret_weight = secret_ckpt[f'net.net.{i}.0.weight']
        if i == 0:
            m = torch.nn.ZeroPad2d((0,0,pad,pad))
            org_weight[pad:-pad, :] = 0
        elif i == (num_hidden_layers+1):
            m = torch.nn.ZeroPad2d((pad,pad,0,0))
            org_weight[:, pad:-pad] = 0
        else:
            m = torch.nn.ZeroPad2d((pad,pad,pad,pad))
            org_weight[pad:-pad, pad:-pad] = 0
        # print(secret_weight.shape)
        secret_weight = m(secret_weight)
        
        # print(secret_weight.shape)
        new_weight = org_weight.cuda() + secret_weight.cuda()
        inr.net.net[i][0].weight.data = new_weight
        
        org_bias = inr.net.net[i][0].bias.data
        secret_bias = secret_ckpt[f'net.net.{i}.0.bias']
        if i != (num_hidden_layers+1):
            m = torch.nn.ConstantPad1d((pad,pad),0)
            org_bias[pad:-pad] = 0
            secret_bias = m(secret_bias)
            new_bias = org_bias.cuda() + secret_bias.cuda()
        else:
            new_bias = secret_bias.cuda()
        inr.net.net[i][0].bias.data = new_bias
    return inr

secret_ckpt = torch.load('PATH for secret model')

img_dataset = dataio.ImageFile('PATH for cover image')
# print(img_dataset.shape)
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=256)

image_resolution = (256, 256)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# print(inr_ckpt)
inr = modules.SingleBVPNet(type='sine', mode='mlp', hidden_features=512, out_features=3)
model = insert_inr(inr, secret_ckpt)

model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_image_summary, image_resolution)

training.freeze_train(model=model, num_hidden_data=opt.num_hidden_data, size1=128, size2=128, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)

# inr.load_state_dict(inr_ckpt)

# org_img1 = inr2img(inr, 256)
# visualize(org_img1, './test2.png')

# reconstruction
# inr_ckpt = torch.load('PATH for trained model')
# print(inr_ckpt)
# inr = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=256, out_features=3)
# inr.load_state_dict(inr_ckpt)

# org_img1 = inr2img(inr, 256)
# visualize(org_img1, './test2.png')
