from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import os.path as osp
import shutil
import random
import time

import torch
import torch.nn as nn
import numpy as np

import src
from src import utils
from src.models.losses import *

from src.data.dataloader import build_val_loader
import pdb

def preprocess_data(images, labels, random_crop, model):
    # images of shape: NxCxHxW
    #print(images.shape)
    if images.ndim == 5:  # 1xNxCxHxW
        images = images.squeeze(0)
        labels = labels.squeeze(0)
    h, w = images.shape[-2:]
    # pdb.set_trace()

    if random_crop:
        ch = random.randint(h * 3 // 4, h)  # h // 2      #256
        cw = random.randint(w * 3 // 4, w)  # square ch   #256

        h0 = random.randint(0, h - ch)  # 128
        w0 = random.randint(0, w - cw)  # 128
    else:
        ch, cw, h0, w0 = h, w, 0, 0


    cw = cw & ~1
    if model ==  'kenet':
        inputs = [
            images[..., h0:h0 + ch, w0:w0 + cw // 2],
            images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
        ]
    elif model == 'xunet':
        inputs = [images[..., h0:h0 + ch, w0:w0 + cw]]


    inputs = [x.cuda() for x in inputs]
    labels = labels.cuda()
    return inputs, labels
    
cover_dir = './images/imagenet_cover/'
stego_dir = './images/imagenet_steg/'

criterion_1 = nn.CrossEntropyLoss()
criterion_2 = ContrastiveLoss(1.00)

val_loader = build_val_loader(cover_dir, stego_dir) ## coverdir, stegodir

net = src.models.KeNet()
# net = src.models.XuNet()
print("kenet")

net = net.cuda()

ckpt = torch.load('./src/kenet.pth.tar')['state_dict']
# ckpt = torch.load('./src/xunet.pth.tar')['state_dict']

net.load_state_dict(ckpt, strict = False)
net.eval()
loss=0
accuracy = 0

alpha  = 0.1
#with torch.no_grad():
#    for data in val_loader:
#            inputs, labels = preprocess_data(data['image'], data['label'], False, 'kenet')
#            #print(labels)
#            
#            outputs, _,_= net(*inputs)
#            #print(outputs)
#
##            loss += criterion_1(outputs, labels).item() + \
##                          alpha * criterion_2(feats_0, feats_1, labels)
#
#            accuracy += src.models.accuracy(outputs, labels).item()
#            #print(src.models.accuracy(outputs, labels).item())
#
#
#    #loss /= len(val_loader)
#    accuracy /= len(val_loader)
#    #print('loss:', loss)
#    print('accuracy:', accuracy)

with torch.no_grad():
    for data in val_loader:
            inputs, labels = preprocess_data(data['image'], data['label'], False, 'kenet')

            # outputs = net(inputs[0])
            outputs, feats_0, feats_1 = net(*inputs)

            accuracy += src.models.accuracy(outputs, labels).item()


    accuracy /= len(val_loader)

    print('accuracy:', accuracy)



