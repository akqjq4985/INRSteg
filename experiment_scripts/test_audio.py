# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, modules
import pdb
from torch.utils.data import DataLoader
import configargparse
import torch
import scipy.io.wavfile as wavfile
import numpy as np
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as SNR


type_dict = {'image': (2,3), 'audio': (1,1), 'sdf': (3,1), 'video': (3,3)}


def get_snr(gt, pred):
    snr = SNR()
    return snr(gt, pred)

def get_mae(gt, pred):
    mae = torch.mean(torch.abs(gt - pred))
    return mae

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


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
# p.add_argument('--experiment_name', type=str, default='audio',
#                help='Name of subdirectory in logging_root where wav file will be saved.')
p.add_argument('--gt_wav_path', type=str, default='../data/pop.00000.wav', help='ground truth wav path')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')
p.add_argument('--checkpoint_path', help='Checkpoint to trained model.')

opt = p.parse_args()

audio_dataset = dataio.AudioFile(filename=opt.gt_wav_path)
coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model and load in checkpoint path
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', in_features=1, hidden_features=256, num_hidden_layers=6)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, fn_samples=len(audio_dataset.data), in_features=1)
else:
    raise NotImplementedError
inr_ckpt = torch.load(opt.checkpoint_path)
inr_ckpt = cut_ckpt(inr_ckpt, in_type="video", out_type="audio")
model.load_state_dict(inr_ckpt)
model.cuda()

# root_path = os.path.join(opt.logging_root, opt.experiment_name)
# utils.cond_mkdir(root_path)

# Get ground truth and input data
model_input, gt = next(iter(dataloader))
model_input = {key: value.cuda() for key, value in model_input.items()}
gt = {key: value.cuda() for key, value in gt.items()}

# Evaluate the trained model
with torch.no_grad():
    model_output = model(model_input)

waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
gt_wf = torch.squeeze(gt['func']).detach().cpu().numpy()
ae = np.abs(gt_wf - waveform)
mae = round(np.mean(ae), 5)
# pdb.set_trace()
snr = get_snr(torch.Tensor(gt_wf), torch.Tensor(waveform))
# psnr = 10*torch.log10(1 / torch.mean((torch.Tensor(gt_wf) - torch.Tensor(waveform))**2))
print("mae ", mae)
print("snr", snr)
wavfile.write('PATH for gt waveform', rate, gt_wf)
wavfile.write('PATH for recon waveform', rate, waveform)