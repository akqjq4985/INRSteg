'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
# p.add_argument('--experiment_name', type=str, required=True,
#                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=5000) # 16384
# p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=512)

opt = p.parse_args()

type_dict = {'image': (2,3), 'audio': (1,1), 'sdf': (3,1), 'video': (3,3)}

def cut_ckpt(ckpt, in_type, out_type, num_hidden_layers=6): ##size1 is the number of hidden features for the hidden data, size2 is the padded size/2
    f_out = type_dict[out_type][1] - type_dict[in_type][1]
    for i in range(num_hidden_layers+2):
        if i == (num_hidden_layers+1):
            if f_out < 0 :
                weight = ckpt[f'net.net.{i}.0.weight']
                ckpt[f'net.net.{i}.0.weight'] = weight[:type_dict[out_type][1], :]

        if i == (num_hidden_layers+1):
            if f_out < 0:
                bias = ckpt[f'net.net.{i}.0.bias']
                ckpt[f'net.net.{i}.0.bias'] = bias[:type_dict[out_type][1]]
    return ckpt

class SDFDecoder(torch.nn.Module):
    def __init__(self, hidden_features, in_type, out_type, checkpoint_path):
        super().__init__()
        # Define the model.
        if opt.mode == 'mlp':
            self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=3,
                                              hidden_features=hidden_features, num_hidden_layers=6)
        elif opt.mode == 'nerf':
            self.model = modules.SingleBVPNet(type='relu', mode='nerf', final_layer_factor=1, in_features=3)
        ckpt = torch.load(checkpoint_path)
        ckpt = cut_ckpt(ckpt, in_type=in_type, out_type=out_type)
        self.model.load_state_dict(ckpt)
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']


path = 'PATH for checkpoint'
sdf_decoder = SDFDecoder(hidden_features=256, in_type='image', out_type='sdf', checkpoint_path=path)

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, 'test'), N=opt.resolution)