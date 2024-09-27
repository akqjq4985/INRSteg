# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import torch
import dataio, meta_modules, utils, training_new, loss_functions, modules
import copy
from torch.utils.data import DataLoader
import configargparse
from functools import partial
import pdb
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.') # 100000

p.add_argument('--epochs_til_ckpt', type=int, default=10000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')
p.add_argument('--num_hidden_data', type=int, default=2, help='Number of hidden data', choices=[1,2])
p.add_argument('--num_hidden_layers', type=int, default=6, help='Number of hidden layers')
p.add_argument('--checkpoint_path1', help='Checkpoint to hidden data1.')
p.add_argument('--checkpoint_path2', help='Checkpoint to hidden data2.')
opt = p.parse_args()


def concat_inr(ckpt1, ckpt2):
    concat_ckpt = copy.deepcopy(ckpt2)
    v1_hidden_layers = opt.num_hidden_layers
    v2_hidden_layers = opt.num_hidden_layers+1
    
    for i in range(1):
        v2_weight = ckpt2[f"net.net.{i}.0.weight"]
        concat_ckpt[f"net.net.{i}.0.weight"] = torch.zeros((256,2)).cuda()
        concat_ckpt[f"net.net.{i}.0.weight"][:v2_weight.size(0), :v2_weight.size(1)] = v2_weight
        v2_bias = ckpt2[f"net.net.{i}.0.bias"]
        zero_tensor = torch.zeros(256-v2_bias.size(0)).cuda()
        concat_ckpt[f"net.net.{i}.0.bias"] = torch.cat((v2_bias, zero_tensor))
    
    for i in range(1, opt.num_hidden_layers+1):
        v1_weight = ckpt1.get(f"net.net.{i-1}.0.weight", torch.empty(0))
        v1_bias = ckpt1.get(f"net.net.{i-1}.0.bias", torch.empty(0))
        if i == 1:
            zero_tensor = torch.zeros((v1_weight.size(0), v1_weight.size(0)-3)).cuda()
            v1_weight = torch.cat((v1_weight, zero_tensor), dim=1)
            
        v2_weight = ckpt2.get(f"net.net.{i}.0.weight", torch.empty(0))    
        v2_bias = ckpt2.get(f"net.net.{i}.0.bias", torch.empty(0))
        v0 = torch.zeros((v1_weight.size()[0], v2_weight.size()[0])).cuda()
        v00 = torch.zeros((v2_weight.size()[0], v1_weight.size()[0])).cuda()
        concat_ckpt[f"net.net.{i}.0.weight"] = torch.cat((torch.cat((v1_weight, v0), dim=1), torch.cat((v00,v2_weight), dim=1)))
        concat_ckpt[f"net.net.{i}.0.bias"] = torch.cat((v1_bias, v2_bias))

    for i in range(opt.num_hidden_layers+1, opt.num_hidden_layers+2):
        v1_weight = ckpt1.get(f"net.net.{i-1}.0.weight", torch.empty(0))
        v1_bias = ckpt1.get(f"net.net.{i-1}.0.bias", torch.empty(0))
        v2_weight = ckpt2.get(f"net.net.{i}.0.weight", torch.empty(0))    
        v2_bias = ckpt2.get(f"net.net.{i}.0.bias", torch.empty(0))
        zero_tensor = torch.zeros((v1_weight.size(0)-1, v2_weight.size(1))).cuda()
        v2_weight = torch.cat((v2_weight, zero_tensor), dim=0)
        concat_ckpt[f"net.net.{i}.0.weight"] = torch.cat((v1_weight, v2_weight), dim=1)
        concat_ckpt[f"net.net.{i}.0.bias"] = v1_bias
    return concat_ckpt

img_dataset = dataio.ImageFile(f'PATH for image file')
# print(img_dataset.shape)
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=256)

image_resolution = (256, 256)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
        or opt.model_type == 'softplus':
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', 
                                sidelength=image_resolution, out_features=3, hidden_features=256, 
                                num_hidden_layers=6)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=image_resolution)
else:
    raise NotImplementedError
model.cuda()
ckpt1 = torch.load(opt.checkpoint_path1)
ckpt2 = torch.load(opt.checkpoint_path2)

new_ckpt = concat_inr(ckpt1, ckpt2)
model.load_state_dict(new_ckpt)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_image_summary, image_resolution)

training_new.freeze_train(model=model, num_hidden_data=opt.num_hidden_data, size1=128, size2=128, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, num_hidden_layers=opt.num_hidden_layers, out_type='image')