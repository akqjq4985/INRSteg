'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import pdb
import matplotlib.pyplot as plt
import modules
import dataio, meta_modules, utils, training, loss_functions, modules
from copy import deepcopy
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as SNR
from video_util import calculate_video_psnr

type_dict = {'image': (2,3), 'audio': (1,1), 'sdf': (3,1), 'video': (3,3)}

def calculate_rmse(img1, img2):
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    mse = torch.mean((img1 - img2)**2)
    return torch.sqrt(mse)

def get_snr(gt, pred):
    snr = SNR()
    return snr(gt, pred)

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, type=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        best_psnr = -10.0
        stop = 0
        epochs_list = []
        total_psnr = []
        total_train_loss = []
        for epoch in range(epochs):
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        single_loss *= loss_schedules[loss_name](total_steps)
                    train_loss += single_loss

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()
                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1
        #     if type == 'audio':
        #         waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
        #         gt_wf = torch.squeeze(gt['func']).detach().cpu().numpy()
        #         psnr = get_snr(torch.Tensor(gt_wf), torch.Tensor(waveform))

        #         if (psnr - best_psnr) > 0.00001:
        #             best_psnr = psnr
        #             best_model = deepcopy(model.state_dict())
        #             ae = np.abs(gt_wf - waveform)
        #             second_metric = round(np.mean(ae), 4)
        #             stop = 0
        #         else:
        #             stop += 1
        #         if stop > 1000:
        #             break
        #     elif type == 'video':
        #         psnr = 10*torch.log10(1 / torch.mean((gt['img'] - model_output['model_out'])**2))
        #         if (psnr - best_psnr) > 0.00001:
        #             best_psnr = psnr
        #             best_model = deepcopy(model.state_dict())
        #             # try:
        #             #     second_metric = round(apd.item(), 4)
        #             # except:
        #             #     second_metric = round(apd, 4)
        #             stop = 0
        #         else:
        #             stop += 1
        #         if stop > 5000:
        #             break
        #     elif type == 'image':
        #         psnr = 10*torch.log10(1 / torch.mean((gt['img'] - model_output['model_out'])**2))
        #         if (psnr - best_psnr) > 0.00001:
        #             best_psnr = psnr
        #             best_model = deepcopy(model.state_dict())
        #             # best_optim = deepcopy(optim.state_dict())
        #             # second_metric = calculate_rmse(gt['img'], model_output['model_out'])
        #             # try:
        #             #     second_metric = round(second_metric, 4)
        #             # except:
        #             #     second_metric = round(second_metric.item(), 4)
        #             stop = 0
        #         else:
        #             stop += 1
        #         if stop > 10000:
        #             break 
        #     elif type == 'sdf':
        #         if epoch == 15000:
        #             best_model = deepcopy(model.state_dict())
        #             break
        #     if not epoch % steps_til_summary:
        #         epochs_list.append(epoch)
        #         total_psnr.append(psnr)
        #         total_train_loss.append(train_loss.item())
        # plt.figure(figsize=(10,5))
        # plt.subplot(1,2,1)
        # plt.xlabel("Epoch")
        # plt.ylabel("Train Loss")
        # plt.plot(epochs_list, total_train_loss)
        # plt.subplot(1,2,2)
        # plt.xlabel("Epoch")
        # plt.ylabel("PSNR")
        # plt.plot(epochs_list, total_psnr)
        # plt.savefig("./images/time_video_cover_256_6.png")
            # if type == 'audio':
            #     waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
            #     gt_wf = torch.squeeze(gt['func']).detach().cpu().numpy()
            #     psnr = 10*torch.log10(1 / torch.mean((gt_wf - waveform)**2))
            # else:
            #     psnr = 10*torch.log10(1 / torch.mean((gt['img'] - model_output['model_out'])**2))
            # if (psnr - best_psnr) >= 0.00001:
            #     best_psnr = psnr
            #     stop = 0
            # else:
            #     stop += 1
            # if stop > 2000:
            #     break 
        # torch.save(best_model, os.path.join(checkpoints_dir, 'model_final.pth'))
        # return best_model
        # torch.save(best_model,
        #            os.path.join(checkpoints_dir, 'model_final.pth'))
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #            np.array(train_losses))
        
def recon(cover, hidden_size, in_type, out_type, num_hidden_layers=6):
    if in_type == 'image':
        inr = modules.SingleBVPNet(type='sine', mode='mlp', hidden_features=hidden_size, out_features=3, in_features=2, sidelength=256, num_hidden_layers=num_hidden_layers)
    elif in_type == 'audio':
        inr = modules.SingleBVPNet(type='sine', mode='mlp', out_features=1, hidden_features=hidden_size, in_features=1, num_hidden_layers=num_hidden_layers)
    elif in_type == 'sdf':
        inr = modules.SingleBVPNet(type='sine', mode='mlp', out_features=1, hidden_features=hidden_size, in_features=3, num_hidden_layers=num_hidden_layers)
    elif in_type == 'video':
        inr = modules.SingleBVPNet(type='sine', mode='mlp', out_features=3, hidden_features=hidden_size, in_features=3, num_hidden_layers=num_hidden_layers)
    size1 = cover.net.net[0][0].weight.data.shape[0]
    pad = int((size1 - hidden_size)/2)
    f_in = type_dict[out_type][0] - type_dict[in_type][0]
    f_out = type_dict[out_type][1] - type_dict[in_type][1]
    for i in range(num_hidden_layers+2):
        weight = cover.net.net[i][0].weight.data
        if i == 0:
            if f_in > 0 :
                weight = weight[pad:-pad, :type_dict[in_type][0]]
            else:
                weight = weight[pad:-pad, :]
        elif i == (num_hidden_layers+1):
            if f_out > 0 : 
                weight = weight[:type_dict[in_type][1], pad:-pad]
            else:
                weight = weight[:, pad:-pad]
        else:
            weight = weight[pad:-pad, pad:-pad]

        inr.net.net[i][0].weight.data = weight
        bias = cover.net.net[i][0].bias.data
        if i != (num_hidden_layers+1):
            bias = bias[pad:-pad]
        else:
            if f_out > 0:
                bias = bias[:type_dict[in_type][1]]
            else:
                bias = bias
        inr.net.net[i][0].bias.data = bias
    return inr

def freeze_train(model, num_hidden_data, size1, size2, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, num_hidden_layers=6, out_partial = False, 
          in_type = None, out_type=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    f_in = type_dict[out_type][0] - type_dict[in_type][0]
    f_out = type_dict[out_type][1] - type_dict[in_type][1]

    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)

    # checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    # utils.cond_mkdir(checkpoints_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        total_train_loss = []
        total_psnr = []
        epochs_list = []
        best_psnr = -10.0
        stop = 0
        for epoch in range(epochs):
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                model_output = model(model_input)
                
                if f_out < 0 :
                    model_output['model_out'] = model_output['model_out'][:, :, :type_dict[out_type][1]]

                
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        # writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    # writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                # writer.add_scalar("total_train_loss", train_loss, total_steps)
                if not total_steps % steps_til_summary:
                    total_train_loss.append(train_loss.item())
                    # psnr = calculate_psnr(model, out_type)
                    # pdb.set_trace()
                    if out_type == 'audio':
                        waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
                        gt_wf = torch.squeeze(gt['func']).detach().cpu().numpy()
                        psnr = get_snr(torch.Tensor(gt_wf), torch.Tensor(waveform))
                    elif out_type == 'sdf':
                        psnr = 0.0
                    else:
                        psnr = 10*torch.log10(1 / torch.mean((gt['img'] - model_output['model_out'])**2))
                    total_psnr.append(psnr)
                    epochs_list.append(epoch)

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    
                    if num_hidden_data == 2: ##size1, size2 are the number of hidden features for each hidden data        
                        for name, param in model.named_parameters():
                            if 'weight' in name:  
                                if '0.0' not in name and str(num_hidden_layers) not in name:
                                    #print(name, param.shape)
                                    param.grad[:size1,:size1] = 0
                                    param.grad[size1:, size1:] = 0
                                elif '0.0' in name or str(num_hidden_layers) in name:
                                    param.grad[:,:] = 0
                            elif 'bias' in name:
                                param.grad[:] = 0
                    elif num_hidden_data == 1: ##size1 is the number of hidden features for the hidden data, size2 is the padded size/2
                        pad = int((size2 - size1)/2)
                        for name, param in model.named_parameters():
                            if 'weight' in name:  
                                if '0.0' not in name and str(num_hidden_layers+1) not in name:
                                    #print(name, param.shape)
                                    param.grad[pad:-pad, pad:-pad] = 0
                                elif '0.0' in name:
                                    if f_in > 0 :
                                        param.grad[pad:-pad, :type_dict[in_type][0]] = 0
                                    else:
                                        param.grad[pad:-pad, :] = 0
                                elif str(int(num_hidden_layers)+1) in name:
                                    if f_out > 0 :
                                        param.grad[:type_dict[in_type][1], pad:-pad] = 0
                                    else:
                                        param.grad[:, pad:-pad] = 0
                            elif 'bias' in name:
                                if str(int(num_hidden_layers)+1) not in name:
                                    param.grad[pad:-pad] = 0
                                else:
                                    if f_out > 0:
                                        param.grad[:type_dict[in_type][1]] = 0
                                    else:
                                        param.grad[:] = 0
                    optim.step()
                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1
            # if out_type == 'audio':
            #     waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
            #     gt_wf = torch.squeeze(gt['func']).detach().cpu().numpy()
            #     psnr = get_snr(torch.Tensor(gt_wf), torch.Tensor(waveform))

            #     if (psnr - best_psnr) > 0.00001:
            #         best_psnr = psnr
            #         best_model = deepcopy(model.state_dict())
            #         ae = np.abs(gt_wf - waveform)
            #         second_metric = round(np.mean(ae), 4)
            #         stop = 0
            #     else:
            #         stop += 1
            #     if stop > 5000:
            #         break
            # elif out_type == 'video':
            #     psnr = 10*torch.log10(1 / torch.mean((gt['img'] - model_output['model_out'])**2))
            #     if (psnr - best_psnr) > 0.00001:
            #         best_psnr = psnr
            #         best_model = deepcopy(model.state_dict())
            #         # try:
            #         #     second_metric = round(apd.item(), 4)
            #         # except:
            #         #     second_metric = round(apd, 4)
            #         stop = 0
            #     else:
            #         stop += 1
            #     if stop > 5000:
            #         break
            # elif out_type == 'image':
            #     psnr = 10*torch.log10(1 / torch.mean((gt['img'] - model_output['model_out'])**2))
            #     if (psnr - best_psnr) > 0.00001:
            #         best_psnr = psnr
            #         best_model = deepcopy(model.state_dict())
            #         # best_optim = deepcopy(optim.state_dict())
            #         second_metric = calculate_rmse(gt['img'], model_output['model_out'])
            #         try:
            #             second_metric = round(second_metric, 4)
            #         except:
            #             second_metric = round(second_metric.item(), 4)
            #         stop = 0
            #     else:
            #         stop += 1
            #     if stop > 10000:
            #         break 
            # elif out_type == 'sdf':
            #     if epoch == 15000:
            #         best_model = deepcopy(model.state_dict())
            #         break
            # pdb.set_trace()
        # print(best_psnr)
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.plot(epochs_list, total_train_loss)
        plt.subplot(1,2,2)
        plt.xlabel("Epoch")
        plt.ylabel("PSNR")
        plt.plot(epochs_list, total_psnr)
        plt.show()
        # torch.save(best_model, os.path.join(checkpoints_dir, 'model_final.pth'))
        # torch.save({'epoch': epoch, 'model': best_model, 'optimizer': best_optim},
        #            os.path.join(checkpoints_dir, 'model_final.pth'))
        # torch.save(model.state_dict(),
        #            os.path.join(checkpoints_dir, 'model_final.pth'))
        # return best_model