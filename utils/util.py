import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from random import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import scipy.misc
import cv2
import yaml
from torch.optim import lr_scheduler
import torch.nn.init as init
import random
from torchvision.utils import save_image, make_grid
import math
from torchvision import transforms as T
import seaborn as sns
sns.set(color_codes=True)
from torchvision.utils import save_image

def show_output2(x_A, x_B_arr, x_AB_arr):

    batch_size = x_A.size(0)
    x_A = var_to_numpy(x_A)

    for i in range(len(x_B_arr)):
        x_B_arr[i] = var_to_numpy(x_B_arr[i])

    for i in range(len(x_AB_arr)):
        x_AB_arr[i] = var_to_numpy(x_AB_arr[i])

    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20,12), gridspec_kw = {'wspace':0, 'hspace':0})
    
    for i in range(5):
        axs[0][i].imshow(x_A[0])
        axs[0][i].axis('off')
        axs[0][i].set_aspect('equal')
        axs[1][i].imshow(x_B_arr[i][0])
        axs[1][i].axis('off')
        axs[1][i].set_aspect('equal')
        axs[2][i].imshow(x_AB_arr[i][0])
        axs[2][i].axis('off')
        axs[2][i].set_aspect('equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def save_output2(x_A, m_A_arr, x_B_arr, m_B_arr, x_AB_arr, idx, name):

    def convert_rgb_to_gray(rgb): # 1, 3, 128, 128
            rgb = T.ToPILImage()(denorm(rgb[0].data.cpu())) # 3, 128, 128
            gray = T.functional.to_grayscale(rgb)
            return T.ToTensor()(gray).to(2)

    def make_rgb_transparent(rgb, mask, alpha):
        return [mask + (alpha) * c1
                for c1 in rgb]

    col_concat_row = []

    for i in range(len(x_B_arr)):
        if i != 1:
            continue

        x_gray_A = convert_rgb_to_gray(x_A) # [1, 128, 128]
        x_gray_B = convert_rgb_to_gray(x_B_arr[i]) # [1, 128, 128]

        masked_A = make_rgb_transparent(x_gray_A, m_A_arr[i], 0.15)
        masked_B = make_rgb_transparent(x_gray_B, m_B_arr[i], 0.15)

        masked_A = torch.cat(masked_A, dim=0) # [1, 1, 128, 128]
        masked_B = torch.cat(masked_B, dim=0) # [1, 1, 128, 128]

        masked_A = masked_A[0].repeat(3,1,1) # [3, 128, 128]
        masked_B = masked_B[0].repeat(3,1,1) # [3, 128, 128]

        x_concat_col = torch.cat([denorm(x_A[0]), masked_A, denorm(x_B_arr[i][0]), masked_B, denorm(x_AB_arr[i][0])], dim=2)
        col_concat_row += [x_concat_col]

    col_concat_row = torch.cat(col_concat_row, dim=2)
    sample_path = 'Fig4_%d_%s.jpg' % (idx, name)
    save_image(col_concat_row.data.cpu(), sample_path, nrow=1, padding=0)

def save_output(x_A, m_A_arr, x_B_arr, m_B_arr, x_AB_arr, idx, name):

    def convert_rgb_to_gray(rgb): # 1, 3, 128, 128
            rgb = T.ToPILImage()(denorm(rgb[0].data.cpu())) # 3, 128, 128
            gray = T.functional.to_grayscale(rgb)
            return T.ToTensor()(gray).to(2)

    def make_rgb_transparent(rgb, mask, alpha):
        return [mask + (alpha) * c1
                for c1 in rgb]

    col_concat_row = []

    for i in range(len(x_B_arr)):

        x_gray_A = convert_rgb_to_gray(x_A) # [1, 128, 128]
        x_gray_B = convert_rgb_to_gray(x_B_arr[i]) # [1, 128, 128]

        masked_A = make_rgb_transparent(x_gray_A, m_A_arr[i], 0.15)
        masked_B = make_rgb_transparent(x_gray_B, m_B_arr[i], 0.15)

        masked_A = torch.cat(masked_A, dim=0) # [1, 1, 128, 128]
        masked_B = torch.cat(masked_B, dim=0) # [1, 1, 128, 128]

        masked_A = masked_A[0].repeat(3,1,1) # [3, 128, 128]
        masked_B = masked_B[0].repeat(3,1,1) # [3, 128, 128]

        x_concat_col = torch.cat([denorm(x_A[0]), masked_A, denorm(x_B_arr[i][0]), masked_B, denorm(x_AB_arr[i][0])], dim=1)
        col_concat_row += [x_concat_col]

    col_concat_row = torch.cat(col_concat_row, dim=2)
    sample_path = 'Fig4_%d_%s.jpg' % (idx, name)
    save_image(col_concat_row.data.cpu(), sample_path, nrow=1, padding=0)
            
def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def show_output(x_A, x_B, x_AB, x_BA, m_A, m_B, x_fA, x_fB, x_bA, x_bB, label_A, label_B,
                attrs, num_print):

    def convert_rgb_to_gray(rgb): # B, 3, 128, 128
        rgb = T.ToPILImage()(denorm(rgb.data.cpu())) # 3, 128, 128
        gray = T.functional.to_grayscale(rgb)
        return T.ToTensor()(gray)

    def make_rgb_transparent(rgb, alpha, mask, beta):
        return beta * mask + alpha * rgb

    batch_size = x_A.size(0)

    m_A = F.interpolate(m_A, None, 4, 'bilinear', align_corners=False)
    m_B = F.interpolate(m_B, None, 4, 'bilinear', align_corners=False)

    x_gray_A = torch.stack([convert_rgb_to_gray(x_A[i]) for i in range(x_A.size(0))], dim=0) # [B, 1, 128, 128]
    x_gray_B = torch.stack([convert_rgb_to_gray(x_B[i]) for i in range(x_B.size(0))], dim=0) # [B, 1, 128, 128]

    # combining the grayscale image with the mask
    masked_A = torch.stack([make_rgb_transparent(x_gray_A[i], 0.2, m_A[i].cpu(), 1) for i in range(x_A.size(0))], dim=0) # [B, 1, 128, 128]
    masked_B = torch.stack([make_rgb_transparent(x_gray_B[i], 0.2, m_B[i].cpu(), 1) for i in range(x_B.size(0))], dim=0) # [B, 1, 128, 128]

    inv_masked_A = torch.stack([make_rgb_transparent(x_gray_A[i], 0.2, 1 - m_A[i].cpu(), 1) for i in range(x_A.size(0))], dim=0)
    inv_masked_B = torch.stack([make_rgb_transparent(x_gray_B[i], 0.2, 1 - m_B[i].cpu(), 1) for i in range(x_B.size(0))], dim=0)

    # grayscale to 3 channels
    # masked_A = torch.stack([masked_A[i].repeat(3,1,1) for i in range(x_A.size(0))], dim=0) # [B, 3, 128, 128]
    # masked_B = torch.stack([masked_B[i].repeat(3,1,1) for i in range(x_B.size(0))], dim=0) # [B, 3, 128, 128]

    masked_A, masked_B = var_to_numpy(masked_A, isReal=False), var_to_numpy(masked_B, isReal=False)
    inv_masked_A, inv_masked_B = var_to_numpy(inv_masked_A, isReal=False), var_to_numpy(inv_masked_B, isReal=False)

    (x_A, x_B, x_AB, x_BA) = (var_to_numpy(x_A), var_to_numpy(x_B), var_to_numpy(x_AB), var_to_numpy(x_BA))

    x_fA, x_fB, x_bA, x_bB = (var_to_numpy(x_fA), var_to_numpy(x_fB), var_to_numpy(x_bA), var_to_numpy(x_bB))

    for x in range(batch_size):
        if x > (num_print-1) :
            break
        if label_A is not None:
            attributes = ''
            for idx, item in enumerate(label_A[x]):
                if int(item.data) == 1:
                    attributes += str(attrs[idx]) + '  '
            print('content attrs_%d: %s' % (x, attributes))

            attributes = ''
            for idx, item in enumerate(label_B[x]):
                if int(item.data) == 1:
                    attributes += str(attrs[idx]) + '  '
            print('style attrs_%d: %s' % (x, attributes))

        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20,15))
        axs[0][0].set_title('A')
        axs[0][0].imshow(x_A[x])
        axs[0][0].axis('off')
        axs[0][1].set_title('B')
        axs[0][1].imshow(x_B[x])
        axs[0][1].axis('off')
        axs[0][2].set_title('A >>> B')
        axs[0][2].imshow(x_AB[x])
        axs[0][2].axis('off')
        axs[0][3].set_title('B >>> A')
        axs[0][3].imshow(x_BA[x])
        axs[0][3].axis('off')


        axs[1][0].set_title('foreground of A')
        axs[1][0].imshow(x_fA[x])
        axs[1][0].axis('off')
        axs[1][1].set_title('foreground of B')
        axs[1][1].imshow(x_fB[x])
        axs[1][1].axis('off')
        axs[1][2].set_title('background of A')
        axs[1][2].imshow(x_bA[x])
        axs[1][2].axis('off')
        axs[1][3].set_title('background of B')
        axs[1][3].imshow(x_bB[x])
        axs[1][3].axis('off')

        axs[2][0].set_title('foreground mask for A')
        divider = make_axes_locatable(axs[2][0])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        vis_mask = axs[2][0].imshow(masked_A[x], cmap='bone', interpolation='bilinear')
        tick_limit(plt.colorbar(vis_mask, cax=cax, orientation='horizontal'))
        axs[2][0].axis('off')

        axs[2][1].set_title('foreground mask for B')
        divider = make_axes_locatable(axs[2][1])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        vis_mask = axs[2][1].imshow(masked_B[x], cmap='bone', interpolation='bilinear')
        tick_limit(plt.colorbar(vis_mask, cax=cax, orientation='horizontal'))
        axs[2][1].axis('off')

        axs[2][2].set_title('background mask for A')
        divider = make_axes_locatable(axs[2][2])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        vis_mask = axs[2][2].imshow(inv_masked_A[x], cmap='bone', interpolation='bilinear')
        tick_limit(plt.colorbar(vis_mask, cax=cax, orientation='horizontal'))
        axs[2][2].axis('off')

        axs[2][3].set_title('background mask for B')
        divider = make_axes_locatable(axs[2][3])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        vis_mask = axs[2][3].imshow(inv_masked_B[x], cmap='bone', interpolation='bilinear')
        tick_limit(plt.colorbar(vis_mask, cax=cax, orientation='horizontal'))
        axs[2][3].axis('off')

        plt.show()

def tick_limit(cb):
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()

def var_to_numpy(obj, isReal=True):
    obj = obj.permute(0,2,3,1)

    if isReal:
        obj = (obj+1) / 2
    else:
        obj = obj.squeeze(3)
    obj = torch.clamp(obj, min=0, max=1)
    return obj.data.cpu().numpy()

def ges_Aonfig(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def get_scheduler(optimizer, config, iterations=-1):
    if 'LR_POLICY' not in config or config['LR_POLICY'] == 'constant':
        scheduler = None # constant scheduler
    elif config['LR_POLICY'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['STEP_SIZE'],
                                        gamma=config['GAMMA'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config['LR_POLICY'])
    return scheduler

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))

def concat_input(x, c):
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, x.size(2), x.size(3))
    x = torch.cat([x, c], dim=1)
    return x
