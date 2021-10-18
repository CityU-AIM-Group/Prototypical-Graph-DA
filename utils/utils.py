import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from PIL import Image
import numpy as np
import os
import yaml
import json
import sys
from easydict import EasyDict
import matplotlib.pyplot as plt

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, learning_rate, cfg):
    # 24966 STEP per EPOCH
    max_iter = cfg.SRC_EPOCHES * cfg.DATASET.SOURCE_LEN / cfg.DATASET.BATCH_SIZE
    lr = lr_poly(learning_rate, i_iter, max_iter, cfg.TRAIN.POWER)
    for groups in optimizer.param_groups:
        groups['lr'] = lr * 10
        #optimizer.param_groups[1]['lr'] = lr * 10
        #optimizer.param_groups[2]['lr'] = lr * 10
    optimizer.param_groups[0]['lr'] = lr

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def cfg2str(cfg_s, cfg):
    for key, value in cfg.items():
        if type(value) is EasyDict:
            cfg_s[key] = EasyDict()
            cfg2str(cfg_s[key], value)
        else:
            cfg_s[key] = str(value)

def store_cfg(cfg):
    cfg_s = EasyDict()
    cfg2str(cfg_s, cfg)
    with open(cfg.LOG_DIR / cfg.EXP / 'configration.json', 'w') as outfile:
        json.dump(cfg_s, outfile, indent=4)
        outfile.write('\n')

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def distribution_plot(x, n):
    max_x = torch.max(x).cpu().detach()
    min_x = torch.min(x).cpu().detach()
    bin_x = (max_x - min_x) / n
    distribution = []
    x_bin = []
    for i in range(n):
        counter = torch.sum(x > (min_x + i * bin_x)) - torch.sum(x > (min_x + (i + 1) * bin_x))
        distribution.append(float(counter.cpu()))
        x_bin.append(float(min_x + i * bin_x))
    plt.plot(x_bin, distribution)
    plt.show()

def print_statis(name, x):
    print(name)
    print(x.shape)
    print('max', torch.max(x))
    print('min', torch.min(x))
    print('mean', torch.mean(x))
    print('var', torch.var(x))