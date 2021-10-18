import os
import sys
import numpy as np
import time
import datetime
from PIL import Image
import numpy as np
import argparse
from pprint import pprint

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

sys.path.append('..')
from config.config import cfg
from utils.utils import store_cfg
from model.IGNet import IGNet
from dataset.endovis_dataset import EndovisDataSet
from utils.utils import per_class_iu

palette = [0, 0, 0,
           0, 137, 255,
           255, 165, 0,
           255, 156, 201,
           99, 0, 255,
           255, 0, 0,
           255, 0, 165,
           141, 141, 141,
           255, 218, 0]
zero_pad = 255 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
palette.append(255)
palette.append(255)
palette.append(255)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation Source Training")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    args = parser.parse_args()
    print(args)

    cfg.EXP = f'{cfg.DATASET.SOURCE}_{cfg.DATASET.TARGET}_{cfg.BACKBONE}'
    if args.exp_suffix:
        cfg.EXP += f'_{args.exp_suffix}'
    os.makedirs(cfg.STORE_DIR / cfg.EXP / 'src', exist_ok=True)
    os.makedirs(cfg.LOG_DIR / cfg.EXP / 'src', exist_ok=True)
    store_cfg(cfg)
    pprint(cfg)

    # dataset loader
    target_dataset = EndovisDataSet(root=cfg.DATASET.TARGET_DIR,
                                list_path=cfg.DATASET.TARGET_LIST,
                                mirror_prob=0,
                                crop_size=cfg.DATASET.INPUT_SIZE_TARGET,
                                max_iters = None,
                                mean=cfg.DATASET.TARGET_IMG_MEAN,
                                std=cfg.DATASET.TARGET_IMG_STD,
                                ignore_label=cfg.DATASET.IGNORED_LABEL,
                                mapping=cfg.DATASET.SOURCE_MAPPING)
    target_loader = data.DataLoader(target_dataset,
                                batch_size=1,
                                num_workers=cfg.DATASET.NUM_WORKS,
                                shuffle=False,
                                pin_memory=True)

    model = IGNet(cfg, 'src')
    model.eval()
    model.cuda(0)

    cudnn.benchmark = True
    cudnn.enabled = True


    model.model_restore_eval(cfg, 95)
    predicted_label = np.zeros((len(target_loader), 256, 320))
    predicted_prob = np.zeros((len(target_loader), 256, 320))
    image_name = []

    with torch.no_grad():
        for i, batch in enumerate(target_loader):
            image, _, _, name = batch
            predict = model.pseudo_step(i, image.float().cuda(cfg.GPU_ID), cfg)
            predict = predict.cpu().data[0].numpy()
            label, prob = np.argmax(predict, axis=0), np.max(predict, axis=0)
            predicted_label[i] = label.copy()
            predicted_prob[i] = prob.copy()
            image_name.append(name[0])
        thres = []
        for i in range(4):
            x = predicted_prob[predicted_label==i]
            if len(x) == 0:
                thres.append(0)
                continue        
            x = np.sort(x)
            thres.append(x[np.int(np.round(len(x)*0.75))])
        print(thres)
        thres = np.array(thres)
        thres[thres>0.95]=0.95
        print(thres)

        for index in range(len(target_loader)):
            name = image_name[index]
            name = name.split('/')[-1]
            label = predicted_label[index]
            prob = predicted_prob[index]
            for i in range(4):
                label[(prob<thres[i])*(label==i)] = 255
            output = np.asarray(label, dtype=np.uint8)
            output_color = colorize_mask(output)
            output_color.save('%s/%s' % ('result',  name.split('.')[0] + '_color.png'))
            output = Image.fromarray(output)
            output.save('%s/%s' % ('result', name))


if __name__ == '__main__':
    main()