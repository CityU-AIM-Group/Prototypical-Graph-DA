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
from proto2center import proto2center


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
    eval_dataset = EndovisDataSet(root=cfg.DATASET.TARGET_DIR,
                                    list_path=cfg.DATASET.TARGET_EVAL_LIST,
                                    mirror_prob=0,
                                    crop_size=cfg.DATASET.INPUT_SIZE_TARGET,
                                    mean=cfg.DATASET.TARGET_IMG_MEAN,
                                    std=cfg.DATASET.TARGET_IMG_STD,
                                    ignore_label=cfg.DATASET.IGNORED_LABEL,
                                    mapping=cfg.DATASET.TARGET_MAPPING,
                                    )
    eval_loader = data.DataLoader(eval_dataset,
                                    batch_size=1,
                                    num_workers=cfg.DATASET.NUM_WORKS,
                                    shuffle=False,
                                    pin_memory=True)

    model = IGNet(cfg, 'src')
    model.eval()
    model.cuda(cfg.GPU_ID)

    cudnn.benchmark = True
    cudnn.enabled = True

    from PIL import Image
    palette = [0, 0, 0,
            0, 137, 255,
            255, 165, 0,
            255, 156, 201,
            99, 0, 255,
            255, 0, 0,
            255, 0, 165,
            141, 141, 141,
            255, 218, 0]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    def colorize_mask(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask
        
    for a in range(100):
        a = 38
        model.model_restore_eval(cfg, a)
        #relation_s, relation_t = proto2center(model, cfg)
        relation_t = np.zeros((4, cfg.MODEL.PROTO_NUM))
        relation_t = torch.tensor(relation_t).float().cuda(cfg.GPU_ID)
        
        hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
        dice = np.zeros(cfg.NUM_CLASSES)
        dice_count = np.zeros(cfg.NUM_CLASSES)
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                image, label, _, name = batch
                single_hist = model.eval_step(i, image.float().cuda(cfg.GPU_ID), label.long().cuda(cfg.GPU_ID),relation_t, cfg)
                
                hist += single_hist
                for k in range(cfg.NUM_CLASSES):
                    if np.sum(single_hist[k]) != 0:
                        dice_count[k] += 1
                        dice[k] += (2*single_hist[k][k]) / (np.sum(single_hist[k,:]) + np.sum(single_hist[:, k]))
                if i % 100 == 0:
                    print(i)
                # predict = np.asarray( predict.cpu().numpy()[0], dtype=np.uint8 )
                # output_col = colorize_mask(predict)
                # output_nomask = Image.fromarray(predict)    
                # name = name[0].split('/')[-1]
                # output_nomask.save(  '%s/%s' % ('./result', name)  )
                # output_col.save(  '%s/%s_color.png' % ('./result', name.split('.')[0])  ) 

        inters_over_union_classes = per_class_iu(hist)
        print(a, round(np.nanmean(inters_over_union_classes) * 100, 2))
        print(inters_over_union_classes)
        print(dice/dice_count)
        break


if __name__ == '__main__':
    main()