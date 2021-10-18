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

category_name = ["road",
                "sidewalk",
                "building",
                "wall",
                "fence",
                "pole",
                "light",
                "sign",
                "vegetation",
                "terrain",
                "sky",
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motocycle",
                "bicycle"]

def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation Source Training")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    args = parser.parse_args()
    print(args)

    cfg.EXP = f'{cfg.DATASET.SOURCE}_{cfg.DATASET.TARGET}_{cfg.BACKBONE}'
    if args.exp_suffix:
        cfg.EXP += f'_{args.exp_suffix}'
    pprint(cfg)

    if not args.random_train:
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.SEED + worker_id)

    # dataset loader
    source_dataset = GTA5DataSet(root=cfg.DATASET.SOURCE_DIR,
                                 list_path=cfg.DATASET.SOURCE_LIST,
                                 mirror_prob=cfg.TRAIN.MIRROR_PROB,
                                 crop_size=cfg.DATASET.INPUT_SIZE_SOURCE,
                                 mean=cfg.DATASET.IMG_MEAN,
                                 ignore_label=cfg.DATASET.IGNORED_LABEL)
    source_loader = data.DataLoader(source_dataset,
                                    batch_size=cfg.DATASET.BATCH_SIZE,
                                    num_workers=cfg.DATASET.NUM_WORKS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    model = IGNet(cfg)
    # restore from
    model.load_state_dict(torch.load('/home/jay/code_jay/graph_domain/experiment/store/gta5_cityscapes_Res101/src/src.pth'))
    model.eval()
    model.cuda(cfg.GPU_ID)

    cudnn.benchmark = True
    cudnn.enabled = True

    category_paths = []
    for i in range(cfg.NUM_CLASSES):
        os.makedirs(cfg.STORE_DIR / cfg.EXP / 'proto' / (str(i) + category_name[i]), exist_ok=True)
        category_paths.append(cfg.STORE_DIR / cfg.EXP / 'proto' / (str(i) + category_name[i]))

    with torch.no_grad():
        for i, batch in enumerate(source_loader):
            image, label, _, name = batch

            model.proto_step(image.cuda(cfg.GPU_ID), label.cuda(cfg.GPU_ID), name, cfg, category_paths)
            if i % 1000 == 999:
                print(i)
            '''
            if i_iter % show_num == show_num - 1:
                writer.add_scalar('loss', ave_loss / show_num, round(i_iter / show_num))
                elapsed_time = time.time() - time0
                elapsed_time = datetime.timedelta(seconds=int(elapsed_time))
                estimated_time = (time.time() - tik_tok) / show_num 
                estimated_time = estimated_time * (cfg.STOP_EPOCHES * cfg.DATASET.SOURCE_LEN / cfg.DATASET.BATCH_SIZE - i_iter)
                estimated_time = datetime.timedelta(seconds=int(estimated_time))
                pprint(f'loss: {ave_loss / show_num} || Elapsed time: {elapsed_time}, Estimated time:{estimated_time}')
                tik_tok = time.time()
                ave_loss = 0
            '''





if __name__ == '__main__':
    main()