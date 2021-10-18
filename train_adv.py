import os
import sys
import numpy as np
import time
import datetime
from PIL import Image
import numpy as np
import argparse
from pprint import pprint
from PIL import Image

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from config.config import cfg
from utils.utils import store_cfg
from model.IGNet import IGNet
from dataset.endovis_dataset import EndovisDataSet
from utils.utils import fast_hist, per_class_iu
from proto2center import proto2center


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
    os.makedirs(cfg.STORE_DIR / cfg.EXP / 'adv', exist_ok=True)
    os.makedirs(cfg.LOG_DIR / cfg.EXP / 'adv', exist_ok=True)
    store_cfg(cfg)
    pprint(cfg)

    if not args.random_train:
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)

    # dataset loader
    source_dataset = EndovisDataSet(root=cfg.DATASET.SOURCE_DIR,
                                list_path=cfg.DATASET.SOURCE_LIST,
                                mirror_prob=cfg.TRAIN.MIRROR_PROB,
                                crop_size=cfg.DATASET.INPUT_SIZE_SOURCE,
                                mean=cfg.DATASET.SOURCE_IMG_MEAN,
                                std=cfg.DATASET.SOURCE_IMG_STD,
                                ignore_label=cfg.DATASET.IGNORED_LABEL,
                                mapping=cfg.DATASET.SOURCE_MAPPING)
    source_loader = data.DataLoader(source_dataset,
                                batch_size=cfg.DATASET.BATCH_SIZE,
                                num_workers=cfg.DATASET.NUM_WORKS,
                                shuffle=True,
                                pin_memory=True)
    cfg.DATASET.SOURCE_LEN = len(source_dataset)
    target_dataset = EndovisDataSet(root=cfg.DATASET.TARGET_DIR,
                                list_path=cfg.DATASET.TARGET_LIST,
                                mirror_prob=cfg.TRAIN.MIRROR_PROB,
                                crop_size=cfg.DATASET.INPUT_SIZE_TARGET,
                                max_iters = len(source_dataset),
                                mean=cfg.DATASET.TARGET_IMG_MEAN,
                                std=cfg.DATASET.TARGET_IMG_STD,
                                ignore_label=cfg.DATASET.IGNORED_LABEL,
                                pseudo_label=True)
    target_loader = data.DataLoader(target_dataset,
                                batch_size=cfg.DATASET.BATCH_SIZE,
                                num_workers=cfg.DATASET.NUM_WORKS,
                                shuffle=True,
                                pin_memory=True)
    
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
                                    batch_size=cfg.DATASET.BATCH_SIZE,
                                    num_workers=cfg.DATASET.NUM_WORKS,
                                    shuffle=False,
                                    pin_memory=True)
    model = IGNet(cfg, 'adv')

    model.train()
    model.cuda(cfg.GPU_ID)

    cudnn.benchmark = True
    cudnn.enabled = True

    writer = SummaryWriter(log_dir=cfg.LOG_DIR / cfg.EXP / 'adv')
    show_num = cfg.SHOW_NUM
    store_num = cfg.STORE_NUM_ADV
    time0 = time.time()
    tik_tok = time.time()
    loss_group = {'loss_s': [], 'loss_adv': [], 'loss_dis': [], 'loss_dis_t': []}
    cfg.DATASET.SOURCE_LEN = len(source_dataset)
    for a in range(cfg.ADV_EPOCHES):
        model.memory_updata(a, cfg)
        relation_s, relation_t = proto2center(model, cfg)
        #relation_s = np.ones((cfg.NUM_CLASSES, cfg.MODEL.PROTO_NUM))
        #relation_t = np.ones((cfg.NUM_CLASSES, cfg.MODEL.PROTO_NUM))
        relation_s = torch.tensor(relation_s).float().cuda(cfg.GPU_ID)
        relation_t = torch.tensor(relation_t).float().cuda(cfg.GPU_ID)
        target_loader_iter = iter(target_loader)
        for i, batch in enumerate(source_loader):
            i_iter = a * cfg.DATASET.SOURCE_LEN // cfg.DATASET.BATCH_SIZE + i
            image_t, pseudo_label, _, _ = target_loader_iter.next()
            image, label, _, _ = batch
            loss_s, loss_adv, loss_dis, loss_dis_t = model.adv_step(a, i, image.cuda(cfg.GPU_ID), label.long().cuda(cfg.GPU_ID), 
                                                            image_t.cuda(cfg.GPU_ID), pseudo_label.long().cuda(cfg.GPU_ID), 
                                                            relation_s, relation_t, cfg)
            loss_group['loss_s'].append(loss_s)
            loss_group['loss_adv'].append(loss_adv)
            loss_group['loss_dis'].append(loss_dis)
            loss_group['loss_dis_t'].append(loss_dis_t)
            
            if i_iter % show_num == 0:
                for k, v in loss_group.items():
                    writer.add_scalar(k, np.mean(v), round(i_iter / show_num))
                elapsed_time = time.time() - time0
                elapsed_time = datetime.timedelta(seconds=int(elapsed_time))
                estimated_time = (time.time() - tik_tok) / show_num 
                estimated_time = estimated_time * (cfg.ADV_STOP_EPOCHES * cfg.DATASET.SOURCE_LEN / cfg.DATASET.BATCH_SIZE - i_iter)
                estimated_time = datetime.timedelta(seconds=int(estimated_time))
                max_iter = cfg.ADV_EPOCHES * cfg.DATASET.SOURCE_LEN / cfg.DATASET.BATCH_SIZE
                pprint(f'{i_iter}/{max_iter} || Elapsed time: {elapsed_time}, Estimated time:{estimated_time}')
                tik_tok = time.time()
                loss_group = {'loss_s': [], 'loss_adv': [], 'loss_dis': [], 'loss_dis_t': []}
        if a % 2 == 0:
            model.model_store(cfg, a)

            model.eval()
            print(f'{a} epoch testing now...')
            hist = np.ones((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            for i, batch in enumerate(eval_loader):
                with torch.no_grad():
                    image, label, _, name = batch
                    hist += model.eval_step(i, image.float().cuda(cfg.GPU_ID), label.long().cuda(cfg.GPU_ID), relation_t, cfg)
            inters_over_union_classes = per_class_iu(hist)
            c_iou = {}
            for index, i in enumerate(inters_over_union_classes):
                c_iou[str(index)] = i
            writer.add_scalars('iou', c_iou, a)
            eval_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
            writer.add_scalar('eval_miou', eval_miou, a)
            model.train()


        if a == (cfg.ADV_STOP_EPOCHES - 1):
            model.model_store(cfg, i_iter)
            break

if __name__ == '__main__':
    main()