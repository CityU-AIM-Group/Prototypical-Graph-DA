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

sys.path.append('..')
from config.config import cfg
from utils.utils import store_cfg
from model.IGNet import IGNet
from dataset.endovis_dataset import EndovisDataSet
from utils.utils import fast_hist, per_class_iu


def proto2center(model, cfg):
    source_dataset = EndovisDataSet(root=cfg.DATASET.SOURCE_DIR,
                                list_path=cfg.DATASET.SOURCE_LIST,
                                mirror_prob=0,
                                crop_size=cfg.DATASET.INPUT_SIZE_SOURCE,
                                mean=cfg.DATASET.SOURCE_IMG_MEAN,
                                std=cfg.DATASET.SOURCE_IMG_STD,
                                ignore_label=cfg.DATASET.IGNORED_LABEL,
                                mapping=cfg.DATASET.SOURCE_MAPPING)
    source_loader = data.DataLoader(source_dataset,
                                batch_size=1,
                                num_workers=cfg.DATASET.NUM_WORKS,
                                pin_memory=True)

    target_dataset = EndovisDataSet(root=cfg.DATASET.TARGET_DIR,
                                list_path=cfg.DATASET.TARGET_LIST,
                                mirror_prob=0,
                                crop_size=cfg.DATASET.INPUT_SIZE_TARGET,
                                max_iters = len(source_dataset),
                                mean=cfg.DATASET.TARGET_IMG_MEAN,
                                std=cfg.DATASET.TARGET_IMG_STD,
                                ignore_label=cfg.DATASET.IGNORED_LABEL,
                                pseudo_label=True)
    target_loader = data.DataLoader(target_dataset,
                                batch_size=1,
                                num_workers=cfg.DATASET.NUM_WORKS,
                                shuffle=True,
                                pin_memory=True)

    model.eval()
    relation_s = np.zeros((cfg.NUM_CLASSES, cfg.MODEL.PROTO_NUM))
    count_s = np.zeros(cfg.NUM_CLASSES)
    relation_t = np.zeros((cfg.NUM_CLASSES, cfg.MODEL.PROTO_NUM))
    count_t = np.zeros(cfg.NUM_CLASSES)
    for i, batch in enumerate(source_loader):
        with torch.no_grad():
            image, label, _, name = batch
            single_relation = model.relation_cal(image.cuda(cfg.GPU_ID), label.long().cuda(cfg.GPU_ID), cfg)
            for j in range(cfg.NUM_CLASSES):
                if np.sum(single_relation[j]):
                    count_s[j] += 1
            relation_s += single_relation
    for j in range(cfg.NUM_CLASSES):
        relation_s[j] = relation_s[j] / count_s[j]

    for i, batch in enumerate(target_loader):
        with torch.no_grad():
            image, label, _, name = batch
            single_relation = model.relation_cal(image.cuda(cfg.GPU_ID), label.long().cuda(cfg.GPU_ID), cfg)
            for j in range(cfg.NUM_CLASSES):
                if np.sum(single_relation[j]):
                    count_t[j] += 1
            relation_t += single_relation
    for j in range(cfg.NUM_CLASSES):
        relation_t[j] = relation_t[j] / count_t[j]
    model.train()
    return relation_s, relation_t