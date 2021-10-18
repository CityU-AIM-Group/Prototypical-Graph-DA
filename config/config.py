from easydict import EasyDict
import pathlib
import numpy as np

cfg = EasyDict()
path_root = pathlib.Path(__file__).resolve().parents[2]

# general setting
cfg.EXP = ''
cfg.BACKBONE = 'Res101' #Vgg16
cfg.NUM_CLASSES = 4
cfg.GPU_ID = 0
cfg.SEED = 1234
cfg.SRC_EPOCHES = 150
cfg.SRC_STOP_EPOCHES = 100
cfg.SRC_MEMORY_START = 30
cfg.ADV_EPOCHES = 150
cfg.ADV_STOP_EPOCHES = 100
cfg.SHOW_NUM = 10
cfg.STORE_NUM = 5000
cfg.STORE_NUM_ADV = 2000
# pretrained model
cfg.FINETUNE_PATH = ''
# logs and snapshots
cfg.EXP_DIR = path_root / 'experiment'
cfg.STORE_DIR = cfg.EXP_DIR / 'store'
cfg.LOG_DIR = cfg.EXP_DIR / 'log'

cfg.MODEL = EasyDict()
cfg.MODEL.PROTO_CHANNEL = 512
cfg.MODEL.PROTO_NUM = 64

# dataset setting
cfg.DATASET = EasyDict()
cfg.DATASET.SOURCE = 'endo17'
cfg.DATASET.TARGET = 'endo18'
cfg.DATASET.BATCH_SIZE = 8
cfg.DATASET.NUM_WORKS = 4
cfg.DATASET.IGNORED_LABEL = 255
# directory
cfg.DATASET.SOURCE_DIR = pathlib.Path('/'.join(str(path_root.resolve()).split('/')[:4])) / 'data/EndoVis17'
cfg.DATASET.TARGET_DIR = pathlib.Path('/'.join(str(path_root.resolve()).split('/')[:4])) / 'data/EndoVis18'
cfg.DATASET.PESUDO_DIR = pathlib.Path('/'.join(str(path_root.resolve()).split('/')[:4])) / 'data/EndoVis18/visualize_typelbl'
# data list
cfg.DATASET.SOURCE_LIST = path_root / 'core/dataset/endovis17/train.txt'
cfg.DATASET.TARGET_LIST = path_root / 'core/dataset/endovis18/train.txt'
cfg.DATASET.TARGET_EVAL_LIST = path_root / 'core/dataset/endovis18/val.txt'
cfg.DATASET.SOURCE_IMG_MEAN = np.array([19.15855265, 44.82533179, 92.64762531], dtype=np.float32)
cfg.DATASET.TARGET_IMG_MEAN = np.array([93.97403134, 88.57638238, 119.21115404], dtype=np.float32)
cfg.DATASET.SOURCE_IMG_STD = np.array([28.64356778, 36.72181802, 54.20063624], dtype=np.float32)
cfg.DATASET.TARGET_IMG_STD = np.array([52.62953975, 50.02263679, 53.63186511], dtype=np.float32)
cfg.DATASET.INPUT_SIZE_SOURCE = (320, 256)
cfg.DATASET.INPUT_SIZE_TARGET = (320, 256)
cfg.DATASET.SOURCE_MAPPING = [0, 1, 1, 2, 255, 255, 3, 255, 255]  # 17 [0, 1, 1, 2, 255, 255, 3, 255, 255]
cfg.DATASET.TARGET_MAPPING = [0, 1, 1, 2, 3, 255, 255, 255] # 18 [0, 1, 1, 2, 3, 255, 255, 255]
#cfg.DATASET.SOURCE_LEN = 1800


# training setting
cfg.TRAIN = EasyDict()
cfg.TRAIN.PSEUDO_DIR = path_root / 'pseudo_label'
cfg.TRAIN.RESTORE_FROM = None
cfg.TRAIN.FINETUNE_PATH = path_root / 'pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
# data_set
cfg.TRAIN.MIRROR_PROB = 0.4
# params for segmentation
cfg.TRAIN.PSUEDO_WEIGHT = 0.5
cfg.TRAIN.AUX_WEIGHT = 0.5
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
# dice loss and focal loss
cfg.TRAIN.DICE_WEIGHT = [1, 1, 1.2, 4]
cfg.TRAIN.FOCAL_GAMMA = 2
# EM Part
cfg.TRAIN.EM_ITER = 4
cfg.TRAIN.MOMENTUM_mu = 0.998
# params for adversarial training
cfg.TRAIN.LEARNING_RATE_D = 1e-4
cfg.TRAIN.LAMBDA_ADV = 0.0001
cfg.TRAIN.GRL_ALPHA = 0.01 # for grl
# prototype
cfg.TRAIN.PROTO_LOAD = False
cfg.TRAIN.PROTO_PTH = path_root / 'pretrained_model/proto32.pth'