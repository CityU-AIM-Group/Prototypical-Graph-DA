import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from PIL import Image

from .feature_extractor import ResBackbone, VggBackbone
from .discriminator import FCDiscriminator_O
from .graph import Intra_graph, Inter_graph
from .classifier import Classifier_Module
from .grl.module import RevGrad

import sys
sys.path.append("..")
from utils.loss import Focal_CrossEntropy2d, BinaryCE, DiceLoss, WeightBinaryCE
from utils.utils import adjust_learning_rate, colorize_mask
from utils.utils import fast_hist, per_class_iu, print_statis, distribution_plot

class IGNet(nn.Module):
    def __init__(self, cfg, stage='src'):
        super(IGNet, self).__init__()
        self.memory_load = cfg.TRAIN.PROTO_LOAD
        self.stage = stage
        self.model_init(cfg)
        self.model_restore(cfg)
        self.optimizer_init(cfg)
        

    def model_init(self, cfg):
        inner_channel = cfg.MODEL.PROTO_CHANNEL
        nodes_num = cfg.MODEL.PROTO_NUM
        if cfg.BACKBONE == 'Res101':
            self.backbone = ResBackbone()
        elif cfg.BACKBONE == 'Vgg16':
            self.backbone = VggBackbone()
        else:
            raise NotImplementedError(f"Not yet supported {cfg.BACKBONE}")
        
        self.intra_graph = Intra_graph(2048, inner_channel, nodes_num, cfg.TRAIN.EM_ITER, cfg.TRAIN.MOMENTUM_mu)
        self.inter_graph = Inter_graph(2048, inner_channel)
        prototype_memory = torch.Tensor(1, inner_channel, nodes_num)
        self.register_buffer('prototype_memory', prototype_memory)
        self.grl = RevGrad(cfg.TRAIN.GRL_ALPHA)
        self.classifier = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], cfg.NUM_CLASSES)
        self.classifier2 = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], cfg.NUM_CLASSES)
        self.ref_classifier = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], cfg.NUM_CLASSES)
        img_size = cfg.DATASET.INPUT_SIZE_SOURCE
        self.interp = nn.Upsample(size=(img_size[1], img_size[0]),
                                    mode='bilinear', align_corners=True)
        self.cross_criterion = Focal_CrossEntropy2d(ignore_label=cfg.DATASET.IGNORED_LABEL, gamma=cfg.TRAIN.FOCAL_GAMMA)
        self.dice_criterion = DiceLoss(weight=cfg.TRAIN.DICE_WEIGHT, ignore_index=[0], class_num=cfg.NUM_CLASSES, ignore_label=255)
        img_size_t = cfg.DATASET.INPUT_SIZE_TARGET
        self.interp_t = nn.Upsample(size=(img_size_t[1], img_size_t[0]),
                                mode='bilinear', align_corners=True)

        if self.stage == 'adv':
            self.discriminator = FCDiscriminator_O(cfg.NUM_CLASSES) #FCDiscriminator_F(2048)
            self.discriminator_aux = FCDiscriminator_O(cfg.NUM_CLASSES) #FCDiscriminator_F(2048)
            self.bce_criterion = BinaryCE()
            self.advbce_criterion = WeightBinaryCE()


    def model_restore(self, cfg):
        saved_state_dict = torch.load(cfg.TRAIN.FINETUNE_PATH)
        params = self.backbone.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                params['.'.join(i_parts[1:])] = saved_state_dict[i]
        self.backbone.load_state_dict(params)
        if self.memory_load:
            prototype_dict = torch.load(cfg.TRAIN.PROTO_PTH, map_location=torch.device('cpu'))
            self.prototype_memory.data = prototype_dict['multi_proto']
        if cfg.TRAIN.RESTORE_FROM:
            pass
 


    def model_store(self, cfg, i_iter):
        torch.save(self.backbone.state_dict(), cfg.STORE_DIR / cfg.EXP / self.stage / f'backbone_{i_iter}.pth')
        torch.save(self.intra_graph.state_dict(), cfg.STORE_DIR / cfg.EXP / self.stage / f'intra_{i_iter}.pth')
        torch.save(self.inter_graph.state_dict(), cfg.STORE_DIR / cfg.EXP / self.stage / f'inter_{i_iter}.pth')
        torch.save(self.classifier.state_dict(), cfg.STORE_DIR / cfg.EXP / self.stage / f'class_{i_iter}.pth')
        torch.save(self.ref_classifier.state_dict(), cfg.STORE_DIR / cfg.EXP / self.stage / f'refclass_{i_iter}.pth')


    def model_restore_eval(self, cfg, i_iter):
        self.backbone.load_state_dict(torch.load(cfg.STORE_DIR / cfg.EXP / 'adv' / f'backbone_{i_iter}.pth'))
        self.intra_graph.load_state_dict(torch.load(cfg.STORE_DIR / cfg.EXP / 'adv' / f'intra_{i_iter}.pth'))
        self.inter_graph.load_state_dict(torch.load(cfg.STORE_DIR / cfg.EXP / 'adv' / f'inter_{i_iter}.pth'))
        self.classifier.load_state_dict(torch.load(cfg.STORE_DIR / cfg.EXP / 'adv' / f'class_{i_iter}.pth'))
        self.ref_classifier.load_state_dict(torch.load(cfg.STORE_DIR / cfg.EXP / 'adv' / f'refclass_{i_iter}.pth'))


    def optimizer_init(self, cfg):
        parameters = [{'params': self.backbone.parameters(), 'lr': cfg.TRAIN.LEARNING_RATE},
                      {'params': self.intra_graph.parameters(), 'lr': cfg.TRAIN.LEARNING_RATE * 5},
                      {'params': self.inter_graph.parameters(), 'lr': cfg.TRAIN.LEARNING_RATE * 5},
                      {'params': self.classifier.parameters(), 'lr': cfg.TRAIN.LEARNING_RATE * 10},
                      {'params': self.ref_classifier.parameters(), 'lr': cfg.TRAIN.LEARNING_RATE * 10},
                      {'params': self.classifier2.parameters(), 'lr': cfg.TRAIN.LEARNING_RATE * 10},
                      ]
        self.optim = optim.SGD(parameters,
                        lr=cfg.TRAIN.LEARNING_RATE,
                        momentum=cfg.TRAIN.MOMENTUM,
                        weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        if self.stage == 'adv':
            self.optim_d = optim.Adam(self.discriminator.parameters(), 
                            lr=cfg.TRAIN.LEARNING_RATE_D, betas=(0.9, 0.99))
            self.optim_d_aux = optim.Adam(self.discriminator_aux.parameters(), 
                                lr=cfg.TRAIN.LEARNING_RATE_D, betas=(0.9, 0.99))


    def memory_updata(self, a, cfg):
        self.prototype_memory.data = self.intra_graph.multi_proto.data
        self.memory_load = True
        

    def src_step(self, a, i, image, label, cfg):
        i_iter = a * cfg.DATASET.SOURCE_LEN // cfg.DATASET.BATCH_SIZE + i
        
        self.optim.zero_grad()
        adjust_learning_rate(self.optim, i_iter, cfg.TRAIN.LEARNING_RATE, cfg)

        x = self.backbone(image)
        x = self.intra_graph(x, True)
        coarse_pred = self.classifier(x)

        coarse_pred = self.interp(coarse_pred)
        loss_c = self.cross_criterion(coarse_pred, label)
        loss = loss_c
        
        '''
        label_img = Image.fromarray(label.cpu().numpy().astype(np.uint8)[0]*10)
        label_img.show()
        coarse = coarse_pred.detach()[0].permute(1, 2, 0).cpu().numpy()
        coarse = np.argmax(coarse, 2)
        predict_img = Image.fromarray(coarse.astype(np.uint8)*10)
        predict_img.show()

        predict = F.softmax(coarse_pred, dim=1)[0]
        predict = predict.permute(1, 2, 0)
        predict = torch.argmax(predict, dim=2)
        hist = self.compute_hist(predict, label, cfg)
        inters_over_union_classes = per_class_iu(hist)
        print(inters_over_union_classes)
        print(round(np.nanmean(inters_over_union_classes) * 100, 2))
        input()
        '''

        loss.backward()

        self.optim.step()

        return loss.item()
    
    def src_step_t(self, image, cfg):
        x = self.backbone(image)
        x = self.intra_graph(x)
        self.optim.zero_grad()

    def adv_step(self, a, i, image, label, image_t, pseudo_label, relation_s, relation_t, cfg):
        source_label = 0
        target_label = 1
        i_iter = a * cfg.DATASET.SOURCE_LEN // cfg.DATASET.BATCH_SIZE + i
        self.optim.zero_grad()
        self.optim_d.zero_grad()
        self.optim_d_aux.zero_grad()
        adjust_learning_rate(self.optim, i_iter, cfg.TRAIN.LEARNING_RATE, cfg)
        adjust_learning_rate(self.optim_d, i_iter, cfg.TRAIN.LEARNING_RATE_D, cfg)
        adjust_learning_rate(self.optim_d_aux, i_iter, cfg.TRAIN.LEARNING_RATE_D, cfg)


        # source supervision process
        x = self.backbone(image)
        x, x2 = self.intra_graph(x, True) # intra graph
        ###
        aux_prediction = self.classifier2(x2)
        aux_prediction = self.interp(aux_prediction)
        loss_aux = self.cross_criterion(aux_prediction, label)
        ###
        coarse_pred = self.classifier(x)
        refine_x = self.inter_graph(x, coarse_pred.detach(), self.prototype_memory.detach(), relation_s) # inter graph
        refine_pred = self.ref_classifier(refine_x)
        refine_pred = self.interp(refine_pred)
        loss_r = self.cross_criterion(refine_pred, label)
        coarse_pred = self.interp(coarse_pred)
        loss_c = self.cross_criterion(coarse_pred, label)
        loss_dice = self.dice_criterion(refine_pred, label)

        loss_s = (loss_c + loss_aux) * cfg.TRAIN.AUX_WEIGHT + loss_r + loss_dice
        loss_s.backward()

        # adversarial process
        x_t = self.backbone(image_t)
        x_t, _ = self.intra_graph(x_t, True) # intra graph
        coarse_pred_t = self.classifier(x_t)
        refine_x_t = self.inter_graph(x_t, coarse_pred_t.detach(), self.prototype_memory.detach(), relation_t)
        with torch.no_grad():
            estimator_s = self.inter_graph(x_t, coarse_pred_t.detach(), self.prototype_memory.detach(), relation_s)
            estimator_s = estimator_s / estimator_s.norm(dim=1, keepdim=True)
            estimator_t = refine_x_t / refine_x_t.norm(dim=1, keepdim=True)
            estimator = estimator_s * estimator_t
            estimator = (1 - estimator.sum(1)) + 0.4
        refine_pred_t = self.ref_classifier(refine_x_t)
        refine_pred_t = self.interp_t(refine_pred_t)
        coarse_pred_t = self.interp_t(coarse_pred_t)
        loss_r_t = self.cross_criterion(refine_pred_t, pseudo_label)
        loss_c_t = self.cross_criterion(coarse_pred_t, pseudo_label)

        d_pred_t = self.discriminator(refine_pred_t)
        d_pred_aux_t = self.discriminator_aux(coarse_pred_t)
        
        loss_adv = self.advbce_criterion(d_pred_t, source_label, estimator.detach())
        loss_adv_aux = self.advbce_criterion(d_pred_aux_t, source_label, estimator.detach())
        loss_a = loss_adv_aux * cfg.TRAIN.AUX_WEIGHT + loss_adv
        loss_psuedo = loss_r_t + loss_c_t * cfg.TRAIN.AUX_WEIGHT
        loss = loss_a * cfg.TRAIN.LAMBDA_ADV + loss_psuedo * cfg.TRAIN.PSUEDO_WEIGHT
        loss.backward()

        

        self.optim.step()
        self.optim_d.zero_grad()
        self.optim_d_aux.zero_grad()

        # discriminator training
        d_pred = self.discriminator(refine_pred.detach())
        d_pred_aux = self.discriminator_aux(coarse_pred.detach())
        loss_dis = self.bce_criterion(d_pred, source_label)
        loss_dis_aux = self.bce_criterion(d_pred_aux, source_label)
        

        d_pred_t = self.discriminator(refine_pred_t.detach())
        d_pred_aux_t = self.discriminator_aux(coarse_pred_t.detach())
        loss_dis_t = self.bce_criterion(d_pred_t, target_label)
        loss_dis_aux_t = self.bce_criterion(d_pred_aux_t, target_label)
        loss_d = (loss_dis_t + loss_dis) * 0.5
        loss_d_aux = (loss_dis_aux + loss_dis_aux_t) * 0.5
        loss = loss_d_aux + loss_d
        loss.backward()

        self.optim_d.step()
        self.optim_d_aux.step()

        return (loss_s.item(), 0, 0, 0)#(loss_s.item(), loss_adv_aux.item(), loss_dis_aux.item(), loss_dis_aux_t.item())
    
    def eval_step(self, i, image, label, relation, cfg):
        x = self.backbone(image)
        x, _ = self.intra_graph(x, False)
        coarse_pred = self.classifier(x)
        refine_x = self.inter_graph(x, coarse_pred, self.prototype_memory, relation)
        refine_pred = self.ref_classifier(refine_x)
        refine_pred = self.interp_t(refine_pred)
        predict = F.softmax(refine_pred, dim=1)

        #coarse_pred = self.interp_t(coarse_pred)
        #predict = F.softmax(coarse_pred, dim=1)
        predict = predict.permute(0, 2, 3, 1)
        predict = torch.argmax(predict, dim=3)

        return self.compute_hist(predict, label, cfg)
        

    def compute_hist(self, predict, label, cfg):
        hist = fast_hist(label.cpu().numpy().flatten(), predict.cpu().numpy().flatten(), cfg.NUM_CLASSES)
        return hist

    def pseudo_step(self, i, image, cfg):
        x = self.backbone(image)
        coarse_pred = self.classifier(x)
        coarse_pred = self.interp_t(coarse_pred)
        predict = F.softmax(coarse_pred, dim=1)

        return predict
    
    def relation_cal(self, image, label, cfg):
        x = self.backbone(image)
        relation = self.intra_graph.relation_cal(x, label, cfg)

        return relation