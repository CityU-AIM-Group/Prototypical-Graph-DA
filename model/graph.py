import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from utils.utils import print_statis, distribution_plot

class FullyConnectGCLayer(nn.Module):
    def __init__(self, in_channel, out_channel, node_num, res_connect=True):
        # if res_connct is true, in_channel should equal to out_channel
        super(FullyConnectGCLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.res_connect = res_connect
        self.diag_channel = 128
        self.adjacency_conv = nn.Conv1d(in_channel, self.diag_channel, 1)
        self.diagonal_conv = nn.Conv1d(in_channel, self.diag_channel, 1)
        self.weight = nn.Parameter(torch.Tensor(out_channel, in_channel), requires_grad=True)
        self.weight.data.normal_(0, 0.01)
        self.sigmoid = nn.Sigmoid()
        self.diagonal_i = nn.Parameter(torch.eye(node_num), requires_grad=False)


    def build_adjacent(self, x):
        b, c, n = x.shape
        # generate diagonal matrix
        x_diag = self.diagonal_conv(x)
        x_diag = F.avg_pool1d(x_diag, n)
        x_diag = torch.reshape(x_diag, (b, self.diag_channel))
        x_diag = torch.stack([torch.diag(a, 0) for a in x_diag])
        x_diag = self.sigmoid(x_diag)
        #reduce channel
        x = self.adjacency_conv(x)
        x_T = torch.transpose(x, 1, 2)
        adjacency_matrix = torch.bmm(torch.bmm(x_T, x_diag), x)
        adjacency_matrix = torch.stack([self._normalize(a) for a in adjacency_matrix])

        return adjacency_matrix

    def _normalize(self, matrix):
        matrix = F.relu(matrix)
        matrix_hat = matrix + self.diagonal_i
        with torch.no_grad():
            degree = torch.diag(torch.pow(torch.sum(matrix_hat, 1), -0.5))
        return torch.mm(degree, torch.mm(matrix_hat, degree))


    def forward(self, x):
        # x [b, c, n]
        adjacency_matrix = self.build_adjacent(x)
        output = torch.stack([torch.mm(self.weight, torch.mm(a, b)) for a, b in zip(x, adjacency_matrix)])
        output = F.relu(output, inplace=True)
        if self.res_connect:
            output = output + x
        return output


class Intra_graph(nn.Module):
    def __init__(self, in_channel, inner_channel, nodes_num, em_num, momentum):
        super(Intra_graph, self).__init__()
        self.in_channel = in_channel
        self.nodes_num = nodes_num
        self.inner_channel = inner_channel
        self.em_num = em_num
        self.momentum = momentum

        multi_proto = torch.Tensor(1, self.inner_channel, self.nodes_num)
        self.register_buffer('multi_proto', multi_proto)
        self.multi_proto.data.normal_(0, 0.001)
        pi = torch.ones([1, self.nodes_num])
        self.register_buffer('pi', pi)
        self.pi = self.pi / self.nodes_num

        self.in_conv = nn.Conv2d(in_channel, self.inner_channel, 1)
        self.fcgcn = FullyConnectGCLayer(self.inner_channel, self.inner_channel, nodes_num)
        self.out_conv = nn.Conv2d(self.inner_channel, int(in_channel), 1)
        self.out_conv2 = nn.Conv2d(self.inner_channel, int(in_channel), 1)
        self.out_bn = nn.BatchNorm2d(int(in_channel))
        self.out_bn2 = nn.BatchNorm2d(int(in_channel))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, flag):
        #(4 2048 41 41)
        b, c, w, h = x.shape
        x_res = x
        x1 = self.in_conv(x)
        #[4 512 41 41]
        x1 = torch.reshape(x1, (b, self.inner_channel, w*h))
        multi_proto = self.multi_proto.repeat(b, 1, 1)

        # em procedure
        with torch.no_grad():
            mu = multi_proto.data
            pi = self.pi.repeat(b, 1, 1)
            for i in range(self.em_num):
                likelihood = torch.bmm(x1.transpose(1, 2), mu)
                likelihood = likelihood - torch.max(likelihood, 2, keepdim=True)[0].detach()
                likelihood = torch.exp(likelihood)
                post_prob = likelihood * pi
                post_prob = post_prob / (1e-18 + torch.sum(post_prob, 2, keepdim=True))
                prob_ = post_prob / (1e-18 + torch.sum(post_prob, 1, keepdim=True))
                mu = torch.bmm(x1, prob_)
                pi = torch.sum(post_prob, 1, keepdim=True) / w / h
            if flag:
                self.multi_proto.data = self.momentum * self.multi_proto.data + (1 - self.momentum) * mu.mean(0)
                self.pi.data = self.momentum * self.pi.data + (1 - self.momentum) * pi.mean(0)
        x2 = torch.bmm(x1, prob_)
        x4 = torch.bmm(x2, post_prob.transpose(1,2))
        x4 = torch.reshape(x4, (b, self.inner_channel, w, h))
        x4 = self.out_bn2(self.out_conv2(x4))
        x4 = F.relu(x4, inplace=True)
        output2 = x4 + x_res
        output2 = F.relu(output2, inplace=True)
        #ema
        # x1t = x1.permute(0, 2, 1)
        # z = torch.bmm(x1t, multi_proto)
        # z = F.softmax(z, dim=2)
        # with torch.no_grad():
        #     normalization = 1 / (1e-6 + z.sum(dim=1, keepdim=True))
        # z_ = z * normalization
        # mu = torch.bmm(x1, z_)
        # x2 = self.fcgcn(mu)

        # z_t = z.permute(0, 2, 1)
        # x4 = multi_proto.matmul(z_t)
        # x4 = torch.reshape(x4, (b, self.inner_channel, w, h))
        # x4 = F.relu(x4, inplace=True)
        # x4 = self.out_conv2(x4)
        # output2 = x4 + x_res
        # output2 = F.relu(output2, inplace=True)

        # after gcn
        x2 = self.fcgcn(x2)
        x3 = torch.bmm(x2, post_prob.transpose(1,2))
        x3 = torch.reshape(x3, (b, self.inner_channel, w, h))
        x3 = self.out_bn(self.out_conv(x3))
        x3 = F.relu(x3, inplace=True)    

        output = x3 + x_res
        output = F.relu(output, inplace=True)
        #distribution_plot(x3, 100)
        return output, output2
    
    def relation_cal(self, x, label, cfg):
        relation = np.zeros((cfg.NUM_CLASSES, cfg.MODEL.PROTO_NUM))
        b, c, w, h = x.shape
        x1 = self.in_conv(x)
        x1 = torch.reshape(x1, (b, self.inner_channel, w*h))
        multi_proto = self.multi_proto.repeat(b, 1, 1)
        mu = multi_proto.data
        pi = self.pi.repeat(b, 1, 1)
        for i in range(self.em_num):
            likelihood = torch.bmm(x1.transpose(1, 2), mu)
            likelihood = likelihood - torch.max(likelihood, 2, keepdim=True)[0].detach()
            likelihood = torch.exp(likelihood)
            post_prob = likelihood * pi
            post_prob = post_prob / (1e-18 + torch.sum(post_prob, 2, keepdim=True)) # [1, 1353, 64], 1353
        label = torch.reshape(F.interpolate(label.float().unsqueeze(1), (w, h)), (b, w*h))
        
        for i in range(cfg.NUM_CLASSES):
            mask = (label == i).repeat(1, cfg.MODEL.PROTO_NUM, 1).transpose(1,2)
            post_prob_list = post_prob[mask].reshape(-1, 64)
            if len(post_prob_list) != 0:
                relation[i] = post_prob_list.mean(0).cpu()
        return relation


            
class Inter_graph(nn.Module):
    def __init__(self, in_channel, inner_channel):
        super(Inter_graph, self).__init__()
        self.inner_channel = inner_channel
        self.reduced_conv = nn.Conv2d(in_channel, inner_channel, 1)
        self.g_embedding = nn.Parameter(torch.Tensor(inner_channel, inner_channel), requires_grad=True)
        self.g_embedding.data.normal_(0, 0.01)
        self.g2i_conv = nn.Parameter(torch.Tensor(inner_channel, inner_channel), requires_grad=True)
        self.g2i_conv.data.normal_(0, 0.01)
        self.out_conv = nn.Conv2d(inner_channel, in_channel, 1)
        self.out_bn = nn.BatchNorm2d(in_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feature, coarse_pred, global_proto, relation):
        #torch.Size([1, 512, 91, 161])
        #torch.Size([1, 19, 91, 161])
        #torch.Size([1, 512, 64])
        #print_statis('feature', feature)

        reduced_feature = self.reduced_conv(feature)
        coarse_pred = F.softmax(coarse_pred, 1)
        b, c, w, h = coarse_pred.shape

        # get image prototype based on the coarse prediction
        coarse_pred = torch.reshape(coarse_pred, (b, c, w * h))
        reduced_feature = torch.reshape(reduced_feature, (b, self.inner_channel, w * h))
        #print(coarse_pred.shape, reduced_feature.shape)
        #input()
        #print_statis('1', reduced_feature)
        image_proto = torch.bmm(reduced_feature, coarse_pred.transpose(1, 2))
        #print(torch.sum(coarse_pred, 2, keepdim=True).transpose(1, 2).repeat(1, self.inner_channel, 1).shape, coarse_pred.shape)
        #input()
        image_proto = image_proto / torch.sum(coarse_pred, 2, keepdim=True).transpose(1, 2).detach()
        #print_statis('1', image_proto)
        #torch.Size([1, 512, 19])
        
        # transform to embedding
        global_proto = global_proto.repeat(b, 1, 1)
        global_proto = torch.stack([torch.mm(self.g2i_conv, a) for a in global_proto])
        adjacency_matrix = torch.bmm(global_proto.transpose(1, 2), image_proto)
        adjacency_matrix = torch.stack([self._normalize(a)*relation.transpose(0,1) for a in adjacency_matrix])
        #distribution_plot(adjacency_matrix, 100)
        #torch.Size([1, 64, 19])
        #GCN AFW
        global2image = torch.bmm(global_proto, adjacency_matrix)
        global2image = torch.stack([torch.mm(self.g_embedding, a) for a in global2image])
        #torch.Size([1, 512, 19])
        
        new_feature = torch.bmm(global2image, coarse_pred)
        #print_statis('new_feature', new_feature)
        
        new_feature = torch.reshape(new_feature, (b, self.inner_channel, w, h))

        new_feature = F.relu(self.out_bn(self.out_conv(new_feature)), inplace=True)
        output = new_feature + feature
        #print_statis('output', output)
        #distribution_plot(output, 100)
        output = F.relu(output, inplace=True)
        return output

    def _normalize(self, matrix):
        matrix = F.relu(matrix)
        with torch.no_grad():
            degree_l = torch.diag(torch.pow(torch.sum(matrix, 1) + 1e-18, -0.5))
            degree_r = torch.diag(torch.pow(torch.sum(matrix, 0) + 1e-18, -0.5))
        return torch.mm(degree_l, torch.mm(matrix, degree_r))


