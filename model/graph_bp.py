import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from utils.utils import print_statis, distribution_plot

# please check with graph

class FullyConnectGCLayer(nn.Module):
    def __init__(self, in_channel, out_channel, res_connect=True):
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
        self.diagonal_i = nn.Parameter(torch.eye(64), requires_grad=False)


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
        matrix = F.relu(matrix, inplace=True)
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
    def __init__(self, in_channel, inner_channel, nodes_num):
        super(Intra_graph, self).__init__()
        self.in_channel = in_channel
        self.nodes_num = nodes_num
        self.inner_channel = inner_channel
        self.multi_proto = nn.Parameter(torch.Tensor(1, self.inner_channel, self.nodes_num), requires_grad=True)
        self.multi_proto.data.normal_(0, math.sqrt(2 / self.nodes_num))
        self.in_conv = nn.Conv2d(in_channel, self.inner_channel, 1)
        self.fcgcn = FullyConnectGCLayer(self.inner_channel, self.inner_channel)
        self.out_conv = nn.Conv2d(self.inner_channel, int(in_channel), 1)
        

    def forward(self, x):
        #(1, 2048, 65, 129)
        b, c, w, h = x.shape
        x_res = x
        x1 = self.in_conv(x)
        x1 = torch.reshape(x1, (b, self.inner_channel, w*h))

        multi_proto = self.multi_proto.repeat(b, 1, 1)
        likelihood = torch.bmm(x1.transpose(1, 2), multi_proto)
        # torch.Size([1, 8385, 64])

        prob = F.softmax(likelihood, 2)
        prob_ = prob / (1e-6 + torch.sum(prob, 1, keepdim=True))
        
        x2 = torch.bmm(x1, prob_)
        #torch.Size([1, 512, 64])

        x2 = self.fcgcn(x2)

        x3 = torch.bmm(x2, prob.transpose(1,2))

        x3 = torch.reshape(x3, (b, self.inner_channel, w, h))
        output = self.out_conv(x3) + x_res
        return output

    def draw_relation(self):
        #print(self.relation_weight.data)
        projection = self.relation_weight.data.T
        print(torch.max(projection))
        print(torch.min(projection))
        print(torch.mean(projection))
        for i in range(64):
            sub_node = projection[i]
            img = torch.reshape(sub_node, (1, 1, 65, 129))
            img = F.interpolate(img, (512, 1024), mode='bilinear')[0][0]
            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img)) * 255
            img = Image.fromarray(img.cpu().numpy().astype(np.uint8))
            img.save(f'./{i}_channel.png')

            
class Inter_graph(nn.Module):
    def __init__(self, in_channel, inner_channel):
        super(Inter_graph, self).__init__()
        self.inner_channel = inner_channel
        self.reduced_conv = nn.Conv2d(in_channel, inner_channel, 1)
        self.g2i_conv = nn.Parameter(torch.Tensor(inner_channel, inner_channel), requires_grad=True)
        self.g2i_conv.data.normal_(0, math.sqrt(2 / inner_channel))
        self.out_conv = nn.Conv2d(in_channel + inner_channel, in_channel, 1)

    def forward(self, feature, coarse_pred, global_proto):
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
        #print_statis('1', reduced_feature)
        image_proto = torch.bmm(reduced_feature, coarse_pred.transpose(1, 2))
        image_proto = image_proto / torch.sum(coarse_pred, 2).repeat(b, self.inner_channel, 1)
        #print_statis('1', image_proto)
        #torch.Size([1, 512, 19])

        adjacency_matrix = torch.bmm(global_proto.transpose(1, 2), image_proto)
        adjacency_matrix = torch.stack([self._normalize(a) for a in adjacency_matrix])
        #print_statis('1', adjacency_matrix)
        #torch.Size([1, 64, 19])

        global2image = torch.bmm(global_proto, adjacency_matrix)
        global2image = torch.stack([torch.mm(self.g2i_conv, a) for a in global2image])
        #torch.Size([1, 512, 19])
        
        new_feature = torch.bmm(global2image, coarse_pred)
        #print_statis('new_feature', new_feature)
        
        new_feature = torch.reshape(new_feature, (b, self.inner_channel, w, h))

        con_feature = torch.cat((feature, new_feature), 1)
        output = self.out_conv(con_feature)
        #print_statis('output', output)
        #distribution_plot(output, 100)
        return output

    def _normalize(self, matrix):
        matrix = F.relu(matrix, inplace=True)
        with torch.no_grad():
            degree_l = torch.diag(torch.pow(torch.sum(matrix, 1) + 1e-6, -0.5))
            degree_r = torch.diag(torch.pow(torch.sum(matrix, 0) + 1e-6, -0.5))
        return torch.mm(degree_l, torch.mm(matrix, degree_r))


