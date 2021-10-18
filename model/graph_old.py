import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

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
    def __init__(self, in_channel, pixel_num):
        super(Intra_graph, self).__init__()
        self.in_channel = in_channel
        self.pixel_num = pixel_num
        self.nodes_num = 64
        self.inner_channel = 256
        self.relation_weight = nn.Parameter(torch.Tensor(self.pixel_num, self.nodes_num), requires_grad=True)
        self.relation_weight.data.normal_(0, 0.001)
        self.channel_conv1 = nn.Conv2d(in_channel, self.inner_channel, 1)
        self.fcgcn = FullyConnectGCLayer(self.inner_channel, self.inner_channel)
        self.channel_conv2 = nn.Conv2d(self.inner_channel, int(in_channel / 2), 1)
        

    def forward(self, x):
        #(1, 2048, 65, 129)
        b, c, w, h = x.shape
        x1 = self.channel_conv1(x)
        x1 = torch.reshape(x1, (b, self.inner_channel, w*h))
        # b, 256, n
        #x1 = torch.stack([torch.mm(a, F.softmax(self.relation_weight, 0)) for a in x1])
        x1 = torch.stack([torch.mm(a, self.relation_weight) for a in x1])
        # b, 256, 64
        x1 = self.fcgcn(x1)
        #x1 = torch.stack([torch.mm(a, F.softmax(self.relation_weight.T, 0)) for a in x1])
        x1 = torch.stack([torch.mm(a, self.relation_weight.T) for a in x1])

        x1 = torch.reshape(x1, (b, self.inner_channel, w, h))
        output = self.channel_conv2(x1)
        #self.draw_relation()
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

            





