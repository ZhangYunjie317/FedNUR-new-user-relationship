import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class GraphAttentionLayer(nn.Module):       #基于图注意力机制的层
    def __init__(self, in_features, out_features, alpha = 0.1):         #输入和输出特征的维度，alpha控制激活函数的负斜率
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size = (in_features, out_features)))      #输入特征的权重矩阵
        nn.init.xavier_uniform_(self.W.data)
        self.a = nn.Parameter(torch.empty(size = (2 * out_features, 1)))        #注意力机制的权重向量
        nn.init.xavier_uniform_(self.a.data)
        self.W_1 = nn.Parameter(torch.randn(in_features, out_features))         #权重矩阵，用于邻居节点特征变换
        self.leakyrelu = nn.LeakyReLU(self.alpha)

#前向传播函数
    def forward(self, h, adj):          #h：输入特征矩阵    adj：邻接矩阵，表示节点间的连接关系
        W_h = torch.matmul(h, self.W)           #将特征矩阵和权重矩阵相乘
        W_adj = torch.mm(adj, self.W)           #将邻接矩阵和权重矩阵相乘
        a_input = torch.cat((W_h.repeat(W_adj.shape[0], 1), W_adj), dim = 1)        #拼接W_h和W_adj，计算注意力得分
        attention = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(-1)       #每对节点的注意力得分

        attention = F.softmax(attention, dim = -1)          #归一化
        W_adj_transform = torch.mm(adj, self.W_1)           #将邻接矩阵adj和权重矩阵W_1相乘，对邻居节点进行特征变化
        h = torch.matmul(attention, W_adj_transform)        #用注意力权重加权 邻居节点特征的和
        return h       #返回更新后的节点特征
