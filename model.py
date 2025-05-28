import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT import GraphAttentionLayer


class AttributeModel(nn.Module):
    def __init__(self, input_size, output_size=21, hidden_size=24):
        """
        定义单个属性的神经网络
        :param input_size: 输入层节点数，对应该属性的类别数
        :param output_size: 输出层节点数，默认为11 (评分从0到5，每0.5为一个节点)
        :param hidden_size: 隐藏层节点数，默认为50
        """
        super(AttributeModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 输入到隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_size, output_size),  # 隐藏层到输出层
            #nn.Softmax(dim=1)  # 对输出应用 softmax 激活函数
        )
        self._initialize_weights()

    # 初始化权重
    def _initialize_weights(self):
        for idx, m in enumerate(self.model):
            if isinstance(m, nn.Linear):
                if idx == 0:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向传播定义
        :param x: 输入张量，形状为 (batch_size, input_size)
        :return: 输出张量，形状为 (batch_size, output_size)
        """
        return self.model(x)


class ClientModels:
    def __init__(self, attribute_sizes):
        """
        初始化多个属性的模型
        :param attribute_sizes: 字典，键为属性名，值为对应的类别数
        """
        super(ClientModels, self).__init__()
        self.models = nn.ModuleDict({
            attr_name: AttributeModel(input_size=input_size)
            for attr_name, input_size in attribute_sizes.items()
        })

    def forward(self, attribute_data):      #好像没用上
        """
        批量前向传播
        :param attribute_data: 字典，键为属性名，值为对应的输入数据张量
        :return: 字典，键为属性名，值为对应的输出结果张量
        """
        print("ClientModel attribute_data: forward run")
        if not attribute_data:
            raise ValueError("attribute_data is empty. Please provide valid input data.")

        outputs = {}
        for attr_name, data in attribute_data.items():
            if attr_name in self.models:
                outputs[attr_name] = self.models[attr_name](data)
            else:
                raise KeyError(f"Attribute {attr_name} not found in models.")
        print("ClientModel outputs: ", outputs)
        return outputs

    def to(self, device):   #好像没用上
        """
        将所有子模型迁移到指定设备
        :param device: 目标设备（如 'cpu' 或 'cuda'）
        """
        for model in self.models.values():
            model.to(device)
        return self



class GraphModel(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        #GraphAttentionLayer实例，用于处理邻居节点和物品节点的注意力
        self.GAT_neighbor = GraphAttentionLayer(embed_size, embed_size)
        self.GAT_item = GraphAttentionLayer(embed_size, embed_size)
        #各种嵌入向量
        self.relation_neighbor = nn.Parameter(torch.randn(embed_size))
        self.relation_item = nn.Parameter(torch.randn(embed_size))
        self.relation_self = nn.Parameter(torch.randn(embed_size))
        self.c = nn.Parameter(torch.randn(2 * embed_size))

#计算物品嵌入和用户嵌入的矩阵乘法，用于评分的预测
    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

#前向传播函数，输入当前节点，邻居节点，物品节点的特征
    def forward(self, feature_self, feature_neighbor, feature_item):
        #如果项目特征是一个张量
        if type(feature_item) == torch.Tensor:
            #使用注意力机制计算邻居和物品节点的特征表示
            f_n = self.GAT_neighbor(feature_self, feature_neighbor)
            f_i = self.GAT_item(feature_self, feature_item)
            #对应公式789，计算用户用户，用户项目，用户自己的注意力权重
            e_n = torch.matmul(self.c, torch.cat((f_n, self.relation_neighbor)))
            e_i = torch.matmul(self.c, torch.cat((f_i, self.relation_item)))
            e_s = torch.matmul(self.c, torch.cat((feature_self, self.relation_self)))
            m = nn.Softmax(dim = -1)
            e_tensor = torch.stack([e_n, e_i, e_s])
            e_tensor = m(e_tensor)
            r_n, r_i, r_s = e_tensor
            user_embedding = r_s * feature_self + r_n * f_n + r_i * f_i
        #不是张量
        else:
            f_n = self.GAT_neighbor(feature_self, feature_neighbor)
            e_n = torch.matmul(self.c, torch.cat((f_n, self.relation_neighbor)))
            e_s = torch.matmul(self.c, torch.cat((feature_self, self.relation_self)))
            m = nn.Softmax(dim = -1)
            e_tensor = torch.stack([e_n, e_s])
            e_tensor = m(e_tensor)
            r_n, r_s = e_tensor
            user_embedding = r_s * feature_self + r_n * f_n

        return user_embedding

