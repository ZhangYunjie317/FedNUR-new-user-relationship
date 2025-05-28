import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import dgl
from model import ClientModels
from model import GraphModel
from random import sample
#from test import compute_gradient_similarity

class Client:
    def __init__(self, client_id, client_data, attribute_sizes, neighbors, items, negative_sample, clip, laplace_lambda, embed_size, ratings, learning_rate=0.0001):
        """
        初始化客户端
        :param client_id: 客户端编号
        :param attribute_sizes: 字典，键为属性名，值为对应类别数 (如 {"feat": 31, "influence": 4, "quality": 4, "video_type": 2})
        :param learning_rate: 学习率
        """
        self.client_id = client_id
        self.client_data = client_data
        self.learning_rate = learning_rate

        # 初始化所有属性的模型

        self.models = ClientModels(attribute_sizes)
        self.optimizers = {
            attr_name: optim.Adam(self.models.models[attr_name].parameters(), lr=learning_rate)
            for attr_name in attribute_sizes.keys()
        }

        self.attribute_graph = {}

        self.embed_size = embed_size        #嵌入向量的维度大小，用于在模型中嵌入用户特征
        self.graph_model = GraphModel(embed_size = 64)
        self.gnn_optimizer = optim.Adam(self.graph_model.parameters())  # GNN模型的优化器

        self.neighbors = neighbors
        self.items = items 
        self.negative_sample = negative_sample
        self.clip = clip        #控制差分隐私的剪裁参数。
        self.laplace_lambda = laplace_lambda        #控制差分隐私的拉普拉斯噪声参数。
        self.ratings = ratings      #用户对这些物品的评分列表。

        self.graph = self.build_local_graph(client_id, items, neighbors)      #构建的本地图
        self.graph = dgl.add_self_loop(self.graph)      #添加自环
        self.user_feature = torch.randn(self.embed_size)        #为当前用户初始化一个随机的用户特征向量


    def local_train_nn(self, epochs=20):
        """
        客户端本地训练用户偏好神经网络,计算 Wb - Wa / -n 作为最终梯度
        :param epochs: 训练轮数
        """
        gradients = {}
        initial_params = {} #保存初始参数
        learning_rate = self.learning_rate

        #1. 存储W_a（训练前的参数）
        for attr_name, model in self.models.models.items():
            initial_params[attr_name] = {name: param.clone().detach() for name, param in model.named_parameters()}

        for epoch in range(epochs):
            epoch_loss = 0.0
            for attr_name, attr_data in self.client_data.items():
                if attr_name not in self.models.models:
                    raise KeyError(f"Attribute {attr_name} is not found in the models.")

                total_correct = 0
                total_samples = 0
                # 针对当前属性逐行处理其列
                for col_idx, col_name in enumerate(attr_data.columns):
                    # 生成输入的 one-hot 向量
                    input_data = torch.zeros(len(attr_data.columns))  # 初始化为全零
                    input_data[col_idx] = 1  # 将当前列对应位置置为 1

                    # 获取对应的评分并转换为 one-hot 向量
                    score_bins = torch.arange(0, 10.5, 0.5)  # 定义分数的区间
                    score = attr_data.iloc[0, col_idx]  # 当前列的评分值

                    # 找到评分值所在的 bin 索引
                    bin_index = (score_bins == score).nonzero(as_tuple=True)[0].item()  # 得到索引
                    # 目标现在是类别的索引
                    target = bin_index  # 目标应该是整数类别索引，而不是 one-hot 向量
                    #print("target:", target)

                    # 将输入数据扩展为 (1, input_size) 的形状
                    input_data = input_data.unsqueeze(0)

                    # 获取对应模型和优化器
                    model = self.models.models[attr_name]
                    optimizer = self.optimizers[attr_name]

                    # 前向传播
                    outputs = model(input_data).squeeze()  # 输出为 (output_size,)
                    #print("outputs:", outputs)

                    # loss = F.mse_loss(outputs, target)  # 使用 MSE Loss 计算损失
                    loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
                    loss = loss_fn(outputs, torch.tensor(target))  # 将 target 作为类别索引传递
                    #print("torch.tensor(target):", torch.tensor(target))

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # #统计正确预测的行数
                    _, predicted = torch.max(outputs.data, 0)
                    correct = (predicted == torch.tensor(target)).item()

                    total_correct += correct
                    total_samples += 1

                    #保存每个属性的梯度信息
                    #创建一个字典，存储每个属性的梯度
                    if attr_name not in gradients:
                        gradients[attr_name] = {}

                    #将model转换为字符串，作为键
                    model_key = id(model)

                    #检查是否已经为该model初始化了字典
                    if model_key not in gradients[attr_name]:
                        gradients[attr_name][model_key] = {}

                    for name, param in model.named_parameters():
                        if(param.requires_grad):
                            gradients[attr_name][model_key][name] = param.grad.clone()

                    # 累计损失
                    epoch_loss += loss.item()

                # 输出调试信息
                #计算准确率
                attribute_accuracy = total_correct / total_samples * 100
                if self.client_id < 3:
                    print(f"Client {self.client_id}, Attr [{attr_name}], Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}， Accuracy: {attribute_accuracy:.2f}")
        
        #2. 计算W_b（训练后的参数）- W_a（训练前的参数）/ -n
        for attr_name, model in self.models.models.items():
            gradients[attr_name] = {}
            for name, param in model.named_parameters():
                if name in initial_params[attr_name]:
                    gradients[attr_name][name] = (param.data - initial_params[attr_name][name]) / (-learning_rate)
        #print("typeof gradients:", type(gradients))
        return gradients    #返回梯度信息


    def receive_attribute_graph(self, attribute_graph):
        """
        接收属性图
        :param attribute_graph: 属性图
        """
        def print_graph_details(graph):
            """
            打印图的详细信息
            :param graph: DGL 图对象
            """
            print(f"Graph Details:")
            print(f"Number of nodes: {graph.num_nodes()}")
            print(f"Number of edges: {graph.num_edges()}")
            
            orig_ids = graph.ndata['_ID'].tolist()
            # 打印边信息
            src_nodes, dst_nodes = graph.edges()
            edge_ids = graph.edge_ids(src_nodes, dst_nodes)
            
            print("\nEdge List:")
            for idx in range(graph.num_edges()):
                src_orig = orig_ids[src_nodes[idx].item()]
                dst_orig = orig_ids[dst_nodes[idx].item()]
                eid = edge_ids[idx].item()
                
                # 获取边的属性
                attributes = {k: graph.edata[k][eid].tolist() for k in graph.edata.keys()}
                
                print(f"Edge {eid}: {src_orig} -> {dst_orig}")
                print(f"    Attributes: {attributes}")
            
            # 打印节点特征（如有）
            if graph.ndata:
                print("\nNode Features:")
                for k in graph.ndata.keys():
                    print(f"Feature '{k}':")
                    print(graph.ndata[k])
            print(f" -------------------------------------------------")

        self.attribute_graph = attribute_graph
        print(f"Client {self.client_id} received attribute graph.")
        print_graph_details(attribute_graph)    
        
    def build_local_graph(self, client_id, items, neighbors):
        """
        用来为当前用户构建一个图，包含用户节点和物品节点，以及它们之间的连接（边）
        TODO: 添加新生成的attribute
        """
        G = dgl.DGLGraph()
        dic_user = {self.client_id: 0}
        dic_item = {}
        count = 1
        for n in neighbors:
            dic_user[n] =  count        #将该邻居的 ID 映射到图中一个新的节点 ID（由 count 指定）。每添加一个用户，count 的值会递增。
            count += 1
        for item in items:
            dic_item[item] = count
            count += 1
        G.add_edges([i for i in range(1, len(dic_user))], 0)        #建立了一个从所有邻居用户到当前用户的单向边，表示这些用户与当前用户之间的连接。
        G.add_edges(list(dic_item.values()), 0)     #创建一条边从所有物品节点（dic_item 中的节点 ID）指向当前用户节点（ID 为 0）。
        G.add_edges(0, 0)      #添加一个自环（self-loop），意味着用户自己与自己建立一条连接。`
        return G

    def user_embedding(self, embedding):
        """
        该方法返回两个张量：
        第一个张量：当前用户的邻居用户的嵌入向量，形状为 (len(self.neighbors), embed_size)。
        第二个张量：当前用户的嵌入向量，形状为 (embed_size,)。
        """
        return embedding[torch.tensor(self.neighbors)], embedding[torch.tensor(self.id_self)]

    def item_embedding(self, embedding):
        """
        该方法返回一个张量，包含当前用户交互过的物品的嵌入向量，形状为 (len(self.items), embed_size)。
        """
        return embedding[torch.tensor(self.items)]


    def local_train_gnn(self, embedding_user, embedding_item):
        embedding_user = torch.clone(embedding_user).detach()
        embedding_item = torch.clone(embedding_item).detach()
        embedding_user.requires_grad = True
        embedding_item.requires_grad = True
        embedding_user.grad = torch.zeros_like(embedding_user)
        embedding_item.grad = torch.zeros_like(embedding_item)

        self.graph_model.train()
        sampled_items, sampled_rating = self.negative_sample_item(embedding_item)
        returned_items = self.items + sampled_items
        predicted = self.GNN(embedding_user, embedding_item, sampled_items)
        loss = self.loss(predicted, sampled_rating)
        self.graph_model.zero_grad()
        loss.backward()
        model_grad = []
        for param in list(self.graph_model.parameters()):
            grad = self.LDP(param.grad)
            model_grad.append(grad)

        item_grad = self.LDP(embedding_item.grad[returned_items, :])
        returned_users = self.neighbors + [self.client_id]
        user_grad = self.LDP(embedding_user.grad[returned_users, :])
        res = (model_grad, item_grad, user_grad, returned_items, returned_users, loss.detach())
        return res
    
    def GNN(self, embedding_user, embedding_item, sampled_items):
        """
        图神经网络（GNN）中进行前向传播计算，并返回预测的评分。
        它通过从全局嵌入矩阵中提取用户和物品的嵌入向量，进行加权计算并输出预测值。
        """
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        items_embedding = self.item_embedding(embedding_item)
        sampled_items_embedding = embedding_item[torch.tensor(sampled_items)]

        # 将当前用户交互过的物品的嵌入（items_embedding）与负样本物品的嵌入（sampled_items_embedding）进行拼接，
        # 形成一个新的张量 items_embedding_with_sampled，它包含了正样本和负样本的物品嵌入。
        items_embedding_with_sampled = torch.cat((items_embedding, sampled_items_embedding), dim = 0)
        user_feature = self.graph_model(self_embedding, neighbor_embedding, items_embedding)
        predicted = torch.matmul(user_feature, items_embedding_with_sampled.t())
        self.user_feature = user_feature.detach()
        return predicted
    
    def update_local_GNN(self, global_model, rating_max, rating_min, embedding_user, embedding_item):
        self.model = copy.deepcopy(global_model)
        self.rating_max = rating_max
        self.rating_min = rating_min
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        if len(self.items) > 0:
            items_embedding = self.item_embedding(embedding_item)
        else:
            items_embedding = False
        user_feature = self.model(self_embedding, neighbor_embedding, items_embedding)
        self.user_feature = user_feature.detach()
    

    def negative_sample_item(self, embedding_item):
        item_num = embedding_item.shape[0]
        ls = [i for i in range(item_num) if i not in self.items]
        sampled_items = sample(ls, self.negative_sample)
        sampled_item_embedding = embedding_item[torch.tensor(sampled_items)]
        predicted = torch.matmul(self.user_feature, sampled_item_embedding.t())
        predicted = torch.round(torch.clip(predicted, min = self.rating_min, max = self.rating_max))
        return sampled_items, predicted


    def LDP(self, tensor):
        tensor_mean = torch.abs(torch.mean(tensor))
        tensor = torch.clamp(tensor, min = -self.clip, max = self.clip)
        noise = np.random.laplace(0, tensor_mean * self.laplace_lambda)
        tensor += noise
        return tensor

    def loss(self, predicted, sampled_rating):
        true_label = torch.cat((torch.tensor(self.ratings).to(sampled_rating.device), sampled_rating))
        return torch.sqrt(torch.mean((predicted - true_label) ** 2))

    def predict(self, item_id, embedding_user, embedding_item):
        self.graph_model.eval()
        item_embedding = embedding_item[item_id]
        return torch.matmul(self.user_feature, item_embedding.t())