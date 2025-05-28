from data_utils import load_attribute_data, distribute_all_attributes_to_clients
from client import Client

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

import config
import torch
import numpy as np
import dgl

class Server:
    def __init__(self):
        self.clients = []
    
    def load_and_distribute_data(self):     
        # Step 1: 读取和拆分数据
        attribute_data = load_attribute_data(config.ATTRIBUTE_DATA_PATHS)
        
        # Step 2: 分发数据到客户端
        clients_data = distribute_all_attributes_to_clients(attribute_data, config.USER_ITEM_PATHS)
        return clients_data

    def initialize_clients(self, clients_data):      
        # Step 3: 初始化客户端
        
        clients = []
        for client_id, client_data in clients_data.items():  # 遍历字典的键和值
            attribute_sizes = {
                attr_name: client_data['attribute'][attr_name].shape[1]  # 获取每个属性列的数量
                for attr_name in client_data['attribute']
            }
            learning_rate = 0.003  # 可以为所有客户端统一设定，也可以根据需要动态调整
            items = client_data['items']
            ratings = client_data['rating']
            client = Client(client_id, client_data, attribute_sizes, items, ratings, learning_rate)  # client_id 是 user_id
            clients.append(client)
        self.clients = clients
    
    def simulate_local_training(self):

        nn_gradients = {}
        # Step 4: 模拟客户端本地训练
        for client in self.clients[:5]:
            client_gradients = client.local_train_nn()
            nn_gradients[client.client_id] = client_gradients
            if(client.client_id <= 5):
                print(f"Receive client {client.client_id} gradient")
        
        # Step 5: 计算属性偏好梯度相似度
        similarity_result = self.calculate_similarity(nn_gradients)
        #print("Type of similarity_results:", type(similarity_result))
        #print("Content of similarity_results:", similarity_result)

        attribute_include = ['feat', 'duration']
        similarity_result_select = {}

        for attribute in attribute_include:
            if attribute in similarity_result:
                similarity_result_select[attribute] = similarity_result[attribute]
        # Step 6: 构建用户之间的相似度关系图
        user_attr_graph = self.user_attribute_graph(similarity_result_select)

        # Step 7: 分发用户属性图到客户端
        self.distribute_user_attribute_graph(user_attr_graph)

        #打印相似度结果
        # for attribute, results in similarity_result.items():
        #     print(f"Similarity results for attribute: {attribute}")
        #     sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        #     # 遍历每对客户端的相似度结果
        #     for (client_id_i, client_id_j), similarity_value in sorted_results:
        #         print(f"  Client {client_id_i} vs Client {client_id_j}: {similarity_value:.4f}")
            
        #     print()  # 空行分隔不同属性的结果


    def distribute_global_model(self):      #待修改
        """
        
        将全局模型分发给所有客户端
        """
        for client in self.client_list:
            client.global_model = copy.deepcopy(self.global_model)
    
    def aggregate_local_models(self):       #待修改
        """
        聚合所有客户端的局部模型
        """
        for client in self.client_list:
            self.global_model = client.local_model
    
    def update_global_model(self):      #待修改
        """
        更新全局模型
        """
        # 更新全局模型
        self.aggregate_local_models()
        # 更新全局模型参数
        self.global_model.update_parameters(self.lr, self.weight_decay)
    
    def train(self, epochs=10):     #待修改
        """
        训练全局模型
        """
    
    def calculate_similarity(self, gradients): 
        """
        计算所有客户端的梯度相似度
        """
        similarity_results = {}

        #获取属性名称
        attributes = list(gradients[next(iter(gradients))].keys())

        # 只选择前五个客户端
        selected_clients = list(gradients.items())[:5]  # 获取前5个客户端数据

        #遍历每个属性
        for attribute in attributes:
            #获取所有客户端该属性的梯度
            attribute_gradients = {}
            for client_id, client_gradients in selected_clients:
                grad_vector = []

                #遍历该客户端的所有梯度参数
                for param_name, grad in client_gradients[attribute].items():
                    if isinstance(grad, torch.Tensor):
                        grad_vector.extend(grad.flatten().cpu().numpy())
                
                attribute_gradients[client_id] = grad_vector
            
            #转换成numpy数组，准备计算相似度
            gradient_matrix = np.array(list(attribute_gradients.values()))

            #计算余弦相似度
            similarity_matrix = cosine_similarity(gradient_matrix)

            #储存相似度结果
            similarity_results[attribute] = {}

            #遍历每对客户端，存储相似度
            client_ids = list(attribute_gradients.keys())
            for i in range(len(client_ids)):
                for j in range(i + 1, len(client_ids)):
                    similarity_results[attribute][(client_ids[i], client_ids[j])] = similarity_matrix[i, j]


        return similarity_results

    

    def user_attribute_graph(self, similarity_results, top_x_percent = 60, global_top_y_percent = 50):
        """
        构建用户之间的相似度关系图，使用 DGL 构建图形结构。
        边的权重为相似属性数量，并记录具体属性。

        :param similarity_results: 每个属性的相似度结果，字典形式：attribute -> {(user1, user2): similarity}
        :param top_x_percent: 每个用户保留的相似邻居的百分比
        :param global_top_y_percent: 全局相似度排序中保留的前百分比
        :return: DGLGraph 对象，用户关系图
        """

        def process_user_graphs(user_graphs):
            """
            按照每个属性的总图分割出每个用户的图
            :param user_graphs: 用户的属性总图字典
            :return: 每个用户在每个属性上的图的字典
            """
            user_attribute_graphs = {}
            attr_to_idx = {attr: idx for idx, attr in enumerate(user_graphs.keys())}
            print("Attribute to Index Mapping:", attr_to_idx)

            for attribute, graph in user_graphs.items():
                # 遍历每个节点，为该用户提取该属性下的子图
                for i in range(graph.num_nodes()):
                    # if i not in user_attribute_graphs:
                    #     user_attribute_graphs[i] = {}
                    # 提取子图
                    src, dst = graph.out_edges(i)
                    eids = graph.edge_ids(src, dst)

                    if len(eids) > 0:
                        
                        sub_graph = graph.edge_subgraph(
                            edges = eids,
                            relabel_nodes = False,
                        )
                        sub_graph.edata['weight'] = torch.ones(sub_graph.num_edges(), dtype = torch.long)

                        attr_code = 1 << attr_to_idx[attribute]
                        sub_graph.edata['attrs_code'] = torch.full((sub_graph.num_edges(),),  attr_code, dtype = torch.long)

                        nx_graph = sub_graph.to_networkx(edge_attrs = ['attrs_code', 'weight'])
                        #print("Sub Graph:", nx_graph.edges(data = True))

                        if i not in user_attribute_graphs:
                            user_attribute_graphs[i] = nx_graph
                        else:
                            for u, v, data in nx_graph.edges(data = True):
                                #print("data:", data)
                                if user_attribute_graphs[i].has_edge(u, v):
                                    # print("User Attribute Graphs:", user_attribute_graphs[i][u][v])
                                    # print("Data:", data['attrs_code'])
                                    # print("User Attribute Graphs:", user_attribute_graphs[i][u][v][0]['attrs_code'])
                                    
                                    user_attribute_graphs[i][u][v][0]['attrs_code'] |= data['attrs_code']
                                    #print("Merged User Attribute Graphs:", user_attribute_graphs[i][u][v][0]['attrs_code'])
                                    user_attribute_graphs[i][u][v][0]['weight'] += data['weight']
                                else:
                                    user_attribute_graphs[i].add_edge(u, v, **data)

            for user_id in user_attribute_graphs:
                nx_graph = user_attribute_graphs[user_id]
                
                # 收集边相关节点（自动去重）
                connected_nodes = sorted({node for edge in nx_graph.edges() for node in edge})
                
                # 创建DGL图时保留原始节点ID
                g = dgl.DGLGraph()
                g.add_nodes(len(connected_nodes))
                g.ndata['_ID'] = torch.tensor(connected_nodes, dtype=torch.long)
                
                # 建立双向映射字典
                orig2idx = {orig: idx for idx, orig in enumerate(connected_nodes)}
                
                # 添加边（需要转换到新索引空间）
                if nx_graph.edges():
                    src = [orig2idx[u] for u, v in nx_graph.edges()]
                    dst = [orig2idx[v] for u, v in nx_graph.edges()]
                    g.add_edges(src, dst)
                    
                    # 保持边特征
                    edge_attrs = {key: torch.tensor([data[key] for _, _, data in nx_graph.edges(data=True)])
                                for key in ['weight', 'attrs_code']}
                    for key in edge_attrs:
                        g.edata[key] = edge_attrs[key]
                
                user_attribute_graphs[user_id] = g
                print(f"User {user_id} Graph:")
                #print_graph_details(user_attribute_graphs[user_id])
            print(user_attribute_graphs)
            return user_attribute_graphs
 

        attr_encoding = {attribute: idx for idx, attribute in enumerate(similarity_results.keys())}
        #print("Attribute Encoding:", attr_encoding)
        #1. 计算全局相似度的前global_top_y_percent 边】

        user_graphs = {}
        all_edges = []
        for attribute, similarity_dict in similarity_results.items():
            for (user1, user2), similarity in similarity_dict.items():
                all_edges.append((user1, user2, similarity, attribute))
            # 按照相似度排序
            all_edges_sorted = sorted(all_edges, key = lambda x: x[2], reverse = True)

            # 取前global_top_y_percent 边
            global_top_y_count = int(len(all_edges_sorted) * global_top_y_percent / 100)
            top_global_edges = all_edges_sorted[:global_top_y_count]

            # 2. 根据每个用户的前top_x_percent 筛选边
            user_neigh_limit = {}
            for user1, user2, similarity, attribute in top_global_edges:
                if user1 not in user_neigh_limit:
                    user_neigh_limit[user1] = {}
                if user2 not in user_neigh_limit:
                    user_neigh_limit[user2] = {}

                if user2 not in user_neigh_limit[user1]:
                    user_neigh_limit[user1][user2] = []
                if user1 not in user_neigh_limit[user2]:
                    user_neigh_limit[user2][user1] = []
                
                user_neigh_limit[user1][user2].append(attribute)
                user_neigh_limit[user2][user1].append(attribute)
            
            
            #过滤每个用户的前top_x_percent 相似邻居
            for user in user_neigh_limit:
                top_x_count = int(len(user_neigh_limit[user]) * top_x_percent / 100)
                #根据边的相似度数量排序，并保留top_x_percent的相似邻居
                user_neigh_limit[user] = sorted(user_neigh_limit[user].items(), key = lambda x: len(x[1]), reverse = True)[:top_x_count]
            #print("user_neigh_limit:", user_neigh_limit)
            #构建图
            G = dgl.DGLGraph()

            #添加节点
            #dic_user = {user: user for user in user_neigh_limit}
            

            #为图添加边
            edges_src = []
            edges_dst = []
            edge_weights = []
            edge_attributes = []

            # 属性类型编码：


            #添加用户到用户的边
            for user in user_neigh_limit:
                for neighbor, attributes in user_neigh_limit[user]:
                    edges_src.append(user)
                    edges_dst.append(neighbor)
                    edge_weights.append(len(attributes))
                    #edge_attributes.append(attributes)

                    encoded_attributes = [attr_encoding[attr] for attr in attributes]
                    edge_attributes.append(encoded_attributes)

            #添加图的边
            G.add_edges(edges_src, edges_dst)
            #print(edge_attributes)
            #edge_attribute_tensor = torch.tensor([torch.tensor(attr) for attr in edge_attributes])

            G.edata['similarity_count'] = torch.tensor(edge_weights)
            G.edata['attributes'] = torch.tensor(edge_attributes)

            user_graphs[attribute] = G

        user_attr_graph = process_user_graphs(user_graphs)
        return user_attr_graph
    
    
                
    def distribute_user_attribute_graph(self, user_attribute_graphs):
        """
        将每个用户的属性图分发到对应客户端
        :param user_attribute_graphs: 每个用户的属性图字典 {user_id: graph}
        :return: 每个客户端的分发数据字典 {client_id: {属性名称: 数据}}
        """
        for user_id in user_attribute_graphs:
            for client in self.clients:
                if client.client_id == user_id:
                    # 将图分发到客户端
                    client.receive_attribute_graph(user_attribute_graphs[user_id])



    def print_graph_details(graph):
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




