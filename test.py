import os
import pandas as pd
import torch
import torch.nn.functional as F

from config import ATTRIBUTE_DATA_PATHS
from data_utils import load_attribute_data, distribute_all_attributes_to_clients
from client import Client


def test_config_paths():
    """
    测试 config 文件中的路径是否正确
    """
    for attribute, path in ATTRIBUTE_DATA_PATHS.items():
        assert os.path.exists(path), f"文件路径不存在：{path}"
    print("路径测试通过：所有路径均存在！")

def test_load_attribute_data():
    """
    测试加载数据是否正确
    """
    attribute_data = load_attribute_data(ATTRIBUTE_DATA_PATHS)

    # 验证是否加载了所有属性
    assert len(attribute_data) == len(ATTRIBUTE_DATA_PATHS), "加载的属性数量不正确！"

    for attribute, user_data in attribute_data.items():
        print(f"属性：{attribute}")
        print(f"用户数量：{len(user_data)}")

        # 验证第一个用户的数据
        first_user_id = list(user_data.keys())[0]
        first_user_data = user_data[first_user_id]
        print(f"用户ID：{first_user_id} 的数据：")
        print(first_user_data.head())
        assert isinstance(first_user_data, pd.DataFrame), "用户数据不是 DataFrame 格式！"

    print("数据加载测试通过！")

def test_distribute_data_to_clients():
    """
    测试数据分发是否正确
    """
    attribute_data = load_attribute_data(ATTRIBUTE_DATA_PATHS)

    clients_data = distribute_all_attributes_to_clients(attribute_data)
    attribute_name = list(attribute_data.keys())[0]
    # 验证客户端数量是否与用户数量一致
    if attribute_name in attribute_data:
        assert len(clients_data) == len(attribute_data[attribute_name]), "分发的客户端数量不正确！"

    first_client_id = list(clients_data.keys())[0]
    print(f"Client {first_client_id} data type: {type(clients_data[first_client_id])}")
    print(clients_data[first_client_id])

    print("数据分发测试通过！")

def test_client_initialization():
    """
    测试客户端的初始化和本地训练函数
    """
    # 示例数据
    client_id = 1
    user_data = {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6]
    }

    # 初始化客户端
    client = Client(client_id=client_id, client_data=pd.DataFrame(user_data))

    # 验证客户端 ID 是否正确
    assert client.client_id == client_id, "客户端 ID 初始化不正确！"


    # 测试本地训练逻辑
    print(f"测试客户端 {client_id} 的本地训练:")
    client.local_train()

    print("客户端初始化测试通过！")

def test_gradient_similarity(model1,model2):
    """
    测试梯度相似度计算
    """
    similarity = 0.0
    total_params = 0
    
    # 遍历两个模型的所有参数并计算它们的梯度相似度
    for (param1, param2) in zip(model1.parameters(), model2.parameters()):
        # 确保两个模型的参数形状一致
        if param1.grad is not None and param2.grad is not None:
            print("param1.grad: ", param1.grad)
            print("param2.grad: ", param2.grad)
            # 计算两个梯度之间的余弦相似度
            # grad1 = param1.grad.view(-1)  # 展平为一维张量
            # grad2 = param2.grad.view(-1)  # 展平为一维张量
            # cos_sim = F.cosine_similarity(grad1, grad2, dim=0)
            # similarity += cos_sim.item()
            # total_params += 1
    
    # 返回平均相似度
    return similarity / total_params if total_params > 0 else 0.0
    







if __name__ == "__main__":
    print("开始测试...")

    # 依次运行各个测试函数
    #test_config_paths()
    #test_load_attribute_data()
    #test_distribute_data_to_clients()
    test_client_initialization()

    print("client初始化测试通过！")
