import pandas as pd
import math


def determine_client_num(user_data):
    """
    确定客户端数量，直接等于用户数量
    :param user_data: 每个用户的数据字典
    :return: 客户端数量
    """
    return len(user_data)  # 客户端数量等于用户数量

# 自动加载所有属性文件，按用户分组
def load_attribute_data(attribute_paths):
    """
    根据属性路径动态加载数据
    :param attribute_paths: 属性文件路径字典 {属性名称: 文件路径}
    :return: 每个属性的用户数据字典 {属性名称: {user_id: data}}
    """
    attribute_data = {}
    for attribute_name, file_path in attribute_paths.items():
        # 读取每个属性的数据文件并按用户分组
        data = pd.read_csv(file_path)
        user_data = {}
        for user_id, group in data.groupby('user_id'):
            user_data[user_id] = group.drop(columns=['user_id'])  # 去掉 user_id 列
        attribute_data[attribute_name] = user_data
    return attribute_data

def distribute_all_attributes_to_clients(attribute_data, user_item_paths):
    """
    将所有属性的数据分发到对应客户端
    :param attribute_data: 每个属性的用户数据字典 {属性名称: {user_id: data}}
    :return: 每个客户端的分发数据字典 {client_id: {属性名称: 数据}}
    """
    clients_data = {}

    for attribute_name, user_data in attribute_data.items():
        for user_id, data in user_data.items():
            if user_id not in clients_data:
                clients_data[user_id] = {'attribute':{}}  # 初始化每个客户端数据
            clients_data[user_id]['attribute'][attribute_name] = data  # 添加属性数据到对应客户端
    
    # 读取用户-物品评分数据
    user_item_data = pd.read_csv(user_item_paths)
    ratings_group = (
        user_item_data
        .groupby('user_id')
        .agg({'video_id':list, 'score':list})
        .to_dict(orient = 'index')
    )
    for user_id, client_dict in clients_data.items():
        user_r = ratings_group.get(user_id, {'video_id':[], 'score':[]})
        client_dict['items'] = user_r['video_id']
        client_dict['rating'] = user_r['score']
    #print("clients_data: ", clients_data)

    return clients_data

