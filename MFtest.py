#测试矩阵分解能否成功判断用户偏好相似度
#2024/12/30

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from numpy.linalg import norm

def perform_svd(matrix, n_components = 2):
    """
    对评分矩阵进行SVD分解。
    
    Args:
        matrix (np.array): 输入矩阵
        n_components (int): 保留的潜在特征维度
    
    Returns:
        U (np.array): 用户潜在特征矩阵
        Sigma (np.array): 奇异值矩阵
        VT (np.array): 物品潜在特征矩阵（转置）
    """
    # 从矩阵中分离出评分列和嵌入向量列
    matrix_array = np.array(matrix)
    ratings = matrix_array[:, 0]  # 评分列
    embeddings = matrix_array[:, 1:]  # 属性嵌入向量列
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(embeddings)  # 用户潜在特征
    Sigma = svd.singular_values_  # 奇异值
    VT = svd.components_  # 物品潜在特征
    
    return U, Sigma, VT



def calculate_similarity(U):
    """
    计算用户之间的偏好相似度，使用余弦相似度方法。

    Args:
        U (np.array): 用户潜在特征矩阵

    Returns:
        similarity_matrix (np.array): 用户之间的相似度矩阵
    """
    # 使用余弦相似度计算用户之间的相似度
    similarity_matrix = cosine_similarity(U)
    return similarity_matrix





# 创建数据
data = {
    "video_id": [0, 1, 2, 3],
    "feat": [[8], [27, 9], [9], [26]],
    "video_type": ["NORMAL", "NORMAL", "AD", "AD"],
    "visible_status": ["public", "public", "public", "public"],
    "influence_level": ["Medium", "Low", "Low", "Low"],
    "quality_level": ["Medium", "Low", "Medium", "Very Low"]
}

# 将数据转化为DataFrame
video_df = pd.DataFrame(data)

# # 用户评分矩阵videotype
# ratings_matrix = np.array([
#     [3, 4, 1, 1],  # 用户1
#     [1, 1, 5, 5],  # 用户2
#     [2, 5, 1, 1]   # 用户3
# ])

# 用户评分矩阵influence
ratings_matrix = np.array([
    [5, 5, 1, 1],  # 用户1喜欢普通
    [1, 1, 5, 5],  # 用户2喜欢广告
    [4, 4, 3, 3],   # 用户3偏好normal，偶尔ad
    [3, 3, 3, 3],  # 用户4平均一般喜欢每一个视频
    [5, 5, 5, 5],   # 用户5每个视频都超级喜欢
    [0, 0, 4, 4]   # 用户6完全不看normal，只喜欢ad
])

ratings_df = pd.DataFrame(ratings_matrix, columns=['Video 0', 'Video 1', 'Video 2', 'Video 3'])
print(ratings_df)

max_feat_category = 28

def generate_feat_embedding(feat):
    embedding = np.zeros(max_feat_category)  # 创建全0的向量
    for category in feat:
        embedding[category] = 1  # 用1标记该类别
    return embedding

# 为每个视频生成feat嵌入
feat_embeddings = [generate_feat_embedding(f) for f in data['feat']]

# 创建一个映射字典
video_type_map = {"NORMAL": [1, 0], "AD": [0, 1]}  # 例如，我们假设有NORMAL和COMEDY两种类型
video_type_embeddings = [video_type_map[v] for v in data["video_type"]]

# 创建一个映射字典
influence_map = {"Low": [1, 0, 0, 0], "Medium": [0, 1, 0, 0], "High": [0, 0, 1, 0], "Very High": [0, 0, 0, 1]}
influence_embeddings = [influence_map[v] for v in data["influence_level"]]

# 创建一个映射字典
quality_map = {"Very Low": [1, 0, 0, 0], "Low": [0, 1, 0, 0], "Medium": [0, 0, 1, 0], "High": [0, 0, 0, 1]}
quality_embeddings = [quality_map[v] for v in data["quality_level"]]

# for i in range(len(data["video_id"])):
#     print(f"Video {data['video_id'][i]} feat embedding: {feat_embeddings[i]}")
#     print(f"Video {data['video_id'][i]} video_type embedding: {video_type_embeddings[i]}")
#     print(f"Video {data['video_id'][i]} influence_level embedding: {influence_embeddings[i]}")
#     print(f"Video {data['video_id'][i]} quality_level embedding: {quality_embeddings[i]}")
#     print()  # 分隔每个视频的输出

#   所有用户的各属性评分矩阵
all_users_feat_matrix = []
all_users_video_type_matrix = []
all_users_influence_matrix = []
all_users_quality_matrix = []

#生成用户-评分矩阵
for user_id, user_ratings in enumerate(ratings_matrix):
    feat_matrix = []
    video_type_matrix = []
    influence_matrix = []
    quality_matrix = []
    for video_id, rating in enumerate(user_ratings):
        if rating != 0:  # 只考虑评分过的视频
            #视频类别
            combined_feat_row = [rating] + feat_embeddings[video_id].tolist()
            feat_matrix.append(combined_feat_row)
            #视频类型（是否广告）
            combined_video_type_row = [rating] + video_type_embeddings[video_id]
            video_type_matrix.append(combined_video_type_row)
            #视频影响力
            combined_influence_row = [rating] + influence_embeddings[video_id]
            influence_matrix.append(combined_influence_row)
            #视频质量
            combined_quality_row = [rating] + quality_embeddings[video_id]
            quality_matrix.append(combined_quality_row)
    
    all_users_feat_matrix.append(feat_matrix)
    all_users_video_type_matrix.append(video_type_matrix)
    all_users_influence_matrix.append(influence_matrix)
    all_users_quality_matrix.append(quality_matrix)

all_users_feat_matrix = [np.array(user_matrix) for user_matrix in all_users_feat_matrix]

all_users_video_type_matrix = [np.array(user_matrix) for user_matrix in all_users_video_type_matrix]

all_users_influence_matrix = [np.array(user_matrix) for user_matrix in all_users_influence_matrix]


# print("all_users_video_type_matrix type:", type(all_users_video_type_matrix))
# for idx, matrix in enumerate(all_users_video_type_matrix):
#     print(f"User {idx + 1} matrix type: {type(matrix)}")
#     print(matrix)


combined_feat_matrix = []
combined_video_type_matrix = []
combined_influence_matrix = []
combined_quality_matrix = []

combined_feat_matrix = np.vstack(all_users_feat_matrix)  # 合并成一个大矩阵
combined_video_type_matrix = np.vstack(all_users_video_type_matrix)
combined_influence_matrix = np.vstack(all_users_influence_matrix)
combined_quality_matrix = np.vstack(all_users_quality_matrix)

#print("combined_quality_matrix:")
#print(combined_quality_matrix)

U_video_type, Sigma_video_type, VT_video_type = perform_svd(combined_video_type_matrix)
# print("===== Combined Feat Matrix SVD =====")
# print("User Latent Features (U):")
# print(U_feat)
# print("Item Latent Features (VT):")
# print(VT_feat)


U_influence, Sigma_influence, VT_influence = perform_svd(combined_influence_matrix)

user_features = []
start_idx = 0
for user_idx, user_matrix in enumerate(all_users_video_type_matrix):
    num_interactions = user_matrix.shape[0]  # 当前用户的交互数量
    user_feature = U_influence[start_idx:start_idx + num_interactions].mean(axis=0)  # 按行求均值
    user_features.append(user_feature)
    start_idx += num_interactions
user_features = np.array(user_features)  # 转为 NumPy 数组




influence_similarity = calculate_similarity(user_features)
#print("\n=====video type Similarity Between All Users =====")
#print(influence_similarity)


'''
比较用户嵌入矩阵分解后相似度
'''



U_matrices = []
for user_matrix in all_users_video_type_matrix:
    U, S, V = np.linalg.svd(user_matrix, full_matrices=False)  # 执行 SVD 分解
    U_matrices.append(U)  # 保存 U 矩阵

# for i, U in enumerate(U_matrices):
#     print(f"User {i + 1} U matrix:")
#     print(U)

# 比较 U 矩阵相似度 (使用余弦相似度)
def cosine_similarity(mat1, mat2):
    mat1_flat = mat1.flatten()
    mat2_flat = mat2.flatten()
    return 1 - cosine(mat1_flat, mat2_flat)  # 余弦相似度


user_factor = np.array([
    [5, 4.2, 1, 0],  # 用户1
    [5, 4.2, 4, 0],
    [5, 3, 1, 0],
    [5, 4, 1, 0],
    [5, 1, 4.8, 0]

])

result_matrix = np.array([np.outer(user, user) for user in user_factor])
print(result_matrix)
for idx, matrix in enumerate(result_matrix):
    U, S, Vt = np.linalg.svd(matrix)
    print(f"Matrix {idx + 1} SVD:")
    print(f"U:\n{U}")
    print(f"S (singular values):\n{S}")
    print(f"Vt (transpose of V):\n{Vt}")
    print("=" * 50)


u1 = np.array([[-0.76325016, -0.11503423], [-0.02284474,  0.9882122]])
u2 = np.array([[-0.15111387, -0.8224026], [-0.70609146,  0.47810412]])
u3 = np.array([[-0.84189165, -0.18459575], [-0.11287465,  0.97912608]])
u4 = np.array([[-0.74848119, -0.24253563], [-0.1871203,   0.9701425]])
u5 = np.array([[-0.1605185,  -0.97704962], [-0.68170115,  0.2123381]])

# 计算余弦相似度的函数
# def cosine_similarity(v1, v2):
#     dot_product = np.dot(v1, v2)  # 点积
#     norm_v1 = np.linalg.norm(v1)  # 向量v1的范数
#     norm_v2 = np.linalg.norm(v2)  # 向量v2的范数
#     return dot_product / (norm_v1 * norm_v2)

def oushi_disitance(u1, u2):
    distance = norm(u1 - u2)  # 计算两个向量之间的欧氏距离
    similarity = 1 / (1 + distance)  # 通过距离转换为相似度
    return similarity

def matrix_euclidean_similarity(matrix1, matrix2):
    # 确保两个矩阵的形状一致
    if matrix1.shape != matrix2.shape:
        raise ValueError("两个矩阵的形状必须相同")
    
    # 计算矩阵元素之间的欧氏距离
    distance = norm(matrix1 - matrix2)  # 计算两个矩阵元素之间的欧氏距离
    similarity = 1 / (1 + distance)  # 通过欧氏距离转换为相似度
    return similarity

# 计算任意两个向量之间的相似度
similarity_1_2 = matrix_euclidean_similarity(u1, u2)
similarity_1_3 = matrix_euclidean_similarity(u1, u3)
similarity_1_4 = matrix_euclidean_similarity(u1, u4)
similarity_2_3 = matrix_euclidean_similarity(u2, u3)
similarity_2_4 = matrix_euclidean_similarity(u2, u4)
similarity_3_4 = matrix_euclidean_similarity(u3, u4)
similarity_1_5 = matrix_euclidean_similarity(u1, u5)
similarity_2_5 = matrix_euclidean_similarity(u2, u5)
similarity_3_5 = matrix_euclidean_similarity(u3, u5)
similarity_4_5 = matrix_euclidean_similarity(u4, u5)


# 打印结果
print(f"Similarity between u1 and u2: {similarity_1_2}")
print(f"Similarity between u1 and u3: {similarity_1_3}")
print(f"Similarity between u1 and u4: {similarity_1_4}")
print(f"Similarity between u2 and u3: {similarity_2_3}")
print(f"Similarity between u2 and u4: {similarity_2_4}")
print(f"Similarity between u3 and u4: {similarity_3_4}")
print(f"Similarity between u1 and u5: {similarity_1_5}")
print(f"Similarity between u2 and u5: {similarity_2_5}")
print(f"Similarity between u3 and u5: {similarity_3_5}")
print(f"Similarity between u4 and u5: {similarity_4_5}")