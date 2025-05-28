import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

# 文件路径列表
file_paths = [
    '/home/zyj/FedNUR/data/useless/feat_preference_table_weighted_1.csv',
    '/home/zyj/FedNUR/data/useless/duration_category_preference_table_weighted_1.csv',
    # '/home/zyj/FedNUR/data/useless/influence_level_preference_table_weighted_1.csv',
    # '/home/zyj/FedNUR/data/useless/quality_level_preference_table_weighted_1.csv',
    # '/home/zyj/FedNUR/data/useless/video_type_preference_table_weighted_1.csv'
]

# 计算相似度并排序的函数
def calculate_similarity(file_path):
    # 读取数据
    df = pd.read_csv(file_path)

    # 选择前五个用户的数据
    df = df.head(5)

    # 获取用户的特征（去除 user_id 列）
    features = df.drop(columns=['user_id'])

    # # 标准化特征
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)

    # # 计算余弦相似度
    similarity_matrix = cosine_similarity(features)
    #similarity_matrix = euclidean_distances(features)

    # 将相似度矩阵转换为 DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=df['user_id'], columns=df['user_id'])

    # 将索引（user_id）转为列
    similarity_df_reset = similarity_df.reset_index()
    # 删除 user_id 列以避免与 reset_index() 产生冲突
    #similarity_df_without_user_id = similarity_df_reset.drop(columns='user_id')

    # 将索引（user_id）转为列
    similarity_df_reset = similarity_df.reset_index()

    # 这里将数据展开成一列，得到用户对比的相似度数据
    similarity_sorted = similarity_df_reset.stack().reset_index()

    # 重命名列以适配相似度的结构
    similarity_sorted.columns = ['user_id_1', 'user_id_2', 'similarity']
    
    # 去掉不必要的 user_id 字符串行，确保没有 'user_id' 的条目
    similarity_sorted = similarity_sorted[similarity_sorted['user_id_2'] != 'user_id']

    # 去掉重复对比的部分（确保user_id_1 < user_id_2）
    similarity_sorted = similarity_sorted[similarity_sorted['user_id_1'] < similarity_sorted['user_id_2']]
    #df['similarity'] = 1 / (1 + df['euclidean_distance'])
    # 根据相似度排序，从高到低
    similarity_sorted = similarity_sorted.sort_values(by='similarity', ascending=False)

    # 输出排序后的相似度



    return similarity_sorted

# 对每个文件计算并输出相似度
for file_path in file_paths:
    print(f"Calculating similarity for: {file_path}")
    similarity_result = calculate_similarity(file_path)
    similarity_result = similarity_result.reset_index(drop=True)
    print(similarity_result)
    print("="*50)
