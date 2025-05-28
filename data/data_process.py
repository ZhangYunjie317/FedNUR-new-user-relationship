import pandas as pd
import numpy

#把watchratio转化成rating
def generate_user_item():
    df = pd.read_csv('/home/zyj/FedNUR/KuaiRec/big_matrix_1.csv')

    def watch_ratio_to_score(ratio):
        if ratio <= 0.3:
            return 1
        elif ratio <= 0.6:
            return 2
        elif ratio <= 1.0:
            return 3
        elif ratio <= 1.5:
            return 4
        else:
            return 5
    
    df['score'] = df['watch_ratio'].apply(watch_ratio_to_score)
    df = df.drop(columns=['watch_ratio'])
    df.to_csv('/home/zyj/FedNUR/KuaiRec/big_matrix_2.csv', index=False)
    print("新文件已保存为 big_matrix_2.csv")

#计算平均influence和quality
def calculate_avarage_factor(df):
    df_last_30 = df.groupby('video_id').tail(30)  
    # 放大 influence_factor 和 quality_factor 的值
    df_last_30['influence_factor'] *= 100
    df_last_30['quality_factor'] *= 100
    result = df_last_30.groupby('video_id').agg({
        'influence_factor': 'mean',  # 计算 influence_factor 的平均值
        'quality_factor': 'mean',   # 计算 quality_factor 的平均值
        'video_type': 'first',      # 获取每组中 video_type 的第一个值（假设相同）
        'visible_status': 'first'   # 获取每组中 visible_status 的第一个值（假设相同）
    }).reset_index()

    result.to_csv('/home/zyj/FedNUR/KuaiRec/item_daily_features_2.csv', index=False)
    print("保存到item_daily_features_2。")
#生成item属性表
def generate_item_attribute():
    df = pd.read_csv('/home/zyj/FedNUR/KuaiRec/item_daily_features.csv')
    columns_to_drop = ['date', 'author_id', 'upload_dt', 'upload_type', 'video_duration',
                   'video_width', 'video_height', 'music_id', 'video_tag_id', 'video_tag_name']
    df = df.drop(columns=columns_to_drop)

    print(f"已成功保存到新文件。")

    df = pd.read_csv('/home/zyj/FedNUR/KuaiRec/item_daily_features_1.csv')
    #df['play_cnt'] = df['play_cnt'].replace(0, 1)
    # df['play_cnt'] = df['play_cnt'].replace(0, None)  # 将 play_cnt 为 0 的数据标记为缺失值

    #计算影响力
    w1 = 0.3    #曝光权重
    w2 = 0.6    #互动权重
    w3 = 0.2    #反馈权重

    alpha = 0.5     #点赞
    beta = 1.0      #评论
    gamma = 1.2     #关注
    delta = 1.0     #关注
    eta = 0.8       #分享
    lambda_ = 1.0   #下载
    mu = 2.0        #举报

    df['influence_factor'] = None  # 初始化列
    valid_rows = (df['show_cnt'] > 0) & (df['play_cnt'] > 0)   
    df.loc[valid_rows, 'influence_factor'] = (
        w1 * (df.loc[valid_rows, 'play_cnt'] / df.loc[valid_rows, 'show_cnt']) +  # 曝光转化率
        w2 * (
            alpha * (df.loc[valid_rows, 'like_cnt'] / df.loc[valid_rows, 'play_cnt']) +
            beta * (df.loc[valid_rows, 'comment_cnt'] / df.loc[valid_rows, 'play_cnt']) +
            gamma * (df.loc[valid_rows, 'follow_cnt'] / df.loc[valid_rows, 'play_cnt']) +
            delta * (df.loc[valid_rows, 'share_cnt'] / df.loc[valid_rows, 'play_cnt']) +
            eta * (df.loc[valid_rows, 'download_cnt'] / df.loc[valid_rows, 'play_cnt']) +
            lambda_ * (df.loc[valid_rows, 'collect_cnt'] / df.loc[valid_rows, 'play_cnt'])
        ) +
        w3 * (
            - mu * (df.loc[valid_rows, 'report_cnt'] / df.loc[valid_rows, 'play_cnt'])
        )
    )

    valid_rows_1 = (df['show_cnt'] > 0) & (df['play_cnt'] == 0)  
    df.loc[valid_rows_1, 'influence_factor'] = (
        w1 * (df.loc[valid_rows_1, 'play_cnt'] / df.loc[valid_rows_1, 'show_cnt'])  # 曝光转化率       
    )

    valid_rows_2 = (df['show_cnt'] == 0) &(df['play_cnt'] != 0)
    df.loc[valid_rows_2, 'influence_factor'] = (
        w2 * (
            alpha * (df.loc[valid_rows_2, 'like_cnt'] / df.loc[valid_rows_2, 'play_cnt']) +
            beta * (df.loc[valid_rows_2, 'comment_cnt'] / df.loc[valid_rows_2, 'play_cnt']) +
            gamma * (df.loc[valid_rows_2, 'follow_cnt'] / df.loc[valid_rows_2, 'play_cnt']) +
            delta * (df.loc[valid_rows_2, 'share_cnt'] / df.loc[valid_rows_2, 'play_cnt']) +
            eta * (df.loc[valid_rows_2, 'download_cnt'] / df.loc[valid_rows_2, 'play_cnt']) +
            lambda_ * (df.loc[valid_rows_2, 'collect_cnt'] / df.loc[valid_rows_2, 'play_cnt'])
        ) +
        w3 * (
            - mu * (df.loc[valid_rows_2, 'report_cnt'] / df.loc[valid_rows_2, 'play_cnt'])
        )
    )

    df['influence_factor'] = df['influence_factor'].fillna(0)
    print("influence_factor计算完成")
    print(df[['video_id', 'influence_factor']].head())


    #计算视频质量
    w4 = 0.3    #有效播放
    w5 = 0.25   #完全播放
    w6 = 0.2    #长时间播放
    w7 = 0.2    #短时间播放
    w8 = 0.15   #播放进程

    df['quality_factor'] = None  # 初始化列
    df.loc[valid_rows,'quality_factor'] = (
        w4 * (df.loc[valid_rows,'valid_play_cnt'] / df.loc[valid_rows,'play_cnt']) +
        w5 * (df.loc[valid_rows,'complete_play_cnt'] / df.loc[valid_rows,'play_cnt']) +
        w6 * (df.loc[valid_rows,'long_time_play_cnt'] / df.loc[valid_rows,'play_cnt']) -
        w7 * (df.loc[valid_rows,'short_time_play_cnt'] / df.loc[valid_rows,'play_cnt']) +
        w8 * df.loc[valid_rows,'play_progress']
    )
    df['quality_factor'] = df['quality_factor'].fillna(0)
    print("quality_factor计算完成")
    print(df[['video_id', 'influence_factor', 'quality_factor']].head())

    #判断计算出的因子是否成功
    user_input = input("是否继续执行？输入 '#' 表示继续，其他内容退出： ")
    if user_input == "#":
        columns_to_keep = ['video_id', 'video_type', 'visible_status', 'influence_factor', 'quality_factor']
        df_filtered = df[columns_to_keep]
        calculate_avarage_factor(df_filtered)
        
    else:
        output_file = '/home/zyj/FedNUR/KuaiRec/item_daily_features_temp.csv'
        df.to_csv(output_file, index = False)
        print("保存到临时文件，用户选择退出程序。")
        exit()

#generate_item_attribute()

def divide_factor():
    df = pd.read_csv('/home/zyj/FedNUR/data/item.csv')

    df['influence_level'] = pd.cut(
        df['influence_factor'], 
        bins=[0, 10, 20, 50, float('inf')],  # 划分区间
        labels=['Low', 'Medium', 'High', 'Very High'],  # 对应标签
        right=False  # 区间左闭右开
    )
    df['quality_level'] = pd.cut(
        df['quality_factor'], 
        bins=[-20, 0, 20, 50, float('inf')],  # 划分区间
        labels=['Very Low', 'Low', 'Medium', 'High'],  # 对应标签
        right=False  # 区间左闭右开
    )

    # 查看结果
    print(df[['influence_factor', 'influence_level', 'quality_factor', 'quality_level']].head())

    columns_to_exclude = ['influence_factor', 'quality_factor']
    df_new = df[[col for col in df.columns if col not in columns_to_exclude]]

    # 保存为新文件
    df_new.to_csv('/home/zyj/FedNUR/data/item_2.csv', index=False)


# divide_factor()
    
def process_scores():
    """
    将指定的CSV文件中的分数处理为靠近最近的分数(0, 0.5, 1, 1.5, ...)。
    处理后的文件将保存到新文件中。
    
    参数：
        file_paths (list of str): 待处理的CSV文件路径列表。
        output_suffix (str): 输出文件名的后缀，默认为"_processed"。
        
    返回：
        list of str: 处理后文件的路径列表。
    """
    def round_to_nearest_score(value):
        if 0 < value <= 0.125:  # 特殊处理 0 到 0.125 区间的值
            return 0.25
        else:
            return round(value * 2) / 2  # 四舍五入到最近的 0.25

    # file_paths = ["/home/zyj/FedNUR/data/useless/feat_preference_table_weighted_1.csv", 
    #               "/home/zyj/FedNUR/data/useless/influence_level_preference_table_weighted_1.csv", 
    #               "/home/zyj/FedNUR/data/useless/quality_level_preference_table_weighted_1.csv", 
    #               "/home/zyj/FedNUR/data/useless/video_type_preference_table_weighted_1.csv"] 
    file_paths = ["/home/zyj/FedNUR/data/useless/duration_category_preference_table_weighted_1.csv"]

    processed_files = []  # 用于保存处理后的文件路径
    
    for file in file_paths:
        try:
            # 读取文件
            df = pd.read_csv(file)
            
            # 对除第一列以外的列进行分数归属
            for col in df.columns[1:]:
                df[col] = df[col].apply(round_to_nearest_score)
            
            output_suffix = "_processed"
            # 保存处理后的数据到新文件
            output_file = file.replace(".csv", f"{output_suffix}.csv")
            df.to_csv(output_file, index=False)
            
            print(f"Processed file saved as: {output_file}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

process_scores()


def process_video_duration():
    # df = pd.read_csv('/home/zyj/FedNUR/KuaiRec/item_daily_features.csv')

    # df_extracted = df[['video_id', 'video_duration']].drop_duplicates(subset = 'video_id', keep = 'first')
    # output_file = '/home/zyj/FedNUR/KuaiRec/unprocessed_video_duration.csv'
    # df_extracted.to_csv(output_file, index=False)

    # print(f"文件已保存为{output_file}，包含视频ID和视频时长。")
    df = pd.read_csv('/home/zyj/FedNUR/KuaiRec/unprocessed_video_duration.csv')
    df['video_duration'].fillna(0, inplace=True)  # 将缺失值填充为0
    df['video_duration'] = df['video_duration'] / 1000

    min_duration = df['video_duration'].min()
    max_duration = df['video_duration'].max()
    mode_duration = df['video_duration'].mode()[0]
    avg_duration = df['video_duration'].mean()

    print(f"视频时长的最小值: {min_duration}")
    print(f"视频时长的最大值: {max_duration}")
    print(f"视频时长的众数: {mode_duration}")
    print(f"视频时长的平均值: {avg_duration}")

    bins = [0, 10, 30, 60, float('inf')]
    labels = ['Short', 'Medium', 'Long', 'Very Long']
    df['duration_category'] = pd.cut(df['video_duration'], bins=bins, labels=labels, right=False)
    
    output_file = '/home/zyj/FedNUR/KuaiRec/video_duration_processed.csv'
    df.to_csv(output_file, index=False)
    print(f"文件已保存为{output_file}，包含视频ID和视频时长。")

#process_video_duration()
    
def merge_user_video_duration():
    # 读取 CSV 文件
    df1 = pd.read_csv('/home/zyj/FedNUR/data/user_item.csv')
    df2 = pd.read_csv('/home/zyj/FedNUR/KuaiRec/video_duration_processed.csv')

    # 打印数据框的前几行
    print("df1:")
    print(df1.head())
    print("df2:")
    print(df2.head())

    # 合并数据框
    merged_df = pd.merge(df1, df2[['video_id', 'duration_category']], on='video_id', how='left')

    # 打印合并后的数据框的前几行
    print("Merged DataFrame:")
    print(merged_df.head())

    # 保存合并后的数据框到新的 CSV 文件
    output_file = '/home/zyj/FedNUR/data/user_video_duration.csv'
    merged_df.to_csv(output_file, index=False)

#merge_user_video_duration()