import pandas as pd
import ast
import os

def detect_complex_columns(df):
     complex_columns = []

     #遍历每一列
     for col in df.columns:
          #检测是否包含复杂数据
          has_complex_data = False
          for value in df[col]:
               if isinstance(value, str) and ('[' in value or '{' in value):
                    try:
                         parsed_value = ast.literal_eval(value)
                         if isinstance(parsed_value, list):
                              has_complex_data = True
                              print(f"列 '{col}' 检测到复杂数据类型：list，准备按行展开...")
                              # 解析为列表类型
                              df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and ('[' in x and ']' in x) else x)
                              df = df.explode(col, ignore_index = True)  # 按行展开该列
                              break
                         elif isinstance(parsed_value, dict):
                              has_complex_data = True
                              print(f"列 '{col}' 检测到复杂数据类型：dict，后续需要特殊处理...")
                              break
                    except (ValueError, SyntaxError):
                         pass
               
          if has_complex_data:
               complex_columns.append(col)
          
     print(complex_columns)
     return df

def custom_aggregation(series, df, feature_col, user_id_col):
     """
     自定义聚合函数，计算用户属性偏好表。
     :param df: 输入的 DataFrame
     :param feature_col: 特征列
     :param score_col: 评分列
     :return: 计算后的加权评分
     """
     #计算特征的平均评分

     print("series:", series)

     avg_score = series.mean()
     print("avg_score:", avg_score)

     user_score_count = df[df[user_id_col] == series.name].shape[0]
     print("user_score_count:", user_score_count)

     feature_counts = series.count()
     print("feature_counts:", feature_counts)


     #计算该特征的比例
     feature_ratio = feature_counts / user_score_count
     print("feature_ratio:", feature_ratio)

     # 使用比例进行加权评分
     weighted_score = avg_score + avg_score * feature_ratio
     print("weighted_score:", weighted_score)
     return weighted_score



df = pd.read_csv('/home/zyj/FedNUR/data/user_video_duration.csv')
output_directory = "/home/zyj/FedNUR/data/useless"  # 目标文件夹

os.makedirs(output_directory, exist_ok=True)

#展开df中的复杂列
df = detect_complex_columns(df)

#识别数据集中的用户列，评分列和特征列
#自动识别用户ID列
user_id_col = next((col for col in df.columns if 'user' in col.lower()), None)
if not user_id_col:
     raise ValueError("无法找到用户 ID 列，请手动指定。")
else:
     print('用户ID列识别完成。')

#自动识别评分列
score_col = next((col for col in df.columns if 'score' in col.lower() or 'rating' in col.lower()),None)
if not score_col:
    raise ValueError("无法找到评分列，请手动指定。")
else:
     print('score列识别完成')

# 自动识别特征列
exclude_cols = ['visible_status', 'video_id'] 

feature_cols = [col for col in df.columns if col not in [user_id_col, score_col] + exclude_cols]

user_attribute_tables = {}
#计算用户属性偏好表
for feature_col in feature_cols:
     print(feature_col)
     print(user_id_col)
     # user_attribute_table = df.pivot_table(
     #      index=user_id_col,  # 用户 ID
     #      columns=feature_col,   # 展开后的属性项
     #      values=score_col,   # 评分
     #      aggfunc=lambda x:custom_aggregation(x, df, feature_col, user_id_col),   # 计算平均评分
     #      fill_value=0      # 填充未交互的属性
     # )
     # 在对数据进行聚合之前，首先按照用户 ID 和特征列分组，计算每个特征的评分数量和平均评分
     user_attribute_table = df.groupby([user_id_col, feature_col]).agg(
          {
               score_col:['count', 'mean']
          }
     )
     print(user_id_col)
     
     user_attribute_table.columns = ['score_count', 'score_mean']
     user_score_counts = user_attribute_table.groupby(user_id_col)['score_count'].sum()
     user_attribute_table['weighted_score'] = user_attribute_table['score_mean'] + user_attribute_table['score_mean'] * (user_attribute_table['score_count'] / user_score_counts[user_attribute_table.index.get_level_values(user_id_col)].values)
     print(user_attribute_table.head())

     user_attribute_table = user_attribute_table.pivot_table(
          index = user_id_col,
          columns = feature_col,
          values = 'weighted_score',
          fill_value = 0
     )
     # user_attribute_table = df.groupby(user_id_col).apply(
     #      lambda x: custom_aggregation(x[score_col], df, feature_col, user_id_col)
     # )



# 将生成的偏好表存入字典
     user_attribute_tables[feature_col] = user_attribute_table
      # 保存为单独的 CSV 文件
     output_file = os.path.join(output_directory, f"{feature_col}_preference_table_weighted_1.csv")
     user_attribute_table.to_csv(output_file, index=True)

     print(f"偏好表 {feature_col} 已保存到: {output_file}")
