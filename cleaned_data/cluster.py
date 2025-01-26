import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np


def normalize_data(df, columns):
    df_normalized = df.copy()  # 保留原始数据
    for column in columns:
        min_val = df[column].min()
        max_val = df[column].max()
        df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    return df_normalized

def cluster1():
    # 读入数据
    data = pd.read_csv("medals_by_country_year.csv")  # 假设数据是存储在 medals_data.csv 文件中

    # 选择所有奖牌数据列，排除NOC列
    medal_columns = [col for col in data.columns if col != 'NOC']

    # 对数据进行标准化
    scaler = StandardScaler() # 标准化方法：将数据转换为均值为0，标准差为1的分布
    scaled_data = scaler.fit_transform(data[medal_columns]) 

    # 查看标准化后的数据
    print(scaled_data[:5])

    # 使用肘部法则选择聚类数
    wcss = []  # Within-cluster sum of squares
    for i in range(1, 220):  # 从1到10个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    # 绘制肘部法则图
    plt.plot(range(1, 220), wcss)
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # 假设选择了 4 个簇作为聚类数
    kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)

    # 添加聚类标签到数据中
    data['Cluster'] = kmeans.labels_

    # 使用 PCA 将数据降维到 2D
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # 绘制聚类结果
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Cluster'], cmap='viridis', s=50)
    plt.title('PCA - KMeans Clustering Results')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    from sklearn.cluster import AgglomerativeClustering

    # 使用层次聚类
    agg_clustering = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    data['Cluster_Agg'] = agg_clustering.fit_predict(scaled_data)

    # PCA 降维后的可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Cluster_Agg'], cmap='viridis', s=50)
    plt.title('PCA - Agglomerative Clustering Results')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

def cluster2():
    # # 假设你已经加载了CSV文件
    # data = pd.read_csv('summerOly_medal_counts.csv')

    # data['NOC']=data['NOC'].str.replace('?','')

    # # 按照国家和年份对数据进行聚合，得到每年每个国家的奖牌  数
    # pivot_data = data.pivot_table(index='NOC',  columns='Year', values=['Gold', 'Silver', 'Bronze'], aggfunc='sum', fill_value=0)

    # # 对列名进行调整，便于后续操作
    # pivot_data.columns = [f'{col[0]}_{col[1]}' for col  in pivot_data.columns]

    # pivot_data.to_csv('pivot_data.csv')
    # # 显示转换后的数据
    # print(pivot_data.head())
    # 读入数据
    data = pd.read_csv("pivot_data.csv")  
    medal_columns = [col for col in data.columns if col != 'NOC']

    # 对数据进行标准化
    scaler = StandardScaler() # 标准化方法：将数据转换为均值为0，标准差为1的分布
    scaled_data = scaler.fit_transform(data[medal_columns]) 

    # 查看标准化后的数据
    print(scaled_data[:5])

    # 使用肘部法则选择聚类数
    wcss = []  # Within-cluster sum of squares
    for i in range(1, 11):  # 从1到10个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    # 绘制肘部法则图
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # 假设选择了 4 个簇作为聚类数
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)

    # 添加聚类标签到数据中
    data['Cluster'] = kmeans.labels_

    # 使用 PCA 将数据降维到 2D
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # 绘制聚类结果
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Cluster'], cmap='viridis', s=50)
    plt.title('PCA - KMeans Clustering Results')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    from sklearn.cluster import AgglomerativeClustering

    # 使用层次聚类
    agg_clustering = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    data['Cluster_Agg'] = agg_clustering.fit_predict(scaled_data)

    # PCA 降维后的可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Cluster_Agg'], cmap='viridis', s=50)
    plt.title('PCA - Agglomerative Clustering Results')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

def cluster3():
  
    df = pd.read_csv("grouped_data_归一化.csv")

    # 分析每个国家的参赛记录
    country_count = df['NOC'].value_counts()
    # print(country_count)
    # 对每个国家进行分类
    for country in country_count.index:
        country_data = df[df['NOC'] == country]
        classify_country(country_data,categories)
    # 输出分类结果
    for category, countries in categories.items():
        print(f"{category}: {', '.join(countries)}\n")

# 分类函数
def classify_teams(row):
    # 分类1: 近20年没有参赛记录的国家
    if row['recent_20_years_count'] == 0:
        return '没有参赛记录的国家'
    
    # 分类2: 有连续参赛多次记录、成绩稳定的国家
    elif row['recent_40_years_count'] >= 8 and row['normalized_avg_diff_total'] < 0.29:
        return '成绩稳定的国家'
    
    # 分类3: 参赛记录多，但排名不稳定的国家
    elif row['recent_40_years_count'] >= 8 and row['normalized_avg_diff_total'] >= 0.29:
        return '排名不稳定的国家'
    
    # 分类4: 近年来刚开始参赛的国家
    elif row['recent_40_years_count'] < 8 and row["recent_20_years_count"]==5:
        return '近年来刚开始参赛的国家'
    
    # 分类5: 从未获得过奖牌的国家
    elif row['avg_gold'] == 0 and row['avg_silver'] == 0 and row['avg_bronze'] == 0:
        return '从未获得过奖牌的国家'
    
    return '近年来参与减少的国家'

# 分类函数
def classify_country_1(country_data, categories):
    # print(country_data)
    country = country_data['NOC'].iloc[0]
    
    # 获取参赛年份并计算最近10年的参赛记录
    recent_years = country_data[country_data['Year'] >= (country_data['Year'].max() - 10)]
    recent_count = len(recent_years)
    # print(recent_count)
    
    # 计算奖牌数量和排名的标准差
    total_medals = country_data['Gold'] + country_data['Silver'] + country_data['Bronze']
    rank_std = np.std(total_medals)
    # print(rank_std)
    # 1. 近几年没有参赛记录的国家
    if recent_count == 0:
        categories['近几年没有参赛记录的国家'].append(country)
    
    # 2. 有连续参赛多次记录、成绩稳定的国家
    elif recent_count >= 4 and rank_std < 10 and total_medals.sum() >= 50:
        categories['有连续参赛多次记录、成绩稳定的国家'].append(country)
    
    # 3. 参赛记录多，但排名不稳定
    elif recent_count >=4 and rank_std > 5 and total_medals.sum() < 10:
        categories['参赛记录多，但排名不稳定'].append(country)
    
    # 4. 近年来刚开始参赛的国家
    elif recent_count <= 3 and (recent_years['Year'].max() - recent_years['Year'].min() > 3):
        categories['近年来刚开始参赛的国家'].append(country)
    
    # 5. 特殊的：从未获得奖牌的国家
    elif total_medals.sum() == 0:
        categories['特殊的：从未获得奖牌的国家'].append(country)
    
    # 如果未匹配到任何类别，可以根据数据添加默认分类
    else:
        categories['参赛记录多，但排名不稳定'].append(country)

# def classify_country_2(country_data, categories):

# 计算每个国家的统计数据
def calculate_statistics():

    df = pd.read_csv("summerOly_medal_counts.csv")

    # 归一化
    columns_to_normalize = ['Gold', 'Silver', 'Bronze', 'Total']
    df_normalized = normalize_data(df, columns_to_normalize)

    df_normalized.to_csv('归一化奖牌.csv')
    # 获取每个国家的分组数据
    grouped = df.groupby('NOC').agg(
        avg_gold=('Gold', 'mean'),
        avg_silver=('Silver', 'mean'),
        avg_bronze=('Bronze', 'mean'),
        avg_total=('Total', 'mean'),
        var_gold=('Gold', 'var'),
        var_silver=('Silver', 'var'),
        var_bronze=('Bronze', 'var'),
        var_total=('Total', 'var'),
        avg_diff_gold=('Gold', lambda x: np.mean(np.abs((x - x.mean()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0))),
        avg_diff_silver=('Silver', lambda x: np.mean(np.abs((x - x.mean()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0))),
        avg_diff_bronze=('Bronze', lambda x: np.mean(np.abs((x - x.mean()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0))),
        avg_diff_total=('Total', lambda x: np.mean(np.abs((x - x.mean()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0))),

        recent_20_years_count=('Year', lambda x: len(x[x >= (2028 - 20)])),
        recent_40_years_count=('Year', lambda x: len(x[x >= (2028 - 40)])),
        recent_80_years_count=('Year', lambda x: len(x[x >= (2028 - 40)])),
    ).reset_index()

    # 计算方差和平均差基于归一化数据
    grouped['normalized_var_gold'] = df_normalized.groupby('NOC')['Gold'].var().values
    grouped['normalized_var_silver'] = df_normalized.groupby('NOC')['Silver'].var().values
    grouped['normalized_var_bronze'] = df_normalized.groupby('NOC')['Bronze'].var().values
    grouped['normalized_var_total'] = df_normalized.groupby('NOC')['Total'].var().values
    
    grouped['normalized_avg_diff_gold'] = df_normalized.groupby('NOC')['Gold'].apply(lambda x: np.mean(np.abs((x - x.mean()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0))).values
    grouped['normalized_avg_diff_silver'] = df_normalized.groupby('NOC')['Silver'].apply(lambda x: np.mean(np.abs((x - x.mean()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0))).values
    grouped['normalized_avg_diff_bronze'] = df_normalized.groupby('NOC')['Bronze'].apply(lambda x: np.mean(np.abs((x - x.mean()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0))).values
    grouped['normalized_avg_diff_total'] = df_normalized.groupby('NOC')['Total'].apply(lambda x: np.mean(np.abs((x - x.mean()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0))).values


    grouped.to_csv('grouped_data_归一化.csv', index=False)

 # 创建分类字典
categories = {
    '近几年没有参赛记录的国家': [],
    '有连续参赛多次记录、成绩稳定的国家': [],
    '参赛记录多，但排名不稳定': [],
    '近年来刚开始参赛的国家': [],
    '特殊的：从未获得奖牌的国家': []
}
import json
if __name__ == '__main__':
    os.chdir(r"C:\Users\xuwen\Desktop\MCM-C\cleaned_data")
    calculate_statistics()
    # cluster3()
    df=pd.read_csv("grouped_data_归一化.csv")
    df['分类'] = df.apply(classify_teams, axis=1)
    # 根据分类过滤数据
    categories = {
        "近20年没有参赛记录的国家": df[df['分类'] == '没有参赛记录的国家']['NOC'].tolist(),
        "有连续参赛多次记录、成绩稳定的国家": df[df['分类'] == '成绩稳定的国家']['NOC'].tolist(),
        "参赛记录多，但排名不稳定": df[df['分类'] == '排名不稳定的国家']['NOC'].tolist(),
        "近年来刚开始参赛的国家": df[df['分类'] == '近年来刚开始参赛的国家']['NOC'].tolist(),
        "从未获得过奖牌的国家": df[df['分类'] == '从未获得过奖牌的国家']['NOC'].tolist(),
        "近20年参与减少的国家": df[df['分类'] == '近年来参与减少的国家']['NOC'].tolist()
    }

    # 导出为一个JSON文件
    with open('NOC_classification.json', 'w',encoding="UTF-8") as json_file:
        json.dump(categories, json_file, indent=4,ensure_ascii=False)

    # 如果你希望查看生成的 JSON 数据，可以打印出来
    print(json.dumps(categories, ensure_ascii=False, indent=4))
