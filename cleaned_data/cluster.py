import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


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


if __name__ == '__main__':
    os.chdir(r"C:\Users\xuwen\Desktop\MCM-C\cleaned_data")
    cluster2()
