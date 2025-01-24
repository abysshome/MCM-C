import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# 读取CSV文件
df = pd.read_csv("weatherHistory.csv", encoding='ISO-8859-1')

# 选择聚类和分类的特征
features = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]

# 对数据进行采样
sampled_df = features.sample(frac=0.2, random_state=42)  # 采样10%的数据

# 使用肘部法确定K值
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(sampled_df)
    inertia.append(kmeans.inertia_)

# 绘制肘部法图表
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, 'bx-')
plt.xticks(K_range)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for optimal K')
plt.grid(True)
plt.show()

# 应用K均值聚类
optimal_k = 3  # 通过肘部法或其他方法确定K值
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
clusters = kmeans.fit_predict(sampled_df)

# 添加聚类结果到采样数据框
sampled_df['Cluster'] = clusters

# 可视化聚类结果
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(sampled_df['Temperature (C)'], sampled_df['Humidity'], sampled_df['Wind Speed (km/h)'], c=sampled_df['Cluster'], cmap='viridis', alpha=0.5)
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Humidity')
ax.set_zlabel('Wind Speed (km/h)')
ax.set_title('3D Clusters of Weather Data')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
plt.show()
