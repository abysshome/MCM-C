import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 读取CSV文件
df = pd.read_csv("weatherHistory.csv", encoding='ISO-8859-1')

# 选择聚类和分类的特征
features = df[['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 
               'Wind Speed (km/h)', 'Wind Bearing (degrees)', 
               'Visibility (km)', 'Pressure (millibars)']]

# 标准化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(features_scaled)

# 将 DBSCAN 的标签映射为 0 和 1，-1 为噪声，标记为 1（异常），其他标记为 0（正常）
y_true = np.where(labels == -1, 1, 0)  # 1: 异常, 0: 正常
y_pred = np.where(labels == -1, 1, 0)  # DBSCAN 输出的标签

# 计算噪声比（DBSCAN 标签为-1的比例）
noise_count = np.sum(labels == -1)
noise_ratio = noise_count / len(labels)

# 输出噪声比
print(f"噪声数量: {noise_count}")
print(f"噪声比: {noise_ratio}")

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("混淆矩阵:\n", cm)

# 计算F1分数
f1 = f1_score(y_true, y_pred)
print(f"F1 分数: {f1:.4f}")

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 返回其他评估指标（详细的分类报告）
print("\n分类报告:\n", classification_report(y_true, y_pred))
