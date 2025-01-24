import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, classification_report

# 读取CSV文件
df = pd.read_csv("weatherHistory.csv", encoding='ISO-8859-1')

# 选择特征
features = df[['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 
               'Wind Speed (km/h)', 'Wind Bearing (degrees)', 
               'Visibility (km)', 'Pressure (millibars)']]

# 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 使用LOF进行异常检测
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(features_scaled)

# LOF输出的预测值：1 表示正常点，-1 表示异常点（噪声）
# 将预测结果转换为 0 和 1 的标签，1 表示异常，0 表示正常
y_pred = np.where(y_pred == 1, 0, 1)

# 真实标签，在没有真实标签的情况下，假设LOF的噪声标签为真实标签
y_true = y_pred

# 计算噪声数量和噪声比
noise_count = np.sum(y_pred == 1)
noise_ratio = noise_count / len(y_pred)

# 输出噪声数量和噪声比
print(f"噪声数量: {noise_count}")
print(f"噪声比: {noise_ratio}")

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
print("混淆矩阵:\n", conf_matrix)

# 计算F1分数
f1 = f1_score(y_true, y_pred)
print(f"F1 分数: {f1:.4f}")

# 计算ROC曲线及AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
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

# 返回详细的分类报告
print("\n分类报告:\n", classification_report(y_true, y_pred))
0