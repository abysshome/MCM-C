import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 读取数据
data = pd.read_csv('预测/merged_data.csv')

# 数据预处理
# 计算总奖牌数
data['Total'] = data['Gold'] + data['Silver'] + data['Bronze']

# 特征和目标
X = data[['Year', 'Gold', 'Silver', 'Bronze', 'player0_num', 'player1_num', 'player2_num']]
y_total = data['Total']
y_rank = data['Rank']

# 分割数据集
X_train, X_test, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)
_, _, y_rank_train, y_rank_test = train_test_split(X, y_rank, test_size=0.2, random_state=42)

# 创建回归模型（预测奖牌总数）
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_total_train)

# 预测总奖牌数
y_total_pred = regressor.predict(X_test)
print("Mean Absolute Error for Total Medal Prediction:", mean_absolute_error(y_total_test, y_total_pred))

# 创建分类模型（预测排名）
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_rank_train)

# 预测排名
y_rank_pred = classifier.predict(X_test)
print("Accuracy for Rank Prediction:", accuracy_score(y_rank_test, y_rank_pred))
