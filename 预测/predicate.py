import pandas as pd
import matplotlib.pyplot as plt
# 提取国家
def extract_country():
    # 读取CSV文件
    data = pd.read_csv("预测/summerOly_medal_counts.csv")

    # 需要提取的国家列表
    countries = [
        "Argentina", "Australia", "Austria", "Azerbaijan", "Belgium", "Bulgaria",   "Canada", 
        "China", "Chinese Taipei", "Colombia", "Croatia", "Denmark", "Finland",     "France", 
        "Georgia", "Germany", "Great Britain", "Greece", "Hungary", "India",    "Indonesia", 
        "Iran", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Kazakhstan",     "Kenya", 
        "Mexico", "Morocco", "Netherlands", "New Zealand", "North Korea", "Norway",  "Poland", 
        "Portugal", "Romania", "Slovakia", "South Africa", "Sweden", "Switzerland",     "Turkey", 
        "United States", "South Korea","Taiwan"
    ]
    # 提取这些国家的数据
    filtered_data = data[data['NOC'].isin(countries)]
    # 保存过滤后的数据到新的CSV文件
    filtered_data.to_csv("预测/filtered_summerOly_medal_counts.csv", index=False)

# 给运动员评级
def player_leveled():
    data = pd.read_csv("预测/summerOly_athletes.csv")

    # 定义奖牌对应的分数
    medal_scores = {
        'No medal': 0,
        'Bronze': 0.3,
        'Silver': 0.5,
        'Gold': 1
    }

    # 为每个奖项添加对应的分数
    data['Score'] = data['Medal'].map(medal_scores)

    # 统计每个运动员的总得分
    total_scores = data.groupby('Name')['Score'].sum()

    total_scores.to_csv("预测/medal_counts_by_player.csv", index=True)

    # 重置索引并命名列
    total_scores = total_scores.reset_index()
    total_scores.columns = ['Name', 'Score']  # 只有两个列名 'Name' 和 'Score'
    # 定义得分区间和评级
    bins = [-0.1, 0, 1.0, float('inf')]  # 分数区间：0，(0, 1.0]，(1.0, inf)
    labels = ['普通运动员', '良好运动员', '优秀运动员']  # 对应的标签

    # 使用 pd.cut() 进行分类
    total_scores['Level'] = pd.cut(total_scores['Score'], bins=bins, labels=labels, right=True)

    # 将结果保存到 CSV 文件
    total_scores.to_csv("预测/medal_counts_by_player.csv", index=False)

    # 查看输出
    print(total_scores)


# 将奖牌数量划分为0、1、2三个等级
def classify_performance(count):
    if count == 0:
        return 0  # 一般
    elif count <= 2:
        return 1  # 良好
    else:
        return 2  # 优秀

# 绘制运动员得分饼状图
def draw_pie_chart():
    # 读取CSV数据
    data = pd.read_csv("预测/medal_counts_by_player.csv")

    # 定义得分区间
    bins = [-1,0, 0.3, 0.5, 1.0, data['Score'].max() + 0.1]
    labels = ['0','[0,0.3]','(0.3,0.5]', '(0.5,1.0]', f'({1.0},{data["Score"].max()}]']

    # 根据得分划分区间
    data['Score Range'] = pd.cut(data['Score'], bins=bins, labels=labels, right=True)

    # 统计每个区间的运动员数量
    score_distribution = data['Score Range'].value_counts()

    # 绘制饼状图
    plt.figure(figsize=(8, 8))
    plt.pie(score_distribution, labels=score_distribution.index, autopct='%1.1f%%',     startangle=90)
    plt.title('Distribution of Athletes by Score Range')
    plt.axis('equal')  # 使饼状图为圆形
    plt.show()

# 统计国家拥有的三类运动员的数量
def count_athletes_by_level():
    # 读取运动员数据集
    athletes_data = pd.read_csv('预测/summerOly_athletes.csv')  # 请根据文件实际路径调整
    ratings_data = pd.read_csv('预测/medal_counts_by_player.csv')  # 请根据文件实际路径调整

    # 合并两个数据集，基于运动员名字进行合并
    merged_data = pd.merge(athletes_data, ratings_data, on='Name', how='left')

    # 对每个NOC和Year进行分组，统计每个等级的数量
    result = merged_data.groupby(['NOC', 'Year', 'Level']).size().unstack(fill_value=0)
    result.to_csv("预测/athletesLevel_num_by_NOC_Year.csv", index=True)
    # 输出结果
    print(result)

# 合并奖牌数和运动员
def merge():
    # 读取奖牌数据集
    medals_data = pd.read_csv('预测/filtered_summerOly_medal_counts.csv')  # 请根据文件实际路径调整

    # 读取运动员数据集
    athletes_data = pd.read_csv('预测/athletesLevel_num_by_NOC_Year.csv')  # 请根据文件实际路径调整

    # 合并两个数据集，基于 'NOC' 和 'Year' 列
    merged_data = pd.merge(medals_data, athletes_data, on=['NOC', 'Year'], how='left')

    merged_data.to_csv("预测/merged_data.csv", index=True)

import pycountry
# Name->NOC
def merage_1():
    # 读取奖牌数据集
    medals_data = pd.read_csv('预测/filtered_summerOly_medal_counts.csv')  # 请根据文件实际路径调整
    # 读取运动员数据集
    athletes_data = pd.read_csv('预测/athletesLevel_num_by_NOC_Year.csv')  # 请根据文件实际路径调整

    # 将奖牌数据集中的 NOC 全称转换为三字母缩写
    medals_data['NOC'] = medals_data['NOC'].apply(full_name_to_abbr)
    medals_data.to_csv("预测/filtered_summerOly_medal_counts.csv", index=True)
    # 合并两个数据集，基于 'NOC' 和 'Year' 列
    merged_data = pd.merge(medals_data, athletes_data, on=['NOC', 'Year'], how='left')

    medals_data.to_csv("预测/meraged_data.csv", index=True)
    # 输出合并后的结果
    print(merged_data)

# 定义一个函数，使用 pycountry 将国家全称转换为三字母缩写
def full_name_to_abbr(full_name):
    try:
        country = pycountry.countries.get(name=full_name)
        return country.alpha_3 if country else None
    except KeyError:
        return None
        
if __name__ == "__main__":
    # extract_country()
    player_leveled()
    draw_pie_chart()
    # count_athletes_by_level()
    # merge()

