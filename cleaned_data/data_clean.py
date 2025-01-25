import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
# 检测缺失值
def detect_missing_value(file_name:str):
    # 读取 CSV 文件
    df = pd.read_csv(file_name)
    # 检查每列的缺失值
    missing_values = df.isnull().sum()
    # 输出每列缺失值的数量
    print("Missing value detection results:")
    print(missing_values)

# 每个NOC对应的不同Team的数量
def delete_team(file_name: str):
    # 读取CSV文件
    df = pd.read_csv(file_name)
    # 按NOC分组，统计每个NOC对应的不同Team数量
    result = df.groupby("NOC")["Team"].nunique().reset_index()
    # 修改列名
    result.columns = ["NOC", "Unique Team Count"]
    # 如果需要将结果保存为新CSV文件
    result.to_csv('team_noc_count.csv', index=False)
    print(result.head(10))

    # 创建子图，1行2列，图1为直方图，图2为饼状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制直方图
    axes[0].hist(result['Unique Team Count'], bins=range(1, result['Unique Team Count'].max() + 2), edgecolor='black', alpha=0.7)
    axes[0].set_title('Frequency Distribution of Unique Team Count', fontsize=14)
    axes[0].set_xlabel('Unique Team Count', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 自定义分组区间
    bins = [0, 3, 10, 50, float('inf')]  # 定义分组：1-3, 4-10, 10-50, >50
    labels = ['1-3', '4-10', '10-50', '>50']  # 区间标签

    # 将数据分组并统计每个区间的频次
    result['Group'] = pd.cut(result['Unique Team Count'], bins=bins, labels=labels, right=True)
    count_distribution = result['Group'].value_counts().sort_index()

    # 绘制饼状图
    axes[1].pie(
        count_distribution,
        labels=count_distribution.index,  # 使用区间标签
        autopct='%1.1f%%',  # 显示百分比
        startangle=90,  # 起始角度
        colors=plt.cm.Paired.colors,  # 配色方案
        wedgeprops={'edgecolor': 'black'}  # 添加边框
    )
    axes[1].set_title('Distribution of Unique Team Count by Range', fontsize=14)

    # 调整图表布局，避免重叠
    plt.tight_layout()
    plt.show()


def padel_num_by_NOC():

    df = pd.read_csv("summerOly_medal_counts.csv")

    # 按国家和年份分组，计算奖牌总数
    grouped = df.groupby(['NOC', 'Year']).agg({'Total': 'sum'}).reset_index()

    # 绘制每个国家的奖牌总数趋势
    plt.figure(figsize=(16, 12))

    # 统计每个国家的出现次数
    country_counts = grouped['NOC'].value_counts()

    # 筛选出现次数大于等于10次的国家
    countries = country_counts[country_counts >= 20].index
    countries = countries.append(pd.Index(['China']))
    print(countries)
    # 绘制每个国家的奖牌总数趋势
    for country in countries:
        country_data = grouped[grouped['NOC'] == country]
        plt.plot(country_data['Year'], country_data['Total'], label=country)

    # 设置图表标题和标签
    plt.xlabel('Year')
    plt.xticks(range(min(grouped['Year']), max(grouped['Year']) + 1, 8))
    plt.ylabel('Total Medals')
    plt.title('Total Medal Trends by Country (Countries with >=20 occurrences)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Countries")
    plt.grid(True)
    plt.tight_layout()

    # 显示图表
    plt.show()

def medals_by_country_year():

    # 创建DataFrame
    columns = ['Name', 'Sex', 'NOC', 'Year', 'City', 'Sport', 'Event', 'Medal']
    df = pd.read_csv("summerOly_athletes.csv")

    # 创建新列，标记金、银、铜奖牌
    df['Gold'] = df['Medal'].apply(lambda x: 1 if x == 'Gold' else 0)
    df['Silver'] = df['Medal'].apply(lambda x: 1 if x == 'Silver' else 0)
    df['Bronze'] = df['Medal'].apply(lambda x: 1 if x == 'Bronze' else 0)
    
    # 按照国家（NOC）和年份（Year）分组，并计算每个奖牌的数量
    grouped = df.groupby(['NOC', 'Year'])[['Gold', 'Silver', 'Bronze']].sum().reset_index()
    
    # 使用 pivot 将数据转换为三维形式（每种奖牌在每年每个国家的数量）
    pivot_df = grouped.pivot_table(index='NOC', columns='Year', values=['Gold', 'Silver', 'Bronze'], fill_value=0)
    
    # 将列名格式化为字符串（可选）
    pivot_df.columns = [f'{medal}_{year}' for medal, year in pivot_df.columns]
    
    # 导出为 CSV 文件
    pivot_df.to_csv('medals_by_country_year.csv')
    
    # 打印输出
    print(pivot_df)

if __name__ == '__main__':
    os.chdir(r"C:\Users\xuwen\Desktop\MCM-C\cleaned_data")
    print(os.getcwd())
    # padel_num_by_NOC()
    # delete_team("summerOly_athletes.csv")
    medals_by_country_year()
