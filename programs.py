import pandas as pd

# 数据加载
athletes_data_path = "2025_Problem_C_Data\summerOly_athletes.csv"
medal_counts_data_path = "2025_Problem_C_Data\summerOly_medal_counts.csv"

# 加载运动员数据
athletes_data = pd.read_csv(athletes_data_path)
medal_counts_data = pd.read_csv(medal_counts_data_path)

# 数据清洗：只保留有奖牌的数据
athletes_data = athletes_data[athletes_data["Medal"].notnull()]

# 按项目和国家统计奖牌数
project_medals = athletes_data.groupby(["Sport", "NOC"])["Medal"].count().reset_index()
project_medals.rename(columns={"Medal": "MedalCount"}, inplace=True)

# 计算每个项目的总奖牌数
project_totals = project_medals.groupby("Sport")["MedalCount"].sum().reset_index()
project_totals.rename(columns={"MedalCount": "TotalMedals"}, inplace=True)

# 合并总奖牌数以计算占比
project_medals = project_medals.merge(project_totals, on="Sport")
project_medals["MedalPercentage"] = (
    project_medals["MedalCount"] / project_medals["TotalMedals"]
)

# 1. 识别主导国家
dominant_projects = project_medals.loc[
    project_medals.groupby("Sport")["MedalCount"].idxmax()
]
dominant_projects = dominant_projects[["Sport", "NOC", "MedalCount", "MedalPercentage"]]
dominant_projects.rename(columns={"NOC": "DominantCountry"}, inplace=True)

# 2. 计算竞争激烈的项目（衡量奖牌分布均匀性，标准差越小竞争越激烈）
competition_stats = (
    project_medals.groupby("Sport")["MedalPercentage"].std().reset_index()
)
competition_stats.rename(
    columns={"MedalPercentage": "CompetitionIntensity"}, inplace=True
)

# 合并主导国家和竞争激烈性
project_analysis = dominant_projects.merge(competition_stats, on="Sport")

# 3. 筛选垄断性项目（某国奖牌比例 > 50%）
monopolized_projects = project_analysis[project_analysis["MedalPercentage"] > 0.5]

# 输出分析结果
print("主导国家分析：")
print(project_analysis)

print("\n竞争激烈性分析（标准差越小越激烈）：")
print(competition_stats)

print("\n垄断性项目：")
print(monopolized_projects)

# 保存结果到文件
project_analysis.to_csv("project_analysis.csv", index=False)
monopolized_projects.to_csv("monopolized_projects.csv", index=False)
