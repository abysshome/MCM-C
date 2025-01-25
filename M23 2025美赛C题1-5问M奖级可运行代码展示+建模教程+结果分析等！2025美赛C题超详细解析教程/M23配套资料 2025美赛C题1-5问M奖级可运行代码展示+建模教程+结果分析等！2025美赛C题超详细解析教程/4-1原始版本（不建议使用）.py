import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
programs_file_path = 'summerOly_programs.csv'
medal_counts_file_path = 'summerOly_medal_counts.csv'

programs_df = pd.read_csv(programs_file_path, encoding='ISO-8859-1')
medal_counts_df = pd.read_csv(medal_counts_file_path, encoding='ISO-8859-1')

# 清理数据，去除非数字列
programs_df_cleaned = programs_df.loc[:, ~programs_df.columns.str.contains("Code|Sport|Discipline|Sports Governing Body")]

# 去除特殊字符年份列，例如 "1906*"
programs_df_cleaned = programs_df_cleaned.rename(columns=lambda x: x.strip().replace('*', '') if isinstance(x, str) else x)

# 将所有年份列转换为数字类型
programs_df_cleaned = programs_df_cleaned.apply(pd.to_numeric, errors='coerce')

# 独热编码处理体育项目
programs_df_cleaned_encoded = pd.get_dummies(programs_df[['Sport']], drop_first=True)

# 合并独热编码的体育项目列到原数据中
programs_df_cleaned = pd.concat([programs_df_cleaned, programs_df_cleaned_encoded], axis=1)

# 重新整理数据为"年份"和"每个运动项目的赛事数"
programs_melted_cleaned = programs_df_cleaned.melt(var_name='Year', value_name='Number_of_Events')

# 确保所有年份都是整数类型
programs_melted_cleaned['Year'] = pd.to_numeric(programs_melted_cleaned['Year'], errors='coerce', downcast='integer')

# 计算每年每个国家的金牌数量
medal_counts_per_year = medal_counts_df.groupby(['Year', 'NOC'])['Gold'].sum().reset_index()

# 合并每年各国金牌数量与项目数量
merged_data_cleaned = pd.merge(medal_counts_per_year, programs_melted_cleaned, how='left', on='Year')

# 绘制每年项目数量与金牌数的折线图
plt.figure(figsize=(12, 6))
for year in merged_data_cleaned['Year'].unique():
    data_for_year = merged_data_cleaned[merged_data_cleaned['Year'] == year]
    plt.plot(data_for_year['Number_of_Events'], data_for_year['Gold'], marker='o', label=f"Year {year}")

# 设置图表标签和标题
plt.title("Impact of Number of Events on Gold Medal Counts (2000 and onward)", fontsize=14)
plt.xlabel("Number of Events", fontsize=12)
plt.ylabel("Gold Medal Count", fontsize=12)
plt.legend(title="Olympic Year")

# 显示图表
plt.tight_layout()
plt.show()
