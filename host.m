% 读取CSV文件
data = readtable("C:\Users\16378\Downloads\summerOly_medal_counts.csv"); 
country = 'United States'; % choose the host country
host_years = [1904 1932 1984 1996];
countryData = data(strcmp(data.NOC, country), :);
totalGoldPerYearAll = varfun(@sum, data, 'InputVariables', 'Gold', 'GroupingVariables', 'Year');
totalMedalsPerYearAll = varfun(@sum, data, 'InputVariables', 'Total', 'GroupingVariables', 'Year');
countryGoldPerYear = varfun(@sum, countryData, 'InputVariables', 'Gold', 'GroupingVariables', 'Year');
countryTotalMedalsPerYear = varfun(@sum, countryData, 'InputVariables', 'Total', 'GroupingVariables', 'Year');
yearsParticipated = unique(countryData.Year);
totalGoldPerYear = totalGoldPerYearAll(ismember(totalGoldPerYearAll.Year, yearsParticipated), :);
totalMedalsPerYear = totalMedalsPerYearAll(ismember(totalMedalsPerYearAll.Year, yearsParticipated), :);


goldPercentage = countryGoldPerYear.sum_Gold ./ totalGoldPerYear.sum_Gold;
totalMedalsPercentage = countryTotalMedalsPerYear.sum_Total ./ totalMedalsPerYear.sum_Total;

non_host_years = ~ismember(countryGoldPerYear.Year, host_years);
average_gold_percentage_non_host = mean(goldPercentage(non_host_years))

% 绘制折线图
figure;
yyaxis left;
plot(countryGoldPerYear.Year, goldPercentage, '-o', 'MarkerFaceColor', 'blue');
ylabel('Gold Medal Percentage');
yyaxis right;
plot(countryTotalMedalsPerYear.Year, totalMedalsPercentage, '-x', 'MarkerFaceColor', 'red');
ylabel('Total Medals Percentage');
xlabel('Year');
title(['Medal Percentages Over Years for ', country]);
grid on;
legend('Gold Medal Percentage', 'Total Medals Percentage');