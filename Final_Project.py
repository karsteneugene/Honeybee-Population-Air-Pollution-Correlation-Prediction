import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split

plt.style.use('fivethirtyeight')

# Loads the necessary data
honeybee_data = pd.read_csv("cleaned_honey.csv")
pollution_data = pd.read_csv("cleaned_pollution.csv")

# Drop columns that unexpectedly appear
honeybee_data = honeybee_data.drop(columns="Unnamed: 0")
pollution_data = pollution_data.drop(columns="Unnamed: 0")

# Mean number of colonies grouped by year
numcol_mean = {"numcol": "mean"}
average_honeybee = honeybee_data.groupby(["year"]).agg(numcol_mean)

# x range from 2000 to 2017
x = range(2000,2017)

# Line graph
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,1,1)
ax.plot(x,average_honeybee)
ax.set_title("Mean number of colonies per year")
ax.set_xlabel("Year")
ax.set_ylabel("Mean number of colonies")
ax.set_ylim(ymin=0)

# Mean number of AQI grouped by year
AQI = {"NO2 AQI": "mean", "O3 AQI": "mean", "SO2 AQI": "mean", "CO AQI": "mean"}
average_pollution = pollution_data.groupby(["Year"]).agg(AQI)
# print(len(honeybee_data))
# print(len(pollution_data))

# Grouped Bar Chart
aqi_mean = {"NO2 AQI": "mean", "O3 AQI": "mean", "SO2 AQI": "mean", "CO AQI": "mean"}
average_pollution = pollution_data.groupby(["Year"]).agg(aqi_mean)
width = 0.2

# split year to two graphs - from 2000 to 2008 and 2009 to 2016
year1 = np.arange(2000, 2009)
year2 = np.arange(2009, 2017)

# layout
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# Grouped bar chart from the year 2000 to 2008
ax.bar(year1-0.4,  average_pollution["NO2 AQI"].iloc[0:9], width)
ax.bar(year1-0.2, average_pollution["O3 AQI"].iloc[0:9], width)
ax.bar(year1, average_pollution["SO2 AQI"].iloc[0:9], width)
ax.bar(year1+0.2, average_pollution["CO AQI"].iloc[0:9], width)

# Grouped bar chart from the year 2009 to 2016
ax2.bar(year2-0.4,  average_pollution["NO2 AQI"].iloc[9:17], width)
ax2.bar(year2-0.2, average_pollution["O3 AQI"].iloc[9:17], width)
ax2.bar(year2, average_pollution["SO2 AQI"].iloc[9:17], width)
ax2.bar(year2+0.2, average_pollution["CO AQI"].iloc[9:17], width)

# Label and legends for the graph
ax.set_title("Mean AQI per year")
ax.set_xlabel("Year")
ax.set_ylabel("Mean AQI")

ax.set_xticks(np.arange(2000, 2009))
ax2.set_xticks(np.arange(2009, 2017))
ax.legend(["NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"])

honeybee_data.index = honeybee_data[['year', 'state']]
honeybee_arima = honeybee_data.drop(['year', 'state'], axis=1)
# print(honeybee_arima)

train, test = train_test_split(honeybee_arima, shuffle=False)

# ARIMA
model = SARIMAX(honeybee_arima,order=(1, 1, 1),seasonal_order=(1,1,1,12))
result = model.fit()
print(result.summary())
print("p-value:", adfuller(honeybee_arima.dropna())[1])

pred_uc = result.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = honeybee_arima.plot(label='observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Year')
ax.set_ylabel('Numcol')
plt.legend()

# train data for ARIMA
model = SARIMAX(train,order=(1, 1, 1),seasonal_order=(1,1,1,12))
result = model.fit()
print(result.summary())
print("p-value:", adfuller(train.dropna())[1])

pred_uc = result.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = train.plot(label='observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Year')
ax.set_ylabel('Numcol')
plt.legend()

# test data for ARIMA
model = SARIMAX(test,order=(1, 1, 1),seasonal_order=(1,1,1,12))
result = model.fit()
print(result.summary())
print("p-value:", adfuller(test.dropna())[1])

pred_uc = result.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = test.plot(label='observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Year')
ax.set_ylabel('Numcol')
plt.legend()

# Pearson Correlation
corr1 = scipy.stats.pearsonr(pollution_data["NO2 AQI"], honeybee_data["numcol"])
corr2 = scipy.stats.pearsonr(pollution_data["O3 AQI"], honeybee_data["numcol"])
corr3 = scipy.stats.pearsonr(pollution_data["SO2 AQI"], honeybee_data["numcol"])
corr4 = scipy.stats.pearsonr(pollution_data["CO AQI"], honeybee_data["numcol"])

print(corr1, corr2, corr3, corr4)

# Spearman Correlation
scorr1 = scipy.stats.spearmanr(pollution_data["NO2 AQI"], honeybee_data["numcol"])
scorr2 = scipy.stats.spearmanr(pollution_data["O3 AQI"], honeybee_data["numcol"])
scorr3 = scipy.stats.spearmanr(pollution_data["SO2 AQI"], honeybee_data["numcol"])
scorr4 = scipy.stats.spearmanr(pollution_data["CO AQI"], honeybee_data["numcol"])

print(scorr1, scorr2, scorr3, scorr4)

# Kendall Correlation
kcorr1 = scipy.stats.kendalltau(pollution_data["NO2 AQI"], honeybee_data["numcol"])
kcorr2 = scipy.stats.kendalltau(pollution_data["O3 AQI"], honeybee_data["numcol"])
kcorr3 = scipy.stats.kendalltau(pollution_data["SO2 AQI"], honeybee_data["numcol"])
kcorr4 = scipy.stats.kendalltau(pollution_data["CO AQI"], honeybee_data["numcol"])

print(kcorr1, kcorr2, kcorr3, kcorr4)

# Scatter plot (correlation)
slope1, intercept1, r1, p1, stderr1 = scipy.stats.linregress(pollution_data["NO2 AQI"], honeybee_data["numcol"])
slope2, intercept2, r2, p2, stderr2 = scipy.stats.linregress(pollution_data["O3 AQI"], honeybee_data["numcol"])
slope3, intercept3, r3, p3, stderr3 = scipy.stats.linregress(pollution_data["SO2 AQI"], honeybee_data["numcol"])
slope4, intercept4, r4, p4, stderr4 = scipy.stats.linregress(pollution_data["CO AQI"], honeybee_data["numcol"])

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(pollution_data["NO2 AQI"], honeybee_data["numcol"])
ax.plot(pollution_data["NO2 AQI"], intercept1 + slope1 * pollution_data["NO2 AQI"])

ax.scatter(pollution_data["O3 AQI"], honeybee_data["numcol"])
ax.plot(pollution_data["O3 AQI"], intercept2 + slope2 * pollution_data["O3 AQI"])

ax.scatter(pollution_data["SO2 AQI"], honeybee_data["numcol"])
ax.plot(pollution_data["SO2 AQI"], intercept3 + slope3 * pollution_data["SO2 AQI"])

ax.scatter(pollution_data["CO AQI"], honeybee_data["numcol"])
ax.plot(pollution_data["CO AQI"], intercept4 + slope4 * pollution_data["CO AQI"])

ax.legend(["NO2 AQI\n Pearson = %.2f\n Spearman = %.2f\n Kendall = %.2f" %(corr1[0], scorr1[0], kcorr1[0]),
           "O3 AQI\n Pearson = %.2f\n Spearman = %.2f\n Kendall = %.2f" %(corr2[0], scorr2[0], kcorr2[0]),
           "SO2 AQI\n Pearson = %.2f\n Spearman = %.2f\n Kendall = %.2f" %(corr3[0], scorr3[0], kcorr3[0]),
           "CO AQI\n Pearson = %.2f\n Spearman = %.2f\n Kendall = %.2f" %(corr4[0], scorr4[0], kcorr4[0])],
          loc="upper right")
ax.set_xlabel('Pollutants')
ax.set_ylabel('Average Number of Colonies')
ax.set_ylim(ymin=0)

plt.show()





