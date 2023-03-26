# Section 1: Data Loading and Inspection
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('climate_change.csv')

# Check for missing values and anomalies in the data
print(data.isnull().sum())
print(data.describe())

# Filter out any irrelevant columns or rows that are not needed for the analysis
data = data.drop(['City', 'Country', 'Latitude', 'Longitude'], axis=1)
data = data[data['Source'] == 'GCAG']

# Print the shape and summary statistics of the data to get a general understanding of the data
print(data.shape)
print(data.head())

# Section 2: Data Cleaning and Preprocessing
# Handle missing values by imputing or dropping them
data = data.dropna()
# Convert data types to appropriate formats
data['Date'] = pd.to_datetime(data['Date'])
# Check for data quality and correct any errors or inconsistencies in the data

# Section 3: Data Exploration and Visualization
# Create a time series plot of the average temperature over the years to identify any changes in global temperatures
plt.plot(data['Date'], data['Mean'])
plt.title('Global Temperature Trend')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.show()

# Use a heat map to visualize the distribution of temperature across different regions of the world
plt.scatter(data['Year'], data['Mean'], c=data['Mean'], cmap='coolwarm')
plt.title('Global Temperature Heatmap')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.colorbar()
plt.show()

# Create scatter plots to identify any relationships between the different variables in the dataset such as latitude, longitude, and temperature
plt.scatter(data['Year'], data['Uncertainty'], c=data['Mean'], cmap='coolwarm')
plt.title('Temperature Uncertainty vs. Mean')
plt.xlabel('Year')
plt.ylabel('Temperature Uncertainty')
plt.colorbar()
plt.show()

# Section 4: Data Analysis and Hypothesis Testing
import scipy.stats as stats

# Conduct statistical analysis to validate any trends or patterns identified in the data
# For example, we can conduct a t-test to check if the mean temperature in recent years is significantly different from the mean temperature in the past.
recent_years = data[data['Year'] >= 2000]['Mean']
past_years = data[data['Year'] < 1900]['Mean']
t_stat, p_value = stats.ttest_ind(recent_years, past_years)
print('t-statistic: ', t_stat)
print('p-value: ', p_value)

# Use hypothesis testing to identify any significant changes in global temperatures over the years
# For example, we can conduct a one-sample t-test to check if the average temperature in recent years is significantly different from a fixed value (such as the average temperature in the 19th century).
fixed_value = data[data['Year'] < 1900]['Mean'].mean()
recent_years = data[data['Year'] >= 2000]['Mean']
t_stat, p_value = stats.ttest_1samp(recent_years, fixed_value)
print('t-statistic: ', t_stat)
print('p-value: ', p_value)

# Use correlation analysis to identify any relationships between different variables in the dataset
# For example, we can calculate the correlation coefficient between temperature and uncertainty to see if they are positively or negatively correlated.
corr_coef, p_value = stats.pearsonr(data['Mean'], data['Uncertainty'])
print('Correlation Coefficient: ', corr_coef)
print('p-value: ', p_value)

# Use regression analysis to identify any possible causes or drivers of climate change
# For example, we can use linear regression to model the relationship between temperature and time, and see if there is a significant increase in temperature over time.
from sklearn.linear_model import LinearRegression
X = data['Year'].values.reshape(-1, 1)
y = data['Mean'].values.reshape(-1, 1)
model = LinearRegression().fit(X, y)
print('Coefficient of determination (R^2): ', model.score(X, y))
print('Slope (temperature increase per year): ', model.coef_)

# Section 5: Data Visualization and Dashboard Creation
import seaborn as sns

# Create visualizations that illustrate the findings and insights gained from the analysis
# For example, we can create a scatter plot with a regression line to show the relationship between temperature and time.
sns.regplot(x='Year', y='Mean', data=data)
plt.title('Global Temperature Trend')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.show()

# Use interactive charts and graphs if possible to enable users to explore the data and gain a better understanding of climate change
# For example, we can create an interactive heat map to show the distribution of temperature across different regions of the world, using Plotly.
import plotly.express as px
fig = px.density_mapbox(data, lat='Latitude', lon='Longitude', z='Mean', radius=10,
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style='carto-positron')
fig.show()

# Create a dashboard to summarize the findings and insights gained from the analysis
# For example, we can use Streamlit to create an interactive dashboard with different charts and visualizations.
import streamlit as st
# code to create dashboard goes here

# Section 6: Report Generation and Conclusion
# Summarize the findings and insights gained from the analysis in a report
