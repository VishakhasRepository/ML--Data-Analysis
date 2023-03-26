import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('grubhub_restaurant_data.csv')

# Check the structure of the dataset
df.head()

# Examine the variables in the dataset
df.info()

# Remove unnecessary variables
df = df.drop(['Unnamed: 0', 'zip', 'yelp_id'], axis=1)

# Check for missing values
df.isnull().sum()

# Clean the data
df['cuisine'] = df['cuisine'].str.lower().str.strip()

# Identify the most popular types of cuisine ordered for delivery during the pandemic
cuisine_counts = df[df['date'] > '2020-03-01']['cuisine'].value_counts().nlargest(10)
print(cuisine_counts)

# Analyze the change in demand for food delivery services before and during the pandemic
demand_before_pandemic = df[df['date'] < '2020-03-01']['date'].value_counts().sort_index()
demand_during_pandemic = df[df['date'] > '2020-03-01']['date'].value_counts().sort_index()
plt.plot(demand_before_pandemic.index, demand_before_pandemic.values, label='Before Pandemic')
plt.plot(demand_during_pandemic.index, demand_during_pandemic.values, label='During Pandemic')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.title('Change in Demand for Food Delivery Services')
plt.legend()
plt.show()

# Identify the most popular times of day for ordering food during the pandemic
df['hour'] = pd.to_datetime(df['time']).dt.hour
hourly_counts = df[df['date'] > '2020-03-01']['hour'].value_counts().sort_index()
plt.bar(hourly_counts.index, hourly_counts.values)
plt.xlabel('Hour of Day')
plt.ylabel('Number of Orders')
plt.title('Most Popular Times of Day for Ordering Food During the Pandemic')
plt.show()

# Split the data into training and testing sets
train = df[df['date'] < '2020-10-01']
test = df[df['date'] >= '2020-10-01']

# Create a time series model to forecast future demand
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(train['hour'].values, order=(2, 1, 2))
model_fit = model.fit()
predictions = model_fit.forecast(steps=len(test))
plt.plot(test['hour'].values)
plt.plot(predictions[0])
plt.xlabel('Time (Days)')
plt.ylabel('Number of Orders')
plt.title('Forecast of Food Delivery Demand')
plt.show()
