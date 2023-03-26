import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
wine_data = pd.read_csv('winemag-data-130k-v2.csv')

# Drop duplicate rows
wine_data.drop_duplicates(inplace=True)

# Explore the dataset
print(wine_data.head())
print(wine_data.info())
print(wine_data.describe())

# Create a histogram of wine prices
plt.hist(wine_data['price'], bins=30)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Wine Prices')
plt.show()

# Create a scatter plot of wine prices vs. points
sns.scatterplot(x='price', y='points', data=wine_data)
plt.xlabel('Price')
plt.ylabel('Points')
plt.title('Wine Prices vs. Points')
plt.show()

# Create a bar chart of wine counts by country
country_counts = wine_data['country'].value_counts().head(10)
sns.barplot(x=country_counts.index, y=country_counts.values)
plt.xlabel('Country')
plt.ylabel('Wine Count')
plt.title('Top 10 Wine Producing Countries')
plt.show()

# Create a heatmap of wine prices by country and variety
country_variety_prices = wine_data.pivot_table(index='country', columns='variety', values='price')
sns.heatmap(country_variety_prices, cmap='YlGnBu')
plt.xlabel('Variety')
plt.ylabel('Country')
plt.title('Wine Prices by Country and Variety')
plt.show()
