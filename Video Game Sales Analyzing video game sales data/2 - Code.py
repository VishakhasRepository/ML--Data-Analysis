# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Section 1: Data Cleaning and Preparation
# Load the dataset
df = pd.read_csv('video_games_sales.csv')

# Check for missing values and duplicates
print('Missing Values:')
print(df.isnull().sum())
print('\nDuplicate Rows:')
print(df[df.duplicated()])

# Handle missing values and duplicates
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Convert data types
df['Year'] = df['Year'].astype('int64')

# Create new features
df['Total_Sales'] = df['NA_Sales'] + df['EU_Sales'] + df['JP_Sales'] + df['Other_Sales']

# Section 2: Data Exploration and Visualization
# Descriptive statistics
print(df.describe())

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, cmap="YlGnBu", annot=True)
plt.show()

# Box plots
sns.boxplot(x='Platform', y='Global_Sales', data=df)
plt.show()

# Histograms
plt.hist(df['Year'], bins=40)
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Section 3: Feature Engineering and Selection
# Feature selection
features = ['Platform', 'Year', 'Genre', 'Publisher', 'Total_Sales']

# Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Section 4: Predictive Modeling
# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Global_Sales'], test_size=0.3, random_state=42)

# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
print('Linear Regression:')
print('MSE:', mean_squared_error(y_test, lr_pred))
print('R2:', r2_score(y_test, lr_pred))

# Section 5: Dashboard Creation
# Creating a dashboard using Plotly or Tableau to visualize the findings and insights gained from the analysis

# Section 6: Report Generation and Conclusion
# Summarize the findings and insights gained from the analysis in a report
# Provide recommendations for game developers and publishers to improve their products and increase sales
# Discuss the limitations of the analysis and possible sources of error in the data
# Identify areas for future research and exploration.

