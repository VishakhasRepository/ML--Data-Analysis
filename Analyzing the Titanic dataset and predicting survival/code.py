# Section 1: Data Preparation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Load the Titanic dataset
titanic_df = pd.read_csv('train.csv')

# Explore the dataset using summary statistics and visualizations
print(titanic_df.describe())
sns.pairplot(titanic_df, hue='Survived')

# Handle missing values in the Age, Cabin, and Embarked columns
imputer = SimpleImputer(strategy='most_frequent')
titanic_df[['Age', 'Cabin', 'Embarked']] = imputer.fit_transform(titanic_df[['Age', 'Cabin', 'Embarked']])

# Handle outliers in the Fare column
sns.boxplot(titanic_df['Fare'])
titanic_df = titanic_df[titanic_df['Fare'] < 100]

# Convert categorical features to numerical features using one-hot encoding
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), ['Sex', 'Embarked'])], remainder='passthrough')
X = transformer.fit_transform(titanic_df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket']))
y = titanic_df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Section 2: Model Building

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train the model on the training set using Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Logistic Regression accuracy: ', accuracy_score(y_test, y_pred))

# Train the model on the training set using Decision Trees
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('Decision Trees accuracy: ', accuracy_score(y_test, y_pred))

# Train the model on the training set using Random Forests
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Random Forests accuracy: ', accuracy_score(y_test, y_pred))

# Section 3: Model Interpretation and Visualization

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Visualize the results of the model using confusion matrices and ROC curves
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

# Section 4: Conclusion and Recommendations

# Summarize the findings and insights from the analysis
print('Based on the analysis, the factors that had the greatest impact on survival were:')
print('- Being female')
print('- Being in a higher passenger class')
print('- Having fewer siblings/spouses onboard')
print('- Having one or two parents/children onboard')

# Provide recommendations for improving survival rates on future voyages based on the insights gained from the analysis
print('To improve survival rates on future voyages, the following measures could be taken:')
print('- Increase the number of lifeboats and ensure that they are easily accessible')
print('- Provide training for passengers and crew on emergency procedures')
print('- Encourage passengers to travel with family members to increase the likelihood of having a support system onboard')
print('- Consider policies to prioritize the evacuation of women and children, and passengers in higher passenger classes')

# Identify areas for further research and analysis to deepen our understanding of the factors that influence survival in disasters such as the sinking of the Titanic
print('Areas for further research and analysis could include:')
print('- Investigating the impact of cabin location and proximity to lifeboats on survival rates')
print('- Analyzing the impact of the crew and their actions on the survival of passengers')
print('- Examining the impact of cultural norms and social expectations on the behavior of passengers and crew during a disaster')
