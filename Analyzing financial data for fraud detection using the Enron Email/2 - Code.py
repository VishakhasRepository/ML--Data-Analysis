# Importing necessary libraries
import os
import pickle
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
enron_data = pickle.load(open("final_project/final_project_dataset.pkl", "rb"))
df = pd.DataFrame(enron_data).transpose()

# Data cleaning
df = df.replace("NaN", np.nan)
df = df.drop(["TOTAL", "THE TRAVEL AGENCY IN THE PARK"], axis=0)

# Feature engineering
df["from_poi_ratio"] = df["from_poi_to_this_person"] / df["to_messages"]
df["to_poi_ratio"] = df["from_this_person_to_poi"] / df["from_messages"]
df["bonus_to_salary_ratio"] = df["bonus"] / df["salary"]
df["deferral_payments_ratio"] = df["deferral_payments"] / df["total_payments"]
df["deferred_income_ratio"] = df["deferred_income"] / df["total_payments"]
df["director_fees_ratio"] = df["director_fees"] / df["total_payments"]
df["exercised_stock_options_ratio"] = df["exercised_stock_options"] / df["total_stock_value"]
df["expenses_ratio"] = df["expenses"] / df["salary"]
df["long_term_incentive_ratio"] = df["long_term_incentive"] / df["total_payments"]
df["other_ratio"] = df["other"] / df["salary"]
df["restricted_stock_ratio"] = df["restricted_stock"] / df["total_stock_value"]
df["salary_ratio"] = df["salary"] / df["total_payments"]
df["shared_receipt_with_poi_ratio"] = df["shared_receipt_with_poi"] / df["to_messages"]
df["total_payments_to_stock_ratio"] = df["total_payments"] / df["total_stock_value"]
df["from_poi_to_this_person"].fillna(0, inplace=True)
df["from_this_person_to_poi"].fillna(0, inplace=True)
df["bonus_to_salary_ratio"].fillna(0, inplace=True)
df["deferral_payments_ratio"].fillna(0, inplace=True)
df["deferred_income_ratio"].fillna(0, inplace=True)
df["director_fees_ratio"].fillna(0, inplace=True)
df["exercised_stock_options_ratio"].fillna(0, inplace=True)
df["expenses_ratio"].fillna(0, inplace=True)
df["long_term_incentive_ratio"].fillna(0, inplace=True)
df["other_ratio"].fillna(0, inplace=True)
df["restricted_stock_ratio"].fillna(0, inplace=True)
df["salary_ratio"].fillna(0, inplace=True)
df["shared_receipt_with_poi_ratio"].fillna(0, inplace=True)
df["total_payments_to_stock_ratio"].fillna(0, inplace=True)

# Feature selection
features = ["poi", "salary", "to_messages", "deferral_payments", "total_payments",            "exercised_stock_options", "bonus", "restricted_stock", "shared_receipt_with_poi",            "total_stock_value", "expenses", "from_messages", "

# Drop unnecessary columns
df.drop(columns=['email_address', 'poi'], inplace=True)

# Replace NaN values with 0
df.replace('NaN', 0, inplace=True)

# Convert columns to numeric dtype
cols = df.columns
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# Remove outliers
df = remove_outliers(df)

# Normalize the data
df = normalize_data(df)

# Create new features
df['to_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['from_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['bonus_to_salary_ratio'] = df['bonus'] / df['salary']
df['expenses_to_salary_ratio'] = df['expenses'] / df['salary']

# Add new features to features_list
features_list += ['to_poi_ratio', 'from_poi_ratio', 'bonus_to_salary_ratio', 'expenses_to_salary_ratio']

# Select features to use in model
features = select_features(df, features_list, 'poi', 10)

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Instantiate models
models = [('Logistic Regression', LogisticRegression()), 
          ('Gaussian Naive Bayes', GaussianNB()),
          ('Random Forest', RandomForestClassifier(random_state=42)),
          ('XGBoost', XGBClassifier(random_state=42))]

# Evaluate models using cross-validation
for name, model in models:
    evaluate_model(model, features, labels, name)

# Define parameter grids for each model
param_grids = {'Logistic Regression': {'C': [0.1, 1, 10]},
               'Gaussian Naive Bayes': {},
               'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [5, 10]},
               'XGBoost': {'learning_rate': [0.1, 0.01], 'max_depth': [5, 10], 'n_estimators': [100, 200]}}

# Perform grid search for each model
for name, model in models:
    best_model = grid_search(model, param_grids[name], features, labels)
    evaluate_model(best_model, features, labels, name + ' (Tuned)')

# Evaluate best model on test set
best_model = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100, random_state=42)
evaluate_model(best_model, test_features, test_labels, 'XGBoost (Test Set)')

# Print feature importances
print_feature_importances(best_model, features)

# Save best model
joblib.dump(best_model, 'fraud_detection_model.joblib')
