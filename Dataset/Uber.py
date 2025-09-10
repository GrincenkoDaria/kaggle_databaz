# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = pd.read_csv('UberDataset.csv')

import matplotlib.pyplot as plt

data.head()

data.info()

print(data.describe())

print(data['CATEGORY'].value_counts())

total_miles = data['MILES'].sum()
print('Total miles driven:', total_miles)

avg_miles_per_trip = data['MILES'].mean()
print('Average miles per trip:', avg_miles_per_trip)

missing_purpose_count = data['PURPOSE'].isnull().sum()
print('Number of missing values in PURPOSE column:', missing_purpose_count)

df = pd.read_csv('UberDataset.csv')

category_counts = df['CATEGORY'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(category_counts.index, category_counts.values)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Record Count by Category')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['MILES'], bins=10, edgecolor='black')
plt.xlabel('Miles')
plt.ylabel('Frequency')
plt.title('Distribution of Miles Driven')
plt.show()

purpose_counts = df['PURPOSE'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(purpose_counts.values, labels=purpose_counts.index, autopct='%1.1f%%')
plt.title('Purpose of Trips')
plt.show()

df = df[df['START_DATE'] != "Totals"]

df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')


df.set_index('START_DATE', inplace=True)

daily_miles = df.resample('D')['MILES'].sum()

plt.figure(figsize=(12, 6))
plt.plot(daily_miles.index, daily_miles.values)
plt.xlabel('Date')
plt.ylabel('Miles Driven')
plt.title('Miles Driven Over Time')
plt.show()

category_purpose = df.pivot_table(index='PURPOSE', columns='CATEGORY', aggfunc='size', fill_value=0)

plt.figure(figsize=(10, 6))
category_purpose.plot(kind='bar', stacked=True)
plt.xlabel('Purpose')
plt.ylabel('Count')
plt.title('Category by Purpose')
plt.legend(title='Category')
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot([df[df['CATEGORY'] == 'Business']['MILES'], df[df['CATEGORY'] == 'Personal']['MILES']], labels=['Business', 'Personal'])
plt.xlabel('Category')
plt.ylabel('Miles')
plt.title('Distribution of Miles by Category')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
import lightgbm as lgb

df = pd.read_csv('UberDataset.csv')

features = ['START', 'STOP', 'CATEGORY', 'PURPOSE']

X = df[features]
X = pd.get_dummies(X) 
y = df['MILES']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor(),
    'Gradient Boosting Regressor' : GradientBoostingRegressor(),
    'ADA Boost' : AdaBoostRegressor(),
    'Linear SVR' : LinearSVR(),
}

Name = ['Linear Regression','Decision Tree','Random Forest','SVR','XGBoost','LightGBM','Gradient Boosting Regressor' ,'ADA Boost','Linear SVR']
accuracy = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Results for {name}:")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_test)), y_test, label='Actual Trend')
    plt.plot(np.arange(len(y_test)), y_pred, label='Predicted Trend')
    plt.xlabel('Data Index')
    plt.ylabel('Trend')
    plt.title(f'{name}: Actual Trend vs. Predicted Trend')
    plt.legend()
    plt.show()
    accuracy.append(r2)
    print()