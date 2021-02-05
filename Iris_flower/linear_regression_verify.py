# ordinary least squares
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import os

full_path = os.path.realpath(__file__)
cur_dir = os.path.dirname(full_path)

# Load dataset
url = os.path.join(cur_dir, "iris.csv")
attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_class']
dataset = pd.read_csv(url, names = attributes)
dataset['iris_class'] = dataset['iris_class'].astype('category').cat.codes

print(dataset.head())

# statsmodels
model = smf.ols(formula='iris_class ~ sepal_length + sepal_width + petal_length + petal_width', data=dataset)
trained_model = model.fit()
print('=== The model from statsmodels ===')
print(trained_model.params)

# scikit-learn
array = dataset.values
X = array[ : , 0:4]
Y = array[ : , 4]
print('=== The model from sklearn =======')
lr = LinearRegression()
lr.fit(X, Y)
print(f'alpha = {lr.intercept_}')
print(f'betas = {lr.coef_}')