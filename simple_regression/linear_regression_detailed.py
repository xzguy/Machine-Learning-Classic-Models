# ordinary least squares
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
x = 2.5 * np.random.randn(100) + 1.5
res = 0.5 * np.random.randn(100)
y = 2 + 0.3* x + res

df = pd.DataFrame(
    {'X': x,
    'Y': y}
)

print(df.head())

xmean = np.mean(x)
ymean = np.mean(y)
df['xycov'] = (df['X'] - xmean) * (df['Y'] - ymean)
df['xvar'] = (df['X'] - xmean)**2

beta = df['xycov'].sum() / df['xvar'].sum()
alpha = ymean - beta * xmean
print(f'alpha = {alpha}')
print(f'beta = {beta}')

ypred = alpha + beta * x

# plot linear regression against actual data
plt.figure(figsize=(12, 6))
plt.plot(x, ypred)
plt.plot(x, y, 'ro')
plt.title('Actual v.s. Predicted')
plt.xlabel('X')
plt.ylabel('Y')

# statsmodels
import statsmodels.formula.api as smf
model = smf.ols('Y ~ X', df)
trained_model = model.fit()
print('=== The model from statsmodels ===')
print(trained_model.params)
print('=== The model from sklearn =======')
smf_pred = trained_model.predict()

plt.figure(figsize=(12, 6))
plt.plot(df['X'], df['Y'], 'o')
plt.plot(df['X'], smf_pred, 'r', linewidth=2)
plt.xlabel('X')
plt.ylabel('Predicted from smf')
plt.show()

# scikit-learn
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
print(f'alpha = {lr.intercept_}')
print(f'betas = {lr.coef_}')