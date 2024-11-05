# Linear regression with sklearn's Linear regression
# ---------------------------------------------------------

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

weight = 0.7
bias = 0.3

x = np.arange(0,1,0.01)
print(f'shape of x before reshaping: {x.shape}')
x = x.reshape(-1,1)
print(f"shape of x after reshaping: {x.shape}")
y = weight * x + bias
print(f"printing samples of x: {x[:5]}")
print(f"printing samples of y: {y[:5]}")

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
print(f"shape of x_train: {x_train.shape}")
print(f"shape of y_train: {y_train.shape}")
print(f"shape of x_test: {x_test.shape}")
print(f"shape of y_test: {y_test.shape}")

# plotting x_test vs y_test
# plt.scatter(x_test, y_test)
# plt.show()

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(f"printing y_pred samples to compare predictions: {y_pred[:5]}")
print(f"printing y_test samples to compare predictions: {y_test[:5]}")

# Plotting test vs preds
# plt.scatter(x_test, y_test)
# plt.scatter(x_test, y_pred)
# plt.show()

print(f'coef_: {model.coef_}')
print(f'intercept: {model.intercept_}')

print(f"printing MSE: {int(mean_squared_error(y_test, y_pred))}")