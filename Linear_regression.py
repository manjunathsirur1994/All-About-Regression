import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

x = np.arange(0,1,0.01)
# print(f'before: {x.shape}')
x = x.reshape(-1,1)
# print(x.shape)
y = weight * x + bias
# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# plt.scatter(x_test, y_test)
# plt.show()

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

# print(y_pred[:5])
# print(y_test[:5])

# plt.scatter(x_test, y_test)
# plt.scatter(x_test, y_pred)
# plt.show()

print(f'coef_: {model.coef_}')
print(f'intercept: {model.intercept_}')