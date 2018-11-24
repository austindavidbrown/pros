import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.datasets import make_regression
from sklearn.metrics import *
from numpy.linalg import norm

p = 5
mu = np.repeat(0.0, p)
E = np.identity(p)
X = np.random.multivariate_normal(mu, E, n)

n = 1000
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
x3 = np.random.normal(0, 1, n)
x4 = np.random.normal(0, 1, n)
y = 3 * x1 + 0 + 10 * x3 + 0
X = np.column_stack((x1, x2, x3, x4))

model = LassoCV(cv=5)
model.fit(X, y)
print(np.sqrt(mean_squared_error(y, model.predict(X))))

model.coef_
model.alpha_

plt.figure()
plt.plot(-np.log10(model.alphas_), model.mse_path_)
plt.show()

lasso_model = Lasso(alpha = model.alpha_)
lasso_model.fit(X, y)
print(np.sqrt(mean_squared_error(y, lasso_model.predict(X))))

from sklearn.linear_model import Lasso, LassoCV, ElasticNet
elnet_model = ElasticNet(alpha = model.alpha_, l1_ratio = 1)
elnet_model.fit(X, y)

elnet_model.coef_
model.alpha_
print(np.sqrt(mean_squared_error(y, elnet_model.predict(X))))
