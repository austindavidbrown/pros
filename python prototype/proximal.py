import numpy as np
from sklearn.metrics import *

def prox(B, X_, y_, lambda_):
  n = X_.shape[0]
  p = X_.shape[1]
  max_iter = 1000
  tolerance = 10**(-8)
  I = np.array([i for i in range(0, p)])

  # Center
  C = np.identity(n) - 1/n * np.outer(np.ones(n),np.ones(n))
  cX = C @ X_
  cy = y_ - np.mean(y_)

  for j in range(0, max_iter):
    h_j = 10**(-(np.log(n)/np.log(10)))
    B_old = np.copy(B) # for stopping criterion
    DL_j = np.ones(p) 
    for i in np.random.permutation(I):
      DL_j[i] = -1 * cX[:, i].T @ (cy - (cX @ B)) # derivative of loss
      u_i = B[i] - h_j * DL_j[i]

      # Soft Thresholding
      if u_i < - h_j * lambda_:
        B[i] = u_i + h_j * lambda_
      if u_i >= - h_j * lambda_ and u_i <= h_j * lambda_:
        B[i] = 0
      if u_i > h_j * lambda_:
        B[i] = u_i - h_j * lambda_

    if np.linalg.norm(DL_j + lambda_ * np.sign(B)) < tolerance:
      return B
  return B

def predict(X, B, intercept):
  n = X.shape[0]
  return np.repeat(intercept, n) + X @ B


# Generate data
n = 1000
p = 10

A = np.array(np.repeat(3, p - int(1/2 * p))) # nonzero features
B_true = np.append(A, np.zeros(p - len(A))) # add sparsity
intercept_true = 5

X_train = np.zeros((n, p))
for i in range(0, n):
  X_train[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
err_train = np.random.multivariate_normal(np.zeros(n), np.identity(n)) # error 
y_train = np.repeat(intercept_true, n) + (X_train @ B_true) + err_train

n_test = int(n/3)
X_test = np.zeros((n_test, p))
for i in range(0, n_test):
  X_test[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
err_test = np.random.multivariate_normal(np.zeros(n_test), np.identity(n_test)) # error
y_test = np.repeat(intercept_true, n_test) + (X_test @ B_true) + err_test


### Single fit test
B = prox(np.zeros(p), X_train, y_train, .1)
B




