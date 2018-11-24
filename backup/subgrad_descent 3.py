import numpy as np

###
# Subgradient Descent
# Coordinate descent
# step size is a really sensitive (be careful)
###
def subgrad_coord_descent(X_, y_, lambda_):
  n = X_.shape[0]
  p = X_.shape[1]
  max_iter = 10000
  tolerance = 10**(-5)
  I = np.array([i for i in range(0, p)])
  B = np.zeros(p) # starting point

  # Center
  C = np.identity(n) - 1/n * np.outer(np.ones(n),np.ones(n))
  cX = C @ X_
  cy = y_ - np.mean(y_)

  for j in range(0, max_iter):
    s = 10**(-(np.log(n)/np.log(10))) # step size of the same order
    B_old = np.copy(B) # this is current B; keep track for stopping criterion
    for i in np.random.permutation(I):
      B_current = B[i] # current B[i]
      cX_i = cX[:, i]
      D_Li = -1 * cX_i.T @ (cy - (cX @ B))
      sub_grad_l1 = np.sign(B_current)

      # update
      # handle subgradient cases of l1
      if sub_grad_l1 != 0:
        B[i] = B_current - s * (D_Li + lambda_ * sub_grad_l1)
      elif sub_grad_l1 == 0:
        if D_Li < -lambda_:
          B[i] = B_current - s * (D_Li + lambda_)
        if D_Li > lambda_:
          B[i] = B_current - s * (D_Li - lambda_)
        if D_Li >= -lambda_ and D_Li <= lambda_:
          B[i] = 0

    # stop if loss is not changing much
    if (np.abs(np.linalg.norm(cy - cX @ B) - np.linalg.norm(cy - cX @ B_old)) < tolerance):
      print(j)
      return B
  return B

def sparsify(B):
  tolerance = .01
  for i in range(0, len(B)):
    if B[i] < tolerance:
      B[i] = 0.0
  return B

def predict(X, B, intercept):
  n = X.shape[0]
  return np.repeat(intercept, n) + X @ B



# Generate data
n = 1000
p = 100

A = np.array(np.repeat(3, p - int(1/2 * p))) # nonzero features
B_true = np.append(A, np.zeros(p - len(A))) # add sparsity
intercept_true = 5

X_train = np.zeros((n, p))
for i in range(0, n):
  X_train[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
y_train = np.repeat(intercept_true, n) + (X_train @ B_true)

n_test = int(n/3)
X_test = np.zeros((n_test, p))
for i in range(0, n_test):
  X_test[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
y_test = np.repeat(intercept_true, n_test) + (X_test @ B_true)

intercept = np.mean(y_train)
B_lasso = sparsify(subgrad_coord_descent(X_train, y_train, .001))
B_lasso

# train error
np.linalg.norm(y_train - predict(X_train, B_lasso, intercept))

# test error
np.linalg.norm(y_test - predict(X_test, B_lasso, intercept))




