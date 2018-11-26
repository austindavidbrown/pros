import numpy as np

####
# ElasticNet
####

###
# CV
###

# Returns matrix of Betas
# Each row is a Beta for the specified lambda
def warm_start_B_matrix(X_, y_, alpha, lambdas__):
  n = X_.shape[0]
  p = X_.shape[1]
  L = len(lambdas__)
  lambdas_ = -np.sort(-lambdas__) # Order lambdas from largest to smallest (sparest to densest)
  B_matrix = np.zeros((L, p))

  # do the first one normally
  B_0 = np.zeros(p)
  B_matrix[0, :] = coord_descent(B_0, X_, y_, alpha, lambdas_[0])

  # Warm start
  for l in range(1, L):
    B_0 = B_matrix[(l - 1), :]
    lambda_ = lambdas_[l]
    B_matrix[l, :] = coord_descent(B_0, X_, y_, alpha, lambda_)  
  return B_matrix

def cross_validation_risks(X, y, K, alpha, lambdas_):
  N = X.shape[0]
  L = len(lambdas_)
  test_risks_matrix = np.zeros((L, K))

  I = np.random.permutation([i for i in range(0, N)])
  partitions = np.array_split(I, K)
  for k in range(0, K):
    partition = partitions[k]
    TRAIN = np.delete(I, partition)
    TEST = partition
    X_train = X[TRAIN, :]
    y_train = y[TRAIN]

    X_test = X[TEST, :]
    y_test = y[TEST]
    # compute error
    intercept = np.mean(y_train)
    B_matrix = warm_start_B_matrix(X_train, y_train, alpha, lambdas_)
    for l in range(0, L):
      B = B_matrix[l, :]
      test_risks_matrix[l, k] = np.linalg.norm(y_test - predict(X_test, B, intercept))

  return { "risks": np.mean(test_risks_matrix, axis = 1), "lambdas_": -np.sort(-lambdas_) } 

###
# Optimization
###
def soft_threshold(x, lambda_):
  if x > lambda_:
    return x - lambda_
  if x < -lambda_:
    return x + lambda_
  else:
    return 0

# random permutation coordinate descent
def coord_descent(B, X_, y_, alpha, lambda_):
  max_iter = 10000
  tolerance = 10**(-6)

  # Center
  N = X_.shape[0]
  C = np.identity(N) - 1/N * np.outer(np.ones(N),np.ones(N))
  cX = C @ X_
  cy = y_ - np.mean(y_)

  I = np.array([i for i in range(0, p)])
  for j in range(0, max_iter):
    B_old = np.copy(B)
    for i in np.random.permutation(I):
      cX_minus_i = cX[:, np.delete(I, i)]
      B_minus_i = B[np.delete(I, i)]
      cX_i = cX[:, i]

      # update
      B[i] = soft_threshold((1 / (cX_i.T @ cX_i + lambda_ * (1 - alpha))) * cX_i @ (y_ - (cX_minus_i @ B_minus_i)), (1 / (cX_i.T @ cX_i + lambda_ * (1 - alpha))) * alpha * lambda_)

    # stop if loss is not changing much
    if (np.abs(np.linalg.norm(cy - cX @ B) - np.linalg.norm(cy - cX @ B_old)) < tolerance):
      return B
  return B

def predict(X, B, intercept):
  n = X.shape[0]
  return np.repeat(intercept, n) + X @ B


# Generate data
n = 100
p = 100

A = np.array(np.repeat(3, p - int(2/3 * p))) # nonzero features
B_true = np.append(A, np.zeros(p - len(A))) # add sparsity
intercept_true = 5

# Correlation matrix
rho = 1/2
E = np.zeros((p, p))
for i in range(0, p):
  for j in range(0, p):
    E[i, j] = rho ** np.abs(j - i)

X_train = np.zeros((n, p))
for i in range(0, n):
  X_train[i, :] = np.random.multivariate_normal(np.zeros(p), E)
err_train = np.random.multivariate_normal(np.zeros(n), np.identity(n)) # error
y_train = np.repeat(intercept_true, n) + (X_train @ B_true) + err_train

n_test = int(n/3)
X_test = np.zeros((n_test, p))
for i in range(0, n_test):
  X_test[i, :] = np.random.multivariate_normal(np.zeros(p), E)
err_test = np.random.multivariate_normal(np.zeros(n_test), np.identity(n_test)) # error
y_test = np.repeat(intercept_true, n_test) + (X_test @ B_true) + err_test

# CV
lambdas_ = np.arange(10**(-3), 1, .1)
K_fold = 5

# Lasso
alpha = 1
cv = cross_validation_risks(X_train, y_train, K_fold, alpha, lambdas_)
best_lambda =  cv["lambdas_"][np.argmin(cv["risks"])]

B_lasso = coord_descent(np.zeros(p), X_train, y_train, alpha, best_lambda)
intercept = np.mean(y_train)

# train error
np.linalg.norm(y_train - predict(X_train, B_lasso, intercept))
# test error
np.linalg.norm(y_test - predict(X_test, B_lasso, intercept))

# ElasticNet
alpha = 1/2
cv = cross_validation_risks(X_train, y_train, K_fold, alpha, lambdas_)
best_lambda =  cv["lambdas_"][np.argmin(cv["risks"])]

B_elast = coord_descent(np.zeros(p), X_train, y_train, alpha, best_lambda)

intercept = np.mean(y_train)

# train error
np.linalg.norm(y_train - predict(X_train, B_elast, intercept))

# test error
np.linalg.norm(y_test - predict(X_test, B_elast, intercept))


# single fit
alpha = 1/2
lambda_ = .01
B_0 = np.zeros(p)
intercept = np.mean(y_train)
B_elnet = coord_descent(B_0, X_train, y_train, alpha, lambda_)
B_elnet

