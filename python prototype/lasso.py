import numpy as np

###
# CV
###

# Returns matrix of Betas
# Each row is a Beta for the specified lambda
def warm_start_B_matrix(X_, y_, lambdas__):
  n = X_.shape[0]
  p = X_.shape[1]
  L = len(lambdas__)
  lambdas_ = -np.sort(-lambdas__) # Order lambdas from largest to smallest (sparest to densest)
  B_matrix = np.zeros((L, p))

  # do the first one normally
  B_0 = np.zeros(p)
  B_matrix[0, :] = coord_descent(lambdas_[0], B_0, X_, y_)

  # Warm start
  for l in range(1, L):
    B_0 = B_matrix[(l - 1), :]
    lambda_ = lambdas_[l]
    B_matrix[l, :] = coord_descent(lambda_, B_0, X_, y_)  
  return B_matrix

def cross_validation_risks(X, y, K, lambdas_):
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
    B_matrix = warm_start_B_matrix(X_train, y_train, lambdas_)
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
def coord_descent(lambda_, B, X_, y_):
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
      B[i] = soft_threshold((1 / (cX_i.T @ cX_i)) * cX_i @ (y_ - (cX_minus_i @ B_minus_i)), (1 / (cX_i.T @ cX_i)) * lambda_)

    # stop if loss is not changing much
    if (np.abs(np.linalg.norm(cy - cX @ B) - np.linalg.norm(cy - cX @ B_old)) < tolerance):
      return B
  return B

def predict(X, B, intercept):
  n = X.shape[0]
  return np.repeat(intercept, n) + X @ B


# Generate data
n = 50
p = 100
X = np.zeros((n, p))
for i in range(0, n):
  X[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
C = np.array([3, 1, 5, 5]) # nonzero features
B_true = np.append(C, np.zeros(p - len(C))) # add sparsity
y = np.repeat(5, n) + X @ B_true

B_0 = np.zeros(p)
intercept = np.mean(y)
B = coord_descent(.01, B_0, X, y)
B

np.linalg.norm(y - (np.repeat(intercept, n) + X @ B))

# CV
lambdas_ = np.arange(10**(-3), 1, .1)

cv = cross_validation_risks(X, y, 5, lambdas_)
M = np.argmin(cv["risks"])
best_lambda =  cv["lambdas_"][M]
best_risk = cv["risks"][M]

intercept = np.mean(y)
B = coord_descent(best_lambda, np.zeros(p), X, y)

np.linalg.norm(y - predict(X, B, intercept))

