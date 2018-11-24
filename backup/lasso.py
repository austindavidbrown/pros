import numpy as np

# TODO intercept

###
# CV
###

# Returns matrix of Betas
# Each row is a Beta for the specified lambda
def warm_start_B_matrix(X, y, lambdas_):
  p = X.shape[1]
  B_0 = np.zeros(p) # initial value
  lambdas_ = -np.sort(-lambdas_) # Order lambdas from largest to smallest (sparest to densest)
  B_matrix = np.zeros((len(lambdas_), len(B_0)))
  for i in range(0, len(lambdas_)):
    # warm start
    if i > 0:
      B_0 = B_matrix[(i - 1), :]
    B_matrix[i, :] = coord_descent(lambdas_[i], B_0, X, y)  
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
    B_matrix = warm_start_B_matrix(X_train, y_train, lambdas_)
    for l in range(0, L):
      B = B_matrix[l, :]
      test_risks_matrix[l, k] = np.linalg.norm(y_test - (X_test @ B))

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
def coord_descent(lambda_, B, X, y):
  max_iter = 10000
  tolerance = 10**(-6)

  I = np.array([i for i in range(0, p)])
  for j in range(0, max_iter):
    B_old = np.copy(B)
    for i in np.random.permutation(I):
      X_minus_i = X[:, np.delete(I, i)]
      B_minus_i = B[np.delete(I, i)]
      X_i = X[:, i]

      # update
      B[i] = soft_threshold(1/ X_i.dot(X_i) * X_i.dot(y - X_minus_i.dot(B_minus_i)), 1/ X_i.dot(X_i) * lambda_)

    # stop if loss is not changing much
    if (np.abs(np.linalg.norm(y - X @ B) - np.linalg.norm(y - X @ B_old)) < tolerance):
      return B
  return B

def predict(X, B)
  return X @ B


# Generate data
n = 50
p = 10
X = np.zeros((n, p))
for i in range(0, n):
  X[i, :] = np.random.multivariate_normal(np.zeros(p), np.identity(p))
C = np.array([3, 1])
B_true = np.append(C, np.zeros(p - len(C)))
y = X @ B_true

B_0 = np.zeros(p)
B = coord_descent(.001, B_0, X, y)

# CV
lambdas_ = np.arange(10**(-3), 1, .1)
cv = cross_validation_risks(X, y, 5, lambdas_)
M = np.argmin(cv["risks"])
best_lambda =  cv["lambdas_"][M]
best_risk = cv["risks"][M]
coord_descent(best_lambda, B_0, X, y)



N = 2
C = np.identity(N) - 1/N * np.outer(np.ones(N),np.ones(N))
C @ np.matrix([[1, 2], [2, 3]])

