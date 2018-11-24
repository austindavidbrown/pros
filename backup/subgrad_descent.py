import numpy as np

def subgrad_desc(lambda_, X, y):
  p = X.shape[1]
  B = np.zeros(p) # starting point
  max_iter = 1000
  tolerance = 10**(-6)
  I = np.array([i for i in range(0, p)])

  for j in range(0, max_iter):
    s = (j + 1)**(-1/2) # diminishing step size
    B_old = np.copy(B)
    for i in I:
      # update
      X_i = X[:, i]
      D_Li = -1 * X_i.T @ (y - (X @ B_old))
      sub_grad = np.sign(B_old[i])
      if sub_grad != 0:
        B[i] = B_old[i] - s * (D_Li + lambda_ * sub_grad)
      elif sub_grad == 0:
        if D_Li < -lambda_:
          B[i] = B_old[i] - s * (D_Li + lambda_)
        if D_Li > lambda_:
          B[i] = B_old[i] - s * (D_Li - lambda_)
        if D_Li <= lambda_ and D_Li >= -lambda_:
          B[i] = 0
  return B

def sparsify(B):
  tolerance = .01
  for i in range(0, len(B)):
    if B[i] < tolerance:
      B[i] = 0.0
  return B


n = 5
p = 4
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
x3 = np.random.normal(0, 1, n)
x4 = np.random.normal(0, 1, n)

X = np.column_stack((x1, x2, x3, x4))
y = 3 * x1 + 0 + x3 + 0

sparsify(subgrad_desc(.1, X, y))
