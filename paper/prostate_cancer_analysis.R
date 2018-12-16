devtools::install_github("austindavidbrown/pros/R-package")
library(pros)
library(glmnet)

random_seed = 0
set.seed(random_seed)

d = data.matrix(read.table("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data"))
train = d[d[, "train"] == TRUE, ]
test = d[d[, "train"] == FALSE, ]

X_train = as.matrix(train[, 1:8])
y_train = as.vector(train[, 9])

X_test = as.matrix(test[, 1:8])
y_test = as.vector(test[, 9])

# Standardize the data
X_train = scale(X_train)
X_test = scale(X_test)

###
# Compare defaults
###
# Lasso (glmnet)
cv_glmnet_lasso = cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10, standardize = F)
mean((y_test - predict(cv_glmnet_lasso, X_test))^2)

# Lasso (pros)
cv_pros_lasso = cv.pros(X_train, y_train, alpha = c(1, 0, 0, 0, 0, 0), K_fold = 10, step_size = 1/35)
mean((y_test - predict(cv_pros_lasso, X_test))^2)

# Elnet (glmnet)
cv_glmnet_elnet = cv.glmnet(X_train, y_train, alpha = 1/2, nfolds = 10, standardize = F)
mean((y_test - predict(cv_glmnet_elnet, X_test))^2)

# Elnet (pros)
cv_pros_elnet = cv.pros(X_train, y_train, alpha = c(1/2, 1/2, 0, 0, 0, 0), K_fold = 10, step_size = 1/55)
mean((y_test - predict(cv_pros_elnet, X_test))^2)

# 4th moment (pros)
cv_pros_4_moment = cv.pros(X_train, y_train, alpha = c(1/2, 0, 1/2, 0, 0, 0), K_fold = 10, step_size = 1/55)
mean((y_test - predict(cv_pros_4_moment, X_test))^2)

# 10th moment (pros)
cv_pros_10_moment = cv.pros(X_train, y_train, alpha = c(1/2, 0, 0, 0, 0, 1/2), K_fold = 10, step_size = 1/500)
mean((y_test - predict(cv_pros_10_moment, X_test))^2)

###
# Compare tuned
###

# 4th moment best
#[1] 11
#[1] 0.96
#[1] 0.4441759

# 10th moment best
#[1] 11
#[1] 0.96
#[1] 0.4441322

# Elnet best
#[1] 11
#[1] 0.98
#[1] 0.4442702

for (i in 8:30) {
  # Tune Pros
  alphas = seq(.5, .99, .01)
  lambdas = seq(i, i + 10, .1)
  risks = c()
  for (alpha in alphas) {
    cv = cv.pros(X_train, y_train, alpha = c(alpha, 1 - alpha, 0, 0, 0,  0), lambdas = lambdas, K_fold = 10, step_size = 1/100)
    mse = mean((y_test - predict(cv, X_test))^2)
    risks = c(risks, mse)
  }
  I = which.min(risks)
  print(i)
  print(alphas[I])
  print(risks[I])
}















