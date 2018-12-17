devtools::install_github("austindavidbrown/pros/R-package")
library(pros)

library(glmnet)
library(MASS)

random_seed = 0
set.seed(random_seed)

data(Boston)
d = data.matrix(Boston)
I = sample(nrow(d), nrow(d)*0.80)
train = d[I,]
test = d[-I,]

X_train = train[, 1:13]
y_train = as.vector(train[, 14])

X_test = test[, 1:13]
y_test = as.vector(test[, 14])

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
cv_pros_lasso = cv.pros(X_train, y_train, alpha = c(1, 0, 0, 0, 0, 0), K_fold = 10, step_size = 1/500)
mean((y_test - predict(cv_pros_lasso, X_test))^2)

# Elnet (glmnet)
cv_glmnet_elnet = cv.glmnet(X_train, y_train, alpha = 1/2, nfolds = 10, standardize = F)
mean((y_test - predict(cv_glmnet_elnet, X_test))^2)

# Elnet (pros)
cv_pros_elnet = cv.pros(X_train, y_train, alpha = c(1/2, 1/2, 0, 0, 0, 0), K_fold = 10, step_size = 1/500)
mean((y_test - predict(cv_pros_elnet, X_test))^2)

# 4th moment (pros)
cv_pros_4_moment = cv.pros(X_train, y_train, alpha = c(1/2, 0, 1/2, 0, 0, 0), K_fold = 10, step_size = 1/500)
mean((y_test - predict(cv_pros_4_moment, X_test))^2)

# 10th moment (pros)
cv_pros_10_moment = cv.pros(X_train, y_train, alpha = c(1/2, 0, 0, 0, 0, 1/2), K_fold = 10, step_size = 1/3000)
mean((y_test - predict(cv_pros_10_moment, X_test))^2)

###
# Compare tuned
###

# elnet best
# [1] 51
# [1] 0.1
# [1] 25.52578
fit = pros(X_train, y_train, alpha = c(.1, 1 - .1, 0, 0, 0, 0), lambda = 51, step_size = 1/1000)
fit$B
mean((y_test - predict(fit, X_test))^2)

# 4th moment best
# [1] 16
# [1] 0.6
# [1] 25.44149
fit = pros(X_train, y_train, alpha = c(.6, 0, 1 - .6, 0, 0, 0), lambda = 16, step_size = 1/1000)
fit$B
mean((y_test - predict(fit, X_test))^2)

# 10th moment best
# [1] 0.1
# [1] 0.9
# [1] 25.68279
fit = pros(X_train, y_train, alpha = c(.9, 0, 0, 0, 0,  1 - .9), lambda = .1, step_size = 1/1000)
fit$B
mean((y_test - predict(fit, X_test))^2)

# Tune
for (i in 8:20) {
  # Tune Pros
  alphas = seq(.1, .99, .1)
  lambdas = seq(i, i + 1, .1)
  risks = c()
  for (alpha in alphas) {
    cv = cv.pros(X_train, y_train, alpha = c(alpha, 0, 0, 1 - alpha, 0,  0), lambdas = lambdas, K_fold = 10, step_size = 1/1000)
    mse = mean((y_test - predict(cv, X_test))^2)
    risks = c(risks, mse)
  }
  I = which.min(risks)
  print(i)
  print(alphas[I])
  print(risks[I])
}






