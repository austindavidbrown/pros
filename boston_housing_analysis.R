devtools::install("R-package")
library(pros)

library(glmnet)
library(MASS)

random_seed = 8989
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

# Lasso
cv_glmnet_lasso = cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10, standardize = F)
mean((y_test - predict(cv_glmnet_lasso, X_test))^2)

# Untuned ElasticNet
cv_glmnet_elnet = cv.glmnet(X_train, y_train, alpha = 1/2, nfolds = 10, standardize = F)
mean((y_test - predict(cv_glmnet_elnet, X_test))^2)

# Tune ElasticNet
alphas = c(1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10)
risks = c()
for (alpha in alphas) {
  cv_glmnet_elnet = cv.glmnet(X_train, y_train, alpha = alpha, nfolds = 10, standardize = F)
  mse = mean((y_test - predict(cv_glmnet_elnet, X_test))^2)
  risks = c(risks, mse)
}
I = which.min(risks)
alphas[I]
risks[I]

# Untuned 10th moment
alpha = c(1/2, 0, 0, 0, 0, 1/2)
lambdas = seq(10^(-3), 1, .1)
cv_pros = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 10, max_iter = 10000, tolerance = 10^(-3), random_seed = random_seed)
mean((y_test - predict(cv_pros, X_test))^2)

# Untuned 4th moment
alpha = c(1/2, 0, 1/2, 0, 0, 0)
lambdas = seq(10^(-3), 1, .1)
cv_pros2 = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 10, max_iter = 10000, tolerance = 10^(-3), random_seed = random_seed)
mean((y_test - predict(cv_pros2, X_test))^2)

# Use the ElasticNet Tuning for 4th moment
alpha = c(.25, 0, .75, 0, 0, 0)
lambdas = seq(10^(-3), 1, .1)
cv_pros3 = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 10, max_iter = 10000, tolerance = 10^(-3), random_seed = random_seed)
mean((y_test - predict(cv_pros3, X_test))^2)

# Tune 4th moment Pros
alphas = c(1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10)
risks = c()
for (alpha in alphas) {
  cv_pros = cv.pros(X_train, y_train, alpha = c(alpha, 0, 1 - alpha, 0, 0, 0), lambdas = lambdas, K_fold = 10, max_iter = 10000, tolerance = 10^(-3), random_seed = random_seed)
  mse = mean((y_test - predict(cv_pros, X_test))^2)
  risks = c(risks, mse)
}
I = which.min(risks)
alphas[I]
risks[I]
