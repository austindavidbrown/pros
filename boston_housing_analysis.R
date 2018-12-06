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

cv_glmnet_lasso = cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10, standardize = F)
mean((y_test - predict(cv_glmnet_lasso, X_test))^2)

cv_glmnet_elnet = cv.glmnet(X_train, y_train, alpha = 1/2, nfolds = 10, standardize = F)
mean((y_test - predict(cv_glmnet_elnet, X_test))^2)

alpha = c(1/2, 0, 0, 0, 0, 1/2)
lambdas = seq(10^(-3), 2, .1)
cv_pros = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 10, max_iter = 10000, tolerance = 10^(-3), random_seed = random_seed)
mean((y_test - predict(cv_pros, X_test))^2)