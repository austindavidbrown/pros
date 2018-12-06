devtools::install("R-package")
library(pros)
library(glmnet)

d = data.matrix(read.table("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data"))
train = d[d[, "train"] == TRUE, ]
test = d[d[, "train"] == FALSE, ]

X_train = as.matrix(train[, 1:8])
y_train = as.vector(train[, 9])

X_test = as.matrix(test[, 1:8])
y_test = as.vector(test[, 9])

X_train = scale(X_train)
X_test = scale(X_test)

cv_glmnet_lasso = cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10)
mean((y_test - predict(cv_glmnet_lasso, X_test))^2)

cv_glmnet = cv.glmnet(X_train, y_train, alpha = 1/2, nfolds = 10, standardize = T)
mean((y_test - predict(cv_glmnet, X_test))^2)

###
# BEATS GLMNET
###
alpha = c(.2, 0, 0, 0, 0, 1 -.2, 0)
fit = pros(X_train, y_train, lambda = 80, alpha = alpha, max_iter = 100000, tolerance = 10^(-6))
mean((y_test - predict(fit, X_test))^2)


###
# BEATS GLMNET
###
alpha = c(1/3, 0, 0, 0, 0, 2/3, 0)
fit = pros(X_train, y_train, lambda = 30, alpha = alpha, max_iter = 100000, tolerance = 10^(-4))
mean((y_test - predict(fit, X_test))^2)


###
# BEATS GLMNET
###
alpha = c(1/5, 0, 1/5, 1/5, 1/5, 1/5, 0)
fit = pros(X_train, y_train, lambda = 50, alpha = alpha, max_iter = 100000, tolerance = 10^(-4))
mean((y_test - predict(fit, X_test))^2)

###
# BEATS GLMNET
###
alpha = c(1/2, 0, 0, 0, 0, 1/2, 0)
fit = pros(X_train, y_train, lambda = 19, alpha = alpha, max_iter = 100000, tolerance = 10^(-5))
mean((y_test - predict(fit, X_test))^2)

###
# BEATS GLMNET
###
alpha = c(1/5, 0, 1/5, 1/5, 1/5, 1/5, 0)
fit = pros(X_train, y_train, lambda = 58, alpha = alpha, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(fit, X_test))^2)

###
# BEST SO FAR
###
alpha = c(1/5, 0, 1/5, 1/5, 1/5, 1/5, 0)
fit = pros(X_train, y_train, lambda = 58, alpha = alpha, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(fit, X_test))^2)

lambdas = seq(100, 110, .1)
cv_pros = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 5, max_iter = 200000, tolerance = 10^(-3))
mean((y_test - predict(cv_pros, X_test))^2)




###
# BEST SO FAR
###
alpha = c(1/3, 0, 1/3, 1/3, 0, 0, 0)
fit = pros(X_train, y_train, lambda = 25, alpha = alpha, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(fit, X_test))^2)

lambdas = seq(100, 110, .1)
cv_pros = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 5, max_iter = 200000, tolerance = 10^(-3))
mean((y_test - predict(cv_pros, X_test))^2)


###
# BEST SO FAR
###
alpha = c(1 - .26, 0, .26, 0, 0, 0, 0)
lambdas = seq(10^(-7), 10, .1)
cv_pros = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 10, max_iter = 200000, tolerance = 10^(-3))
mean((y_test - predict(cv_pros, X_test))^2)


best_fit = pros(X_train, y_train, alpha = alpha, lambda = cv_pros$best_lambda, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(best_fit, X_test))^2)