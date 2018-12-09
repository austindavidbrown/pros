devtools::install_github("austindavidbrown/pros/R-package")
library(pros)
library(glmnet)

random_seed = 8989
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


write.table(X_train,file="prostate_X_train.csv",row.names=FALSE, col.names=FALSE, sep = ",")
write.table(y_train,file="prostate_y_train.csv",row.names=FALSE, col.names=FALSE, sep = ",")
write.table(X_test,file="prostate_X_test.csv",row.names=FALSE, col.names=FALSE, sep = ",")
write.table(y_test,file="prostate_y_test.csv",row.names=FALSE, col.names=FALSE, sep = ",")

# Lasso (glmnet)
cv_glmnet_lasso = cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10, standardize = F)
mean((y_test - predict(cv_glmnet_lasso, X_test))^2)

coef(cv_glmnet_lasso)
fit = pros(X_train, y_train, alpha = c(1, 0, 0, 0, 0, 0), lambda = cv_glmnet_lasso$min.lambda, max_iter = 100000, tolerance = 10^(-7), random_seed = random_seed)
mean((y_test - predict(fit, sX_test))^2)
fit$B

# "Tuned" ElasticNet (glmnet)
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


lambdas = seq(10^(-3), 1, .01)
lambdas
cv = cv.pros(X_train, y_train, alpha = c(1, 0, 0, 0, 0, 0), lambdas = lambdas, K_fold = 10, max_iter = 10000, tolerance = 10^(-3))
mean((y_test - predict(cv, X_test))^2)

fit = pros(X_train, y_train, alpha = c(1, 0, 0, 0, 0, 0), lambda = 0.0100293, max_iter = 10000, tolerance = 10^(-3), random_seed = random_seed)
mean((y_test - predict(fit, X_test))^2)

# Tune Pros
alphas = c(1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10)
risks = c()
for (alpha in alphas) {
  lambdas = seq(10^(-3), 10, .1)
  cv = cv.pros(X_train, y_train, alpha = c(alpha, 1/2*(1 - alpha), 1/2*(1 - alpha), 0, 0, 0), lambdas = lambdas, K_fold = 10, max_iter = 10000, tolerance = 10^(-3))
  mse = mean((y_test - predict(cv, X_test))^2)
  risks = c(risks, mse)
}
I = which.min(risks)
alphas[I]
risks[I]


fit = pros(X_train, y_train, lambda = cv$best_lambda, alpha = c(1/2, 1/4, 1/4, 0, 0, 0), max_iter = 10000, tolerance = 10^(-3))

fit_elnet = cv.glmnet(X_train, y_train, alpha = 1/2, standardize = F)
coef(fit_elnet)
























###
# Pros
###
alpha = c(.2, 0, 0, 0, 0, 1 -.2)
fit = pros(X_train, y_train, lambda = 54.02, alpha = alpha, max_iter = 10000, tolerance = 10^(-3))
mean((y_test - predict(fit, X_test))^2)

alpha = c(1/5, 0, 1/5, 1/5, 1/5, 1/5)
fit = pros(X_train, y_train, lambda = 54, alpha = alpha, max_iter = 10000, tolerance = 10^(-3))
mean((y_test - predict(fit, X_test))^2)


alpha = c(1/5, 0, 1/5, 1/5, 1/5, 1/5)
lambdas = seq(10^(-3), 5, .5)
cv = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 10, max_iter = 10000, tolerance = 10^(-3))
mean((y_test - predict(cv, X_test))^2)


























alpha = c(1/2, 0, 0, 0, 0, 1/2)
fit = pros(X_train, y_train, lambda = 21.05, alpha = alpha, max_iter = 100000, tolerance = 10^(-5))
mean((y_test - predict(fit, X_test))^2)










###
# BEATS GLMNET
###
alpha = c(.2, 0, 0, 0, 0, 1 -.2)
fit = pros(X_train, y_train, lambda = 80, alpha = alpha, max_iter = 100000, tolerance = 10^(-6))
mean((y_test - predict(fit, X_test))^2)


###
# BEATS GLMNET
###
alpha = c(1/3, 0, 0, 0, 0, 2/3)
fit = pros(X_train, y_train, lambda = 30, alpha = alpha, max_iter = 100000, tolerance = 10^(-4))
mean((y_test - predict(fit, X_test))^2)


###
# BEATS GLMNET
###
alpha = c(1/5, 0, 1/5, 1/5, 1/5, 1/5)
fit = pros(X_train, y_train, lambda = 50, alpha = alpha, max_iter = 100000, tolerance = 10^(-4))
mean((y_test - predict(fit, X_test))^2)

###
# BEATS GLMNET
###
alpha = c(1/2, 0, 0, 0, 0, 1/2)
fit = pros(X_train, y_train, lambda = 19, alpha = alpha, max_iter = 100000, tolerance = 10^(-5))
mean((y_test - predict(fit, X_test))^2)

###
# BEATS GLMNET
###
alpha = c(1/5, 0, 1/5, 1/5, 1/5, 1/5)
fit = pros(X_train, y_train, lambda = 58, alpha = alpha, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(fit, X_test))^2)

###
# BEST SO FAR
###
alpha = c(1/5, 0, 1/5, 1/5, 1/5, 1/5)
fit = pros(X_train, y_train, lambda = 58, alpha = alpha, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(fit, X_test))^2)

lambdas = seq(100, 110, .1)
cv_pros = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 5, max_iter = 200000, tolerance = 10^(-3))
mean((y_test - predict(cv_pros, X_test))^2)




###
# BEST SO FAR
###
alpha = c(1/3, 0, 1/3, 1/3, 0, 0)
fit = pros(X_train, y_train, lambda = 25, alpha = alpha, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(fit, X_test))^2)

lambdas = seq(100, 110, .1)
cv_pros = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 5, max_iter = 200000, tolerance = 10^(-3))
mean((y_test - predict(cv_pros, X_test))^2)


###
# BEST SO FAR
###
alpha = c(1 - .26, 0, .26, 0, 0, 0)
lambdas = seq(10^(-7), 10, .1)
cv_pros = cv.pros(X_train, y_train, alpha = alpha, lambdas = lambdas, K_fold = 10, max_iter = 200000, tolerance = 10^(-3))
mean((y_test - predict(cv_pros, X_test))^2)


best_fit = pros(X_train, y_train, alpha = alpha, lambda = cv_pros$best_lambda, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(best_fit, X_test))^2)