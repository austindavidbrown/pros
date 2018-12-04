devtools::install("R-package")
library(pros)
library(glmnet)

d = data.matrix(read.table("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data"))
head(d)
train = d[d[, "train"] == TRUE, ]
test = d[d[, "train"] == FALSE, ]

X_train = as.matrix(train[, 1:8])
y_train = as.vector(train[, 9])

X_test = as.matrix(test[, 1:8])
y_test = as.vector(test[, 9])

intercept = mean(y_train)

# Center
C = diag(1, nrow(X_train)) - 1/nrow(X_train) * rep(1, nrow(X_train)) %*% t(rep(1, nrow(X_train)))
cX_train = C %*% X_train
cy_train = y_train - rep(mean(y_train), length(y_train))


fit_glmnet = glmnet(cX_train, cy_train, alpha = 1, lambda = .1, intercept = F, standardize = F)
B_glmnet = coef(fit_glmnet)[2:9]
pred_glmnet = rep(intercept, nrow(X_test)) + X_test %*% B_glmnet
mean((y_test - pred_glmnet)^2)

cv_glmnet = cv.glmnet(cX_train, cy_train, alpha = 1, standardize = T)
coef(cv_glmnet)
mean((y_test - predict(cv_glmnet, X_test))^2)

fit = pros(cX_train, cy_train, lambda = 0.01456746, max_iter = 100000, tolerance = 10^(-3))
fit
mean((y_test - predict(fit, X_test))^2)

cv_pros = cv.pros(cX_train, cy_train, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(cv_pros, X_test))^2)

best_fit = pros(cX_train, cy_train, lambda = cv_pros$best_lambda, max_iter = 100000, tolerance = 10^(-3))
mean((y_test - predict(best_fit, X_test))^2)


